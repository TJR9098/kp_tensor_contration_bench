#!/usr/bin/env python3
"""Rerun only PyTorch-KML on the unchanged historical 120-case matrix."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import run_tblis_batch_experiment as old

ROOT=Path(__file__).resolve().parent
METHOD='pytorch_2.5.0_kml_multi'
DEFAULT_BASELINE=ROOT/'results/tblis_batch_cpu0-31_20261011T074035858907Z'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def read_rows(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))


def openblas_audit(baseline):
    code='''import torch,json,hashlib
torch.set_num_threads(1)
x=torch.ones(128,64);y=torch.ones(64,32);assert bool((x@y==64).all())
libs=sorted({line.split()[-1] for line in open('/proc/self/maps') if '.so' in line and '/' in line and any(k in line for k in ['libopenblas','libtorch_cpu','libkblas','libgomp','libomp'])})
print(json.dumps({'version':torch.__version__,'config_sha256':hashlib.sha256(torch.__config__.show().encode()).hexdigest(),'libraries':libs,'torch_path':torch.__file__}))
'''
    env=os.environ.copy()
    for name in ['LD_PRELOAD','LD_LIBRARY_PATH','OMP_TOOL','OMP_TOOL_LIBRARIES','BLAS_NUM_THREADS']:
        env.pop(name,None)
    env.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                               'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','GOTO_NUM_THREADS']})
    run=subprocess.run([str(old.PYTORCH_ROOT/'env-native/bin/python'),'-c',code],
                       env=env,capture_output=True,text=True,timeout=60,check=True)
    data=json.loads(run.stdout)
    previous=json.loads((baseline/'metadata/pytorch_2.5.0_openblas/XL_X0_B1_T32_c0.json').read_text())
    data['config_matches_history']=data['config_sha256']==previous['torch_config_sha256']
    data['no_kml_loaded']=not any('libkblas' in p for p in data['libraries'])
    if not data['config_matches_history'] or not data['no_kml_loaded']:
        raise RuntimeError('OpenBLAS impact audit failed; reassess rerun scope')
    data['binary_sha256']={p:sha(p) for p in data['libraries']}
    return data


def environment(case,observer=None):
    _,env=old.command_for(case,'pytorch_2.5.0_kml_patched',0,ROOT,False)
    env.pop('LD_PRELOAD',None);env.pop('OMP_TOOL_LIBRARIES',None)
    env.update(OMP_TOOL='disabled',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',
               BLAS_NUM_THREADS=str(int(case['threads']) if int(case['batch'])==1 else 1),
               PYTHONFAULTHANDLER='1',PYTHONDONTWRITEBYTECODE='1')
    if observer:
        env.update(OMP_TOOL='enabled',OMP_TOOL_LIBRARIES=str(observer),LD_PRELOAD=str(observer))
    return env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--case',action='append')
    p.add_argument('--timeout',type=int,default=300)
    args=p.parse_args()
    baseline=args.baseline.resolve()
    if not set(range(32))<=os.sched_getaffinity(0):raise RuntimeError('CPU0-31 required')
    quota=Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    if quota.exists() and int(quota.read_text())>0:
        cores=int(quota.read_text())/int((quota.parent/'cpu.cfs_period_us').read_text())
        if cores<32:raise RuntimeError(f'CPU quota {cores} <32')
    output=(args.output_dir or ROOT/'results'/('pytorch_kml_multi_cpu0-31_'+old.utc_stamp())).resolve()
    output.mkdir(parents=True,exist_ok=True)
    print('RESULTS '+str(output),flush=True)
    source_manifest=baseline/'case_manifest.tsv'
    cases=read_rows(source_manifest)
    if cases!=old.read_manifest(old.MANIFEST):raise RuntimeError('case parameters differ from history')
    histprov=json.loads((baseline/'provenance.json').read_text())
    if sha(ROOT/'benchmark_tblis_batch_pytorch.py')!=histprov['artifact_sha256']['pytorch_runner']:
        raise RuntimeError('shared runner changed; cannot reuse historical OpenBLAS automatically')
    library=(old.PYTORCH_ROOT/'hpckit-components/KunpengHPCKit-kml.26.1.RC1/bishenglib/sve/kblas/multi/libkblas.so.1.26.1.RC1').resolve()
    compiler=old.PYTORCH_ROOT/'hpckit-components/BiShengCompiler-5.2.0-aarch64-linux-mini-hpc'
    observer=ROOT/'bin/kml_batch_observer.so'
    observer.parent.mkdir(exist_ok=True)
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wall','-Wextra',
                    '-idirafter',str(compiler/'lib/clang/19/include'),str(ROOT/'kml_batch_observer.cpp'),
                    '-ldl','-o',str(observer)],check=True)
    names=['run_kml_multi_batch.py','benchmark_kml_multi_batch.py','kml_batch_observer.cpp',
           'benchmark_tblis_batch_pytorch.py','benchmark_tblis_pytorch.py']
    hashes={name:sha(ROOT/name) for name in names}
    provenance_path=output/'provenance.json'
    if provenance_path.exists():
        provenance=json.loads(provenance_path.read_text())
        if provenance['source_sha256']!=hashes or provenance['library_sha256']!=sha(library):
            raise RuntimeError('resume refused: sources/library changed')
    else:
        if any(output.iterdir()):raise RuntimeError('nonempty output without provenance')
        audit=openblas_audit(baseline)
        reference_hashes={}
        for sc in ['S','M','L','XL']:
            f=baseline/'references'/f'{sc}_B32.f64'
            expected=json.loads(f.with_suffix('.json').read_text())['sha256']
            digest=sha(f)
            if digest!=expected:raise RuntimeError(f'reference hash mismatch {sc}')
            reference_hashes[sc]=digest
        provenance={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'method':METHOD,
                    'baseline':str(baseline),'reference_dir':str(baseline/'references'),
                    'reference_sha256':reference_hashes,'source_sha256':hashes,
                    'library':str(library),'library_sha256':sha(library),'manifest_sha256':sha(source_manifest),
                    'campaigns':3,'warmup':10,'repeat':50,'host_affinity':sorted(os.sched_getaffinity(0)),
                    'openblas_before':audit,'openblas_rerun_required':False,
                    'policy':'B1: KML inner=T; B4/B32: existing ATen outer, KML inner=1; max active levels=1',
                    'environment_changes':'KML subprocess only; original OpenBLAS runner/env/binaries unchanged',
                    'historical_results_note':'OpenBLAS, locking KML and TBLIS are historical, not contemporaneous reruns'}
        dump(provenance_path,provenance)
        shutil.copy2(source_manifest,output/'case_manifest.tsv')
        (output/'lscpu.txt').write_text(subprocess.run(['lscpu'],capture_output=True,text=True,check=True).stdout)
        (output/'logs').mkdir();(output/'source_snapshot').mkdir()
        for name in names:shutil.copy2(ROOT/name,output/'source_snapshot'/name)
    (output/'status.txt').write_text('PREFLIGHT\n' if args.preflight_only else 'RUNNING\n')
    def invoke(case,campaign,mode):
        stem=f"{case['case_id']}_c{campaign}_{mode}"
        cmd=[str(old.PYTORCH_ROOT/'env-kml/bin/python'),str(ROOT/'benchmark_kml_multi_batch.py'),
             '--kml-library',str(library),'--manifest',str(output/'case_manifest.tsv'),
             '--case',case['case_id'],'--method',METHOD,'--campaign',str(campaign),
             '--output-dir',str(output),'--reference-dir',str(baseline/'references')]
        if mode=='ompt':
            cmd += ['--kml-observer',str(observer),'--kml-ompt-only']
            expected=[output/'kml_thread_diagnostics'/f"{case['case_id']}.json"]
        elif mode=='distinct':
            cmd += ['--kml-distinct-only']
            expected=[output/'distinct_validation'/f"{case['case_id']}.json"]
        else:
            trace=old.should_trace(case,campaign)
            if campaign==0:cmd+=['--diagnostics']
            if trace:cmd+=['--export-trace']
            expected=old.expected_paths(output,case,METHOD,campaign,trace)
        if all(f.exists() for f in expected):return False
        if any(f.exists() for f in expected):raise RuntimeError(f'partial task {stem}')
        env=environment(case,observer if mode=='ompt' else None)
        log=output/'logs'/f'{stem}.log'
        try:
            run=subprocess.run(cmd,env=env,cwd=ROOT,text=True,capture_output=True,timeout=args.timeout)
            log.write_text(run.stdout+'\n'+run.stderr)
        except subprocess.TimeoutExpired as error:
            log.write_text(str(error.stdout)+'\n'+str(error.stderr))
            raise RuntimeError(f'timeout {stem}; see {log}') from error
        if run.returncode:raise RuntimeError(f'failed {stem}: {run.returncode}; see {log}')
        if not all(f.exists() for f in expected):raise RuntimeError(f'missing task artifact {stem}')
        return True
    try:
        if args.preflight_only:
            selected=args.case or ['S_X0_B1_T1','S_X0_B1_T32','XL_X0_B1_T32',
                                   'XL_X4_B4_T32','XL_X4_B32_T32','S_X4_B32_T4']
            for cid in selected:
                case=next(c for c in cases if c['case_id']==cid)
                invoke(case,0,'ompt');invoke(case,0,'distinct')
                print('PREFLIGHT_PASS '+cid,flush=True)
            (output/'status.txt').write_text('PREFLIGHT_COMPLETE\n')
        else:
            # Full correctness preflight for each batch/layout/thread combination;
            # the random small tensors distinguish batch indices unlike old performance inputs.
            for case in cases:
                if case['scale']=='S':invoke(case,0,'distinct')
            tasks=[(case,campaign) for campaign in range(3) for case in cases]
            random.Random(20261013).shuffle(tasks)
            for index,(case,campaign) in enumerate(tasks,1):
                invoke(case,campaign,'formal')
                if campaign==0:invoke(case,campaign,'ompt')
                if index%5==0 or index==len(tasks):
                    print(json.dumps({'completed':index,'total':len(tasks),'output':str(output)}),flush=True)
            audit=openblas_audit(baseline)
            if audit['binary_sha256']!=provenance['openblas_before']['binary_sha256']:
                raise RuntimeError('OpenBLAS binaries changed during experiment')
            provenance['openblas_after']=audit
            provenance['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
            dump(provenance_path,provenance)
            (output/'status.txt').write_text('DATA_COMPLETE\n')
        print(output,flush=True)
    except Exception:
        (output/'status.txt').write_text('FAILED_RESUMABLE\n')
        raise


if __name__=='__main__':main()
