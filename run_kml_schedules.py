#!/usr/bin/env python3
"""Run all 120 cases with explicitly separate KML multi inter and intra."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import run_kml_multi_batch as previous
from benchmark_kml_schedules import METHODS

ROOT=Path(__file__).resolve().parent
OLD=previous.DEFAULT_BASELINE
BUILD=ROOT/'bin/kml_schedules'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if not set(range(32))<=os.sched_getaffinity(0):raise RuntimeError('CPU0-31 required')
    quota=Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    if quota.exists() and int(quota.read_text())>0 and int(quota.read_text())/int((quota.parent/'cpu.cfs_period_us').read_text())<32:
        raise RuntimeError('CPU quota <32 cores')
    build=json.loads((BUILD/'build.json').read_text())
    schedule_lib=BUILD/'libtorch_cpu.so'
    if previous.sha(schedule_lib)!=build['library_sha256']:raise RuntimeError('isolated library changed')
    project=previous.old.PYTORCH_ROOT
    kml=project/'hpckit-components/KunpengHPCKit-kml.26.1.RC1/bishenglib/sve/kblas/multi/libkblas.so.1.26.1.RC1'
    observer=ROOT/'bin/kml_batch_observer.so'
    compiler=project/'hpckit-components/BiShengCompiler-5.2.0-aarch64-linux-mini-hpc'
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wall','-Wextra','-idirafter',
                    str(compiler/'lib/clang/19/include'),str(ROOT/'kml_batch_observer.cpp'),'-ldl','-o',str(observer)],check=True)
    output=(args.output_dir or ROOT/'results'/('kml_inter_intra_cpu0-31_'+previous.old.utc_stamp())).resolve()
    output.mkdir(parents=True,exist_ok=True);print('RESULTS '+str(output),flush=True)
    cases=previous.read_rows(OLD/'case_manifest.tsv')
    if len(cases)!=120:raise RuntimeError('full 120-case manifest required')
    sources=['run_kml_schedules.py','benchmark_kml_schedules.py','build_kml_schedules.py',
             'kml_bmm_schedule.patch','kml_batch_observer.cpp','benchmark_tblis_batch_pytorch.py',
             'benchmark_tblis_pytorch.py','run_kml_multi_batch.py','run_tblis_batch_experiment.py']
    hashes={f:previous.sha(ROOT/f) for f in sources}
    history=json.loads((OLD/'provenance.json').read_text())
    if hashes['benchmark_tblis_batch_pytorch.py']!=history['artifact_sha256']['pytorch_runner']:
        raise RuntimeError('historical timer modified')
    if (output/'provenance.json').exists():
        prov=json.loads((output/'provenance.json').read_text())
        if prov['source_sha256']!=hashes or prov['schedule_library_sha256']!=previous.sha(schedule_lib):
            raise RuntimeError('resume sources changed')
    else:
        if any(output.iterdir()):raise RuntimeError('nonempty new output')
        reference_hashes={}
        for scale in ['S','M','L','XL']:
            f=OLD/'references'/f'{scale}_B32.f64';digest=previous.sha(f)
            if digest!=json.loads(f.with_suffix('.json').read_text())['sha256']:raise RuntimeError('reference checksum')
            reference_hashes[scale]=digest
        prov={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'methods':list(METHODS),
              'baseline':str(OLD),'previous_mixed_policy':str(ROOT/'results/pytorch_kml_multi_cpu0-31_20261014T101523721193Z'),
              'source_sha256':hashes,'schedule_library':str(schedule_lib),'schedule_library_sha256':previous.sha(schedule_lib),
              'library':str(kml.resolve()),'library_sha256':previous.sha(kml),'build':build,
              'reference_sha256':reference_hashes,'campaigns':3,'warmup':10,'repeat':50,
              'affinity':sorted(os.sched_getaffinity(0)),'openblas_before':previous.openblas_audit(OLD),
              'policy':{'kml-multi-inter':'ATen parallel_for batches; KML inner=1, including B1',
                        'kml-multi-intra':'C++ serial batches; KML inner=T, including B4/B32'},
              'formal_inputs':'unchanged historical input generator and reference; supplementary distinct-batch validation',
              'timing':'unchanged legacy full einsum timer; no Python batch loop; same isolated torch_cpu for both methods'}
        previous.dump(output/'provenance.json',prov)
        shutil.copy2(OLD/'case_manifest.tsv',output/'case_manifest.tsv')
        (output/'logs').mkdir();(output/'source_snapshot').mkdir()
        for f in sources:shutil.copy2(ROOT/f,output/'source_snapshot'/f)
        (output/'lscpu.txt').write_text(subprocess.run(['lscpu'],capture_output=True,text=True,check=True).stdout)
    (output/'status.txt').write_text('RUNNING\n')
    def invoke(case,method,campaign,mode):
        cid=case['case_id'];schedule=METHODS[method]
        cmd=[str(project/'env-kml/bin/python'),str(ROOT/'benchmark_kml_schedules.py'),
             '--kml-library',str(kml),'--schedule-library',str(schedule_lib),
             '--manifest',str(output/'case_manifest.tsv'),'--case',cid,'--method',method,
             '--campaign',str(campaign),'--output-dir',str(output),'--reference-dir',str(OLD/'references')]
        env=previous.environment(case,observer if mode=='ompt' else None)
        env['LD_LIBRARY_PATH']=str(project/'pytorch-v2.5.0/torch/lib')+':'+env['LD_LIBRARY_PATH']
        env['BLAS_NUM_THREADS']=case['threads'] if schedule=='intra' else '1'
        env['KML_BMM_SCHEDULE']=schedule
        if mode=='ompt':
            cmd+=['--kml-observer',str(observer),'--kml-ompt-only']
            expected=[output/'kml_thread_diagnostics'/method/f'{cid}.json']
        elif mode=='distinct':
            cmd+=['--kml-distinct-only'];expected=[output/'distinct_validation'/method/f'{cid}.json']
        else:
            trace=previous.old.should_trace(case,campaign)
            if campaign==0:cmd+=['--diagnostics']
            if trace:cmd+=['--export-trace']
            expected=previous.old.expected_paths(output,case,method,campaign,trace)
        if all(f.exists() for f in expected):return
        if any(f.exists() for f in expected):raise RuntimeError('partial output')
        log=output/'logs'/f'{cid}_{method}_c{campaign}_{mode}.log'
        with log.open('w') as stream:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise RuntimeError(f'failed {result.returncode}, see {log}')
        if not all(f.exists() for f in expected):raise RuntimeError(f'missing outputs {log}')
    try:
        if args.preflight_only:
            for cid in ['S_X0_B1_T32','XL_X0_B4_T32','XL_X4_B4_T32','XL_X4_B32_T32','S_X4_B32_T4']:
                case=next(c for c in cases if c['case_id']==cid)
                for method in METHODS:
                    invoke(case,method,0,'distinct');invoke(case,method,0,'ompt')
                    print('PREFLIGHT PASS '+cid+' '+method,flush=True)
            (output/'status.txt').write_text('PREFLIGHT_COMPLETE\n')
        else:
            for case in cases:
                if case['scale']=='S':
                    for method in METHODS:invoke(case,method,0,'distinct')
            tasks=[(case,method,campaign) for campaign in range(3) for case in cases for method in METHODS]
            random.Random(20261014).shuffle(tasks)
            for index,(case,method,campaign) in enumerate(tasks,1):
                invoke(case,method,campaign,'formal')
                if campaign==0:invoke(case,method,campaign,'ompt')
                if index%5==0 or index==len(tasks):print(f'PROGRESS {index}/{len(tasks)}',flush=True)
            prov['openblas_after']=previous.openblas_audit(OLD)
            if prov['openblas_before']['binary_sha256']!=prov['openblas_after']['binary_sha256']:
                raise RuntimeError('OpenBLAS binary drift')
            if previous.sha(project/'pytorch-v2.5.0/torch/lib/libtorch_cpu.so')!=build['inputs']['original_library']:
                raise RuntimeError('installed KML torch changed')
            prov['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();previous.dump(output/'provenance.json',prov)
            (output/'status.txt').write_text('DATA_COMPLETE\n')
        print(output,flush=True)
    except Exception:
        (output/'status.txt').write_text('FAILED_RESUMABLE\n');raise


if __name__=='__main__':main()
