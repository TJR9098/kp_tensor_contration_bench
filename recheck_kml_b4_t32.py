#!/usr/bin/env python3
"""Repeat XL/X0 and XL/X4 B4/T32 for the historical and current KML variants."""
from __future__ import annotations
import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics as st
import subprocess
import sys

import run_tblis_batch_experiment as legacy_run
import run_kml_multi_batch as multi_run
import benchmark_tblis_batch_pytorch as legacy_worker
from benchmark_kml_multi_batch import OBSERVER_FIELDS
from summarize_tblis_experiment import write_cdf_grid

ROOT=Path(__file__).resolve().parent
OLD=multi_run.DEFAULT_BASELINE
PREVIOUS=ROOT/'results/pytorch_kml_multi_cpu0-31_20261014T101523721193Z'
CASES=['XL_X0_B4_T32','XL_X4_B4_T32']
LOCK='pytorch_2.5.0_kml_patched'
MULTI='pytorch_2.5.0_kml_multi'


def rows(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def write_tsv(path,data):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]),delimiter='\t')
        writer.writeheader();writer.writerows(data)


def locking_worker(remaining):
    original=legacy_worker.validate_result
    def finite(result,reference):
        import torch
        if not bool(torch.isfinite(result).all()):raise RuntimeError('nonfinite output')
        return original(result,reference)
    legacy_worker.validate_result=finite
    sys.argv=[legacy_worker.__file__,*remaining]
    legacy_worker.main()


def locking_diagnostic(args):
    case=next(c for c in rows(OLD/'case_manifest.tsv') if c['case_id']==args.case)
    os.sched_setaffinity(0,set(range(32)))
    lib=ctypes.CDLL(str(args.library),mode=ctypes.RTLD_GLOBAL)
    import torch
    torch.set_num_threads(32);torch.set_num_interop_threads(1)
    observer=ctypes.CDLL(str(args.observer),mode=ctypes.RTLD_GLOBAL)
    observer.kml_batch_observer_get.argtypes=[ctypes.c_int]
    observer.kml_batch_observer_get.restype=ctypes.c_int
    a,_,_=legacy_worker.build_operand(torch,case,'A')
    b,_,_=legacy_worker.build_operand(torch,case,'B')
    with torch.inference_mode():
        torch.einsum(case['equation'],a,b)
        observer.kml_batch_observer_begin()
        result=torch.einsum(case['equation'],a,b)
        observer.kml_batch_observer_end()
        reference=torch.from_file(str(OLD/'references/XL_B32.f64'),shared=False,
            size=32*2048*2048,dtype=torch.float64).reshape(32,32,64,32,64)[:4]
        if not bool(torch.isfinite(result).all()):raise RuntimeError('nonfinite diagnostic')
        abs_error,rel_error=legacy_worker.validate_result(result,reference)
    observed={name:observer.kml_batch_observer_get(i) for i,name in enumerate(OBSERVER_FIELDS)}
    if observed['sgemm_calls']!=4 or observed['sgemm_calling_threads']!=4:
        raise RuntimeError(f'unexpected batch dispatch {observed}')
    if observed['blas_local_threads_min']!=1 or observed['blas_local_threads_max']!=1 or observed['kml_max_team']>1:
        raise RuntimeError('unexpected inner threading')
    if rel_error>1e-4:raise RuntimeError('accuracy failure')
    dump(args.output/'locking_thread_diagnostics'/f'{args.case}.json',
         {'case':case,'method':LOCK,'observation':observed,'max_abs_error':abs_error,
          'max_rel_error':rel_error,'loaded_shared_objects':legacy_worker.loaded_libraries()})


def run(args):
    if not set(range(32))<=os.sched_getaffinity(0):raise RuntimeError('CPU0-31 required')
    quota=Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    if quota.exists() and int(quota.read_text())>0:
        if int(quota.read_text())/int((quota.parent/'cpu.cfs_period_us').read_text())<32:
            raise RuntimeError('insufficient CPU quota')
    previous=json.loads((PREVIOUS/'provenance.json').read_text())
    for name,digest in previous['source_sha256'].items():
        if multi_run.sha(ROOT/name)!=digest:raise RuntimeError(f'source differs from previous experiment: {name}')
    output=args.output or ROOT/'results'/('kml_b4_t32_recheck_'+legacy_run.utc_stamp())
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'logs').mkdir();(output/'status.txt').write_text('RUNNING\n')
    cases=[c for c in rows(OLD/'case_manifest.tsv') if c['case_id'] in CASES]
    write_tsv(output/'case_manifest.tsv',cases)
    source_names=['recheck_kml_b4_t32.py',*previous['source_sha256']]
    (output/'source_snapshot').mkdir()
    for name in source_names:shutil.copy2(ROOT/name,output/'source_snapshot'/name)
    kml_root=legacy_run.PYTORCH_ROOT/'hpckit-components/KunpengHPCKit-kml.26.1.RC1/bishenglib/sve/kblas'
    libraries={LOCK:(kml_root/'locking/libkblas.so.1.26.1.RC1').resolve(),
               MULTI:(kml_root/'multi/libkblas.so.1.26.1.RC1').resolve()}
    observer=output/'kml_batch_observer.so'
    compiler=legacy_run.PYTORCH_ROOT/'hpckit-components/BiShengCompiler-5.2.0-aarch64-linux-mini-hpc'
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wall','-Wextra',
                    '-idirafter',str(compiler/'lib/clang/19/include'),str(ROOT/'kml_batch_observer.cpp'),
                    '-ldl','-o',str(observer)],check=True)
    ref=OLD/'references/XL_B32.f64'
    if multi_run.sha(ref)!=json.loads(ref.with_suffix('.json').read_text())['sha256']:
        raise RuntimeError('reference hash mismatch')
    provenance={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'cases':CASES,
        'methods':[LOCK,MULTI],'campaigns':3,'warmup':10,'repeat':50,'affinity':list(range(32)),
        'old_locking_run':str(OLD),'previous_multi_run':str(PREVIOUS),
        'source_sha256':{name:multi_run.sha(ROOT/name) for name in source_names},
        'library_sha256':{m:multi_run.sha(p) for m,p in libraries.items()},
        'reference_sha256':multi_run.sha(ref),
        'policy':'both methods retain B4 outer dispatch; each GEMM inner=1; no alternative schedule mixed in'}
    dump(output/'provenance.json',provenance)
    (output/'lscpu.txt').write_text(subprocess.run(['lscpu'],capture_output=True,text=True,check=True).stdout)
    print('RESULTS '+str(output),flush=True)
    tasks=[(c,m,k) for k in range(3) for c in cases for m in [LOCK,MULTI]]
    random.Random(20261014).shuffle(tasks)
    raw=[];campaigns=[];metadata=[]
    def invoke(command,env,name):
        p=subprocess.run(command,env=env,cwd=ROOT,capture_output=True,text=True,timeout=180)
        (output/'logs'/f'{name}.log').write_text(p.stdout+'\n'+p.stderr)
        if p.returncode:raise RuntimeError(f'failed {name}, see logs')
    try:
        for index,(case,method,campaign) in enumerate(tasks,1):
            common=['--manifest',str(output/'case_manifest.tsv'),'--case',case['case_id'],
                    '--method',method,'--campaign',str(campaign),'--output-dir',str(output),
                    '--reference-dir',str(OLD/'references')]
            python=str(legacy_run.PYTORCH_ROOT/'env-kml/bin/python')
            if method==MULTI:
                cmd=[python,str(ROOT/'benchmark_kml_multi_batch.py'),'--kml-library',str(libraries[method]),*common]
                env=multi_run.environment(case)
            else:
                cmd=[python,str(Path(__file__).resolve()),'--locking-worker',*common]
                _,env=legacy_run.command_for(case,method,campaign,output,False)
                env.pop('LD_PRELOAD',None)
            if campaign==0:cmd+=['--diagnostics']
            name=f"{case['case_id']}_{method}_c{campaign}"
            invoke(cmd,env,name)
            r=rows(output/'raw'/method/f"{case['case_id']}_c{campaign}.tsv")
            if len(r)!=50 or {int(x['repeat']) for x in r}!=set(range(50)):raise RuntimeError('sample count')
            meta=json.loads((output/'metadata'/method/f"{case['case_id']}_c{campaign}.json").read_text())
            baseline=OLD if method==LOCK else PREVIOUS
            prior=json.loads((baseline/'metadata'/method/f"{case['case_id']}_c0.json").read_text())
            for f in ['a_logical_sha256','b_logical_sha256','torch_config_sha256','affinity','warmup','repeat']:
                if meta[f]!=prior[f]:raise RuntimeError(f'metadata mismatch {f}')
            for operand in ['a_entry','b_entry']:
                for f in ['shape','stride_elements','storage_offset_elements']:
                    if meta[operand][f]!=prior[operand][f]:raise RuntimeError('layout mismatch')
            mapped=[p for p in meta['loaded_shared_objects'] if 'libkblas' in p]
            if len(mapped)!=1 or Path(mapped[0]).resolve()!=libraries[method]:raise RuntimeError('wrong library')
            if not math.isfinite(meta['max_rel_error']) or meta['max_rel_error']>1e-4:raise RuntimeError('accuracy')
            vals=sorted(int(x['total_ns']) for x in r)
            campaigns.append({'case_id':case['case_id'],'method':method,'campaign':campaign,
                              'median_ns':round(st.median(vals)),'p95_ns':vals[math.ceil(.95*len(vals))-1]})
            raw.extend(r);metadata.append(meta)
            print(f"PASS {index}/12 {name}: {st.median(vals)/1e6:.6f} ms",flush=True)
        for case in cases:
            env=multi_run.environment(case,observer)
            invoke([python,str(ROOT/'benchmark_kml_multi_batch.py'),'--kml-library',str(libraries[MULTI]),
                    '--kml-observer',str(observer),'--kml-ompt-only','--manifest',str(output/'case_manifest.tsv'),
                    '--case',case['case_id'],'--method',MULTI,'--campaign','0','--output-dir',str(output),
                    '--reference-dir',str(OLD/'references')],env,case['case_id']+'_multi_ompt')
            _,env=legacy_run.command_for(case,LOCK,0,output,False)
            env.update(OMP_TOOL='enabled',OMP_TOOL_LIBRARIES=str(observer),LD_PRELOAD=str(observer))
            invoke([python,str(Path(__file__).resolve()),'--locking-ompt','--case',case['case_id'],
                    '--library',str(libraries[LOCK]),'--observer',str(observer),'--output',str(output)],
                   env,case['case_id']+'_locking_ompt')
        summary=[];panels=[]
        for case in cases:
            curves=[]
            for method in [LOCK,MULTI]:
                base=OLD if method==LOCK else PREVIOUS
                previous_rows=rows(base/'summary.tsv')
                previous_row=next(r for r in previous_rows if r['case_id']==case['case_id'] and r['method']==method)
                vals=sorted(int(r['total_ns']) for r in raw if r['case_id']==case['case_id'] and r['method']==method)
                med=round(st.median(vals));p95=vals[math.ceil(.95*len(vals))-1]
                summary.append({'case_id':case['case_id'],'method':method,'samples':len(vals),
                    'previous_median_ns':int(previous_row['median_total_ns']),'median_total_ns':med,
                    'median_change_pct':100*(med/int(previous_row['median_total_ns'])-1),
                    'previous_p95_ns':int(previous_row['p95_total_ns']),'p95_total_ns':p95,
                    'p95_change_pct':100*(p95/int(previous_row['p95_total_ns'])-1),
                    'max_rel_error':max(x['max_rel_error'] for x in metadata if x['case']['case_id']==case['case_id'] and x['method']==method)})
                label='locking' if method==LOCK else 'multi';color='#4c78a8' if method==LOCK else '#e45756'
                curves.append({'legend':label+'/recheck','color':color,'values':vals})
                prior_raw=[int(r['total_ns']) for r in rows(base/'raw.tsv') if r['case_id']==case['case_id'] and r['method']==method]
                curves.append({'legend':label+'/previous','color':color,'dash':'6,3','values':prior_raw})
            panels.append((case['case_id'],curves))
        write_tsv(output/'raw.tsv',raw);write_tsv(output/'summary.tsv',summary);write_tsv(output/'campaigns.tsv',campaigns)
        write_cdf_grid(output/'cdf.svg','XL/B4/T32: KML timing recheck',panels,2,
                       subtitle='x: whole-batch E2E microseconds; y: empirical CDF; 150 samples per curve')
        report=['# XL/X0、XL/X4，B4/T32：KML 定向复测','',
                '不变量：M=N=2048、K=512，batch=4，CPU0–31，T32；各版本分别沿用此前输入、计时循环和运行环境。',
                '变量：重新启动的独立 campaign；locking/multi 分别与各自历史结果对照，不混用线程策略。',
                '每个用例/版本预热 10 次、正式 50 次，独立运行 3 轮。时间为完整四项缩并的端到端时间，包含输出分配。','',
                '## 结果','',
                '| 用例 | KML 版本 | 原中位数/ms | 复测中位数/ms | 变化 | 原 P95/ms | 复测 P95/ms | 最大归一化误差 |',
                '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            report.append(f"| {r['case_id']} | {'locking' if r['method']==LOCK else 'multi'} | {r['previous_median_ns']/1e6:.6f} | {r['median_total_ns']/1e6:.6f} | {r['median_change_pct']:+.3f}% | {r['previous_p95_ns']/1e6:.6f} | {r['p95_total_ns']/1e6:.6f} | {r['max_rel_error']:.3e} |")
        report += ['', '## 每轮中位数','', '| 用例 | 版本 | campaign | 中位数/ms | P95/ms |','|---|---|---:|---:|---:|']
        for r in sorted(campaigns,key=lambda r:(r['case_id'],r['method'],r['campaign'])):
            report.append(f"| {r['case_id']} | {'locking' if r['method']==LOCK else 'multi'} | {r['campaign']} | {r['median_ns']/1e6:.6f} | {r['p95_ns']/1e6:.6f} |")
        report += ['', '## 线程核验','', '| 用例 | 版本 | GEMM 次数 | GEMM 调用线程 | 每项内部线程 | KML 内部并行 team |','|---|---|---:|---:|---:|---:|']
        for case in cases:
            for method,folder in [(LOCK,'locking_thread_diagnostics'),(MULTI,'kml_thread_diagnostics')]:
                ob=json.loads((output/folder/f"{case['case_id']}.json").read_text())['observation']
                assert ob['sgemm_calls']==4 and ob['sgemm_calling_threads']==4
                assert ob['blas_local_threads_min']==ob['blas_local_threads_max']==1 and ob['kml_max_team']==0
                report.append(f"| {case['case_id']} | {'locking' if method==LOCK else 'multi'} | 4 | 4 | 1 | 0 |")
        maxchange=max(abs(r['median_change_pct']) for r in summary)
        verdict=(f'约 50 ms 的中位数在当前相同策略下得到复现；四组最大相对变化 {maxchange:.3f}%。'
                 if maxchange<=5 else f'存在超过 5% 的中位数变化，最大 {maxchange:.3f}%，需进一步解释。')
        report.insert(6,'结论：'+verdict)
        report += ['', 'T32 是请求线程预算，不表示四个 GEMM 都使用 32 个内部线程；这里实际为 4 个并发单线程 GEMM。',
                   '因此这次复测确认的是原 outer=4、inner=1 策略的时间，不代表 KML 在该数学用例上的最佳 32 核调度。',
                   'KML 内部多线程顺序处理 batch 或 hybrid 属于另一种实验策略，本次没有混入。',
                   '完整原始样本见 raw.tsv，逐轮统计见 campaigns.tsv，CDF 见 cdf.svg。参考校验在每轮计时结束后检查末次输出的全部元素。']
        (output/'recheck.md').write_text('\n'.join(report)+'\n')
        (output/'validation.log').write_text('PASS: 2 cases x 2 KML variants x 3 campaigns x 50 = 600 samples\nPASS: source, inputs, shape/stride, affinity and library mapping match original versions\nPASS: finite outputs and full-output FP64 reference validation <= 1e-4\nPASS: four independent BLAS/OMPT diagnostics confirm outer=4 and inner=1\n')
        provenance['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();dump(output/'provenance.json',provenance)
        (output/'status.txt').write_text('COMPLETE\n')
        print('COMPLETE '+str(output),flush=True)
    except Exception:
        (output/'status.txt').write_text('FAILED\n');raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--locking-worker',action='store_true')
    p.add_argument('--locking-ompt',action='store_true')
    p.add_argument('--case',choices=CASES)
    p.add_argument('--library',type=Path)
    p.add_argument('--observer',type=Path)
    args,remaining=p.parse_known_args()
    if args.locking_worker:
        # --case was consumed by this parser; restore it for the unchanged worker.
        locking_worker(['--case',args.case,*remaining])
    elif args.locking_ompt:locking_diagnostic(args)
    else:
        if remaining:p.error('unexpected arguments')
        run(args)


if __name__=='__main__':main()
