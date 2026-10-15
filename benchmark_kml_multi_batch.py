#!/usr/bin/env python3
"""KML-only launcher; reuse the historical timing/input code unchanged."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

import benchmark_tblis_batch_pytorch as legacy
from run_tblis_batch_experiment import THREAD_POLICY

METHOD = 'pytorch_2.5.0_kml_multi'
OBSERVER_FIELDS = ['ompt_initialized','sgemm_calls','sgemm_calling_threads','sgemm_calls_in_outer_region',
                   'blas_local_threads_min','blas_local_threads_max','sgemm_entry_active_level_max',
                   'kml_parallel_regions','kml_max_team','kml_implicit_tasks',
                   'torch_parallel_regions','torch_max_team','other_parallel_regions','other_max_team',
                   'parallel_callback_status','implicit_callback_status']


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--kml-library', required=True, type=Path)
    parser.add_argument('--kml-observer', type=Path)
    parser.add_argument('--kml-ompt-only', action='store_true')
    parser.add_argument('--kml-distinct-only', action='store_true')
    custom, remaining=parser.parse_known_args()
    p=argparse.ArgumentParser(add_help=False)
    for flag in ['manifest','output-dir','reference-dir']: p.add_argument('--'+flag, type=Path, required=True)
    for flag in ['case','method']: p.add_argument('--'+flag, required=True)
    p.add_argument('--campaign',type=int,required=True)
    args,_=p.parse_known_args(remaining)
    if args.method!=METHOD: raise ValueError('launcher is restricted to KML multi')
    case=legacy.read_case(args.manifest,args.case)
    threads,batch=int(case['threads']),int(case['batch'])
    cpus=set(THREAD_POLICY[str(threads)])
    if not cpus <= os.sched_getaffinity(0): raise RuntimeError('required affinity unavailable')
    os.sched_setaffinity(0,cpus)
    library=custom.kml_library.resolve(strict=True)
    lib=ctypes.CDLL(str(library),mode=ctypes.RTLD_GLOBAL)
    for name in ['BlasGetParallel','BlasGetNumThreads','BlasGetNumThreadsLocal','BlasGetNumProcs']:
        getattr(lib,name).argtypes=[];getattr(lib,name).restype=ctypes.c_int
    for name in ['BlasSetNumThreads','BlasSetNumThreadsLocal']:
        getattr(lib,name).argtypes=[ctypes.c_int];getattr(lib,name).restype=None
    inner=threads if batch==1 else 1
    lib.BlasSetNumThreads(inner);lib.BlasSetNumThreadsLocal(inner)
    import torch
    torch.set_num_threads(threads)
    # Both setters are process-isolated; no global environment or installed library changes.
    lib.BlasSetNumThreads(inner);lib.BlasSetNumThreadsLocal(inner)
    loaded=[p for p in legacy.loaded_libraries() if 'libkblas' in p]
    if len(loaded)!=1 or Path(loaded[0]).resolve()!=library: raise RuntimeError(f'wrong KML: {loaded}')
    if lib.BlasGetParallel()!=2: raise RuntimeError('expected OpenMP multi variant')
    if lib.BlasGetNumThreadsLocal()!=inner: raise RuntimeError('KML thread request not realized')
    plan={'mode':'inner' if batch==1 else 'outer','requested_threads':threads,
          'configured_blas_inner_threads':inner,'configured_outer_gemm_tasks':min(batch,threads) if batch>1 else 1,
          'library':str(library),'blas_parallel_type':lib.BlasGetParallel(),
          'blas_num_procs':lib.BlasGetNumProcs(),'max_active_levels':os.environ.get('OMP_MAX_ACTIVE_LEVELS'),
          'omp_tool':os.environ.get('OMP_TOOL'),'actual_global_threads':lib.BlasGetNumThreads(),
          'actual_main_local_threads':lib.BlasGetNumThreadsLocal()}
    validate_original=legacy.validate_result
    def validate_finite(result, reference):
        if not bool(torch.isfinite(result).all()): raise RuntimeError('nonfinite output')
        return validate_original(result, reference)
    legacy.validate_result=validate_finite
    if custom.kml_distinct_only:
        import numpy as np
        rng=np.random.default_rng(20261014+batch)
        aa=rng.uniform(-1,1,(batch,3,5,3,9)).astype(np.float32)
        bb=rng.uniform(-1,1,(batch,3,9,4,7)).astype(np.float32)
        expected=aa.reshape(batch,15,27).astype(np.float64) @ bb.reshape(batch,27,28).astype(np.float64)
        a=torch.from_numpy(aa);b=torch.from_numpy(bb)
        if case['layout']=='X4':
            a=a.permute(0,1,3,2,4).contiguous().permute(0,1,3,2,4)
            b=b.permute(0,1,3,2,4).contiguous().permute(0,1,3,2,4)
        with torch.inference_mode(): result=torch.einsum(case['equation'],a,b).reshape(batch,15,28)
        abs_error,rel_error=validate_finite(result,torch.from_numpy(expected))
        if rel_error>1e-4: raise RuntimeError('distinct-batch validation failed')
        dump(args.output_dir/'distinct_validation'/f'{args.case}.json',
             {'case_id':args.case,'plan':plan,'M':15,'N':28,'K':27,'batch':batch,
              'layout':case['layout'],'seed':20261014+batch,'max_rel_error':rel_error,
              'reference':'independent NumPy FP64 batched matmul',
              'scope':'supplemental correctness; formal inputs unchanged'})
        print('DISTINCT_BATCH_PASS '+args.case,flush=True)
        return
    if custom.kml_ompt_only:
        observer=ctypes.CDLL(str(custom.kml_observer),mode=ctypes.RTLD_GLOBAL)
        observer.kml_batch_observer_get.argtypes=[ctypes.c_int]
        observer.kml_batch_observer_get.restype=ctypes.c_int
        a,_,_=legacy.build_operand(torch,case,'A');b,_,_=legacy.build_operand(torch,case,'B')
        with torch.inference_mode():
            torch.einsum(case['equation'],a,b)
            observer.kml_batch_observer_begin()
            result=torch.einsum(case['equation'],a,b)
            observer.kml_batch_observer_end()
            reference=torch.from_file(str(args.reference_dir/f"{case['scale']}_B32.f64"),
                shared=False,size=32*int(case['M'])*int(case['N']),dtype=torch.float64)
            reference=reference.reshape([32,*legacy.parse_shape(case['c_shape'])])[:batch]
            abs_error,rel_error=validate_finite(result,reference)
        data={name:observer.kml_batch_observer_get(i) for i,name in enumerate(OBSERVER_FIELDS)}
        if data['sgemm_calls']!=batch: raise RuntimeError(f'BLAS boundary call count mismatch: {data}')
        if data['blas_local_threads_min']!=inner or data['blas_local_threads_max']!=inner:
            raise RuntimeError(f'KML worker TLS does not match plan: {data}')
        if data['sgemm_entry_active_level_max']>1 or (batch>1 and data['kml_max_team']>1):
            raise RuntimeError(f'nested parallel team exceeded plan: {data}')
        if rel_error>1e-4: raise RuntimeError('error tolerance exceeded')
        dump(args.output_dir/'kml_thread_diagnostics'/f'{args.case}.json',
             {'case':case,'method':METHOD,'plan':plan,'observation':data,'max_rel_error':rel_error,
              'max_abs_error':abs_error,'scope':'separate one-einsum diagnostic; LD_PRELOAD + OMPT enabled'})
        print(json.dumps({'case':args.case,**data}),flush=True)
        return
    sys.argv=[str(Path(legacy.__file__)),*remaining]
    write_original=legacy.write_one_row
    def write_with_plan(path,row):
        if 'thread_diagnostics' in path.parts:
            row=dict(row)
            row['configured_inner_threads']=inner
            row['configured_outer_workers']=plan['configured_outer_gemm_tasks']
            row['thread_configuration_source']='explicit KML multi plan; activity count is not GEMM team count'
        write_original(path,row)
    legacy.write_one_row=write_with_plan
    legacy.main()
    metadata_path=args.output_dir/'metadata'/METHOD/f'{args.case}_c{args.campaign}.json'
    metadata=json.loads(metadata_path.read_text());metadata['kml_multi_plan']=plan
    metadata['legacy_runner_sha256']=hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()
    dump(metadata_path,metadata)


if __name__=='__main__': main()
