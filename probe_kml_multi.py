#!/usr/bin/env python3
"""Isolated KML locking/multi GEMM checks and a resource-aware thread sweep.

Use the existing env-kml Python (NumPy) for workers; the coordinator needs only
stdlib. Native calls include the ctypes boundary, BLAS packing and computation.
PyTorch is tested in separate processes with the selected KML loaded before torch.
Historical environments, libraries, and reports are never modified.
"""
from __future__ import annotations

import argparse
import csv
import ctypes as ct
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SCALES = {
    "S": (16, 16, 16, 16, 8, 8),
    "M": (16, 32, 16, 32, 8, 16),
    "L": (32, 32, 32, 32, 16, 16),
    "XL": (32, 64, 32, 64, 16, 32),
}


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def mappings():
    return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if '/' in line and any(s in line for s in
                       ('libkblas', 'libomp', 'libgomp', 'libopenblas', 'libscipy_openblas', 'libtorch_cpu'))})


def check_error(np, actual, reference):
    if not np.isfinite(actual).all():
        raise RuntimeError('nonfinite output')
    error = np.abs(actual.astype(np.float64) - reference)
    absolute = float(error.max())
    normalized = float((error / np.maximum(np.abs(reference), 1.0)).max())
    if not math.isfinite(normalized) or normalized > 1e-4:
        raise RuntimeError(f'normalized error exceeds 1e-4: {normalized}')
    return {'max_abs_error': absolute, 'max_normalized_error': normalized,
            'reference': 'independent NumPy FP64 matmul', 'checked_elements': int(actual.size)}


def bind(lib, name, argtypes, restype=None):
    f = getattr(lib, name)
    f.argtypes = argtypes
    f.restype = restype
    return f


def worker(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    allowed = sorted(os.sched_getaffinity(0))
    affinity = allowed[:min(args.threads, len(allowed))]
    os.sched_setaffinity(0, affinity)
    observer = None
    if args.diagnostic:
        observer = ct.CDLL(str(args.observer), mode=ct.RTLD_GLOBAL)
        bind(observer, 'kml_probe_begin', [])
        bind(observer, 'kml_probe_end', [])
        bind(observer, 'kml_probe_get', [ct.c_int], ct.c_int)
        if args.native_backtrace:
            observer.kml_probe_fault_handler()
    lib = ct.CDLL(str(args.library), mode=ct.RTLD_GLOBAL)
    getters = {name: bind(lib, name, [], ct.c_int) for name in
               ['BlasGetParallel', 'BlasGetNumThreads', 'BlasGetNumThreadsLocal', 'BlasGetNumProcs']}
    global_set = bind(lib, 'BlasSetNumThreads', [ct.c_int])
    local_set = bind(lib, 'BlasSetNumThreadsLocal', [ct.c_int])
    before = {k: f() for k, f in getters.items()}
    global_set(args.threads)
    local_set(args.threads)
    after = {k: f() for k, f in getters.items()}
    print('loaded ' + json.dumps(after), flush=True)

    import numpy as np
    a_dim, b_dim, c_dim, d_dim, k_dim, l_dim = SCALES[args.scale]
    m, n, k = a_dim*b_dim, c_dim*d_dim, k_dim*l_dim
    rng = np.random.default_rng(20261014 + m*31 + n*17 + k)
    a = rng.uniform(-1.0, 1.0, (m, k)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, (k, n)).astype(np.float32)
    reference = a.astype(np.float64) @ b.astype(np.float64)
    ptr = ct.POINTER(ct.c_float)
    gemm = bind(lib, 'cblas_sgemm', [ct.c_int]*6 +
                [ct.c_float, ptr, ct.c_int, ptr, ct.c_int, ct.c_float, ptr, ct.c_int])
    def pointer(x):
        return x.ctypes.data_as(ptr)

    edge_checks = []
    if args.engine == 'cblas':
        # Independent FP64 reference, every output, transpose/lda/ldc and beta.
        for ta, tb in [(False, False), (True, False), (False, True), (True, True)]:
            aa = rng.uniform(-1, 1, (13, 29)).astype(np.float32)
            bb = rng.uniform(-1, 1, (29, 17)).astype(np.float32)
            pa, pb = (aa.T if ta else aa), (bb.T if tb else bb)
            ap = np.full((pa.shape[0], pa.shape[1]+3), np.nan, dtype=np.float32)
            bp = np.full((pb.shape[0], pb.shape[1]+5), np.nan, dtype=np.float32)
            cp = np.full((13, 17+7), np.nan, dtype=np.float32)
            ap[:, :pa.shape[1]] = pa; bp[:, :pb.shape[1]] = pb
            cp[:, :17] = rng.uniform(-1, 1, (13, 17))
            expected = .75*(aa.astype(np.float64) @ bb.astype(np.float64)) - .25*cp[:, :17].astype(np.float64)
            gemm(101, 112 if ta else 111, 112 if tb else 111, 13, 17, 29,
                 .75, pointer(ap), ap.shape[1], pointer(bp), bp.shape[1], -.25, pointer(cp), cp.shape[1])
            if not np.isnan(cp[:, 17:]).all():
                raise RuntimeError('output padding overwritten')
            edge_checks.append({'transpose_a': ta, 'transpose_b': tb, **check_error(np, cp[:, :17], expected)})
        out = np.full((m, n), np.nan, dtype=np.float32)
        ap, bp, cp = pointer(a), pointer(b), pointer(out)
        def operation():
            gemm(101, 111, 111, m, n, k, 1., ap, k, bp, n, 0., cp, n)
            return out
        # PyTorch FP32 uses this Fortran ABI. Row-major C = A B is C^T = B^T A^T.
        ip = ct.POINTER(ct.c_int)
        fp = bind(lib, 'sgemm_', [ct.c_char_p, ct.c_char_p, ip, ip, ip, ptr, ptr, ip, ptr, ip, ptr, ptr, ip])
        mi, ni, ki = ct.c_int(m), ct.c_int(n), ct.c_int(k)
        alpha, beta = ct.c_float(1.), ct.c_float(0.)
        fp(b'N', b'N', ct.byref(ni), ct.byref(mi), ct.byref(ki), ct.byref(alpha),
           bp, ct.byref(ni), ap, ct.byref(ki), ct.byref(beta), cp, ct.byref(ni))
        fortran_check = check_error(np, out, reference)
        actual_array = lambda value: value
        torch_meta = None
    else:
        print('importing torch', flush=True)
        import torch
        print('torch imported', flush=True)
        if 'BLAS_INFO=KML' not in torch.__config__.show():
            raise RuntimeError('expected KML PyTorch build')
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        # Keep the explicit KML setting after any framework runtime setup.
        global_set(args.threads); local_set(args.threads)
        torch.set_grad_enabled(False)
        left = torch.from_numpy(a.reshape(1, a_dim, b_dim, k_dim, l_dim))
        right = torch.from_numpy(b.reshape(1, k_dim, l_dim, c_dim, d_dim))
        if args.layout == 'X4':
            left = left.permute(0, 1, 3, 2, 4).contiguous().permute(0, 1, 3, 2, 4)
            right = right.permute(0, 1, 3, 2, 4).contiguous().permute(0, 1, 3, 2, 4)
        def operation():
            with torch.inference_mode():
                return torch.einsum('pabkl,pklcd->pabcd', left, right)
        actual_array = lambda value: value.reshape(m, n).numpy()
        fortran_check = None
        torch_meta = {'version': torch.__version__, 'path': torch.__file__,
                      'config': torch.__config__.show(), 'threads': torch.get_num_threads(),
                      'a_shape': list(left.shape), 'b_shape': list(right.shape),
                      'a_stride': list(left.stride()), 'b_stride': list(right.stride())}
    libs = mappings()
    loaded_kml = [p for p in libs if 'libkblas' in p]
    if len(loaded_kml) != 1 or Path(loaded_kml[0]).resolve() != args.library.resolve():
        raise RuntimeError(f'wrong/mixed KML mapping: {loaded_kml}')
    for _ in range(args.warmup):
        result = operation()
    result = None
    samples = []
    diagnostic = None
    if args.diagnostic:
        # Formal timings and OMPT are collected in separate processes.
        observer.kml_probe_begin()
        result = operation()
        observer.kml_probe_end()
        fields = ['initialized', 'parallel_events', 'max_requested_team', 'max_actual_team',
                  'implicit_task_begins', 'parallel_callback_status', 'implicit_callback_status']
        diagnostic = {f: observer.kml_probe_get(i) for i, f in enumerate(fields)}
        diagnostic['scope'] = 'cblas_sgemm' if args.engine == 'cblas' else 'whole_einsum_including_transform'
    else:
        for repeat in range(args.repeat):
            result = None  # Release old framework output outside the timed boundary.
            cpu = time.process_time_ns(); start = time.perf_counter_ns()
            result = operation()
            wall = time.perf_counter_ns()-start; cpu = time.process_time_ns()-cpu
            samples.append({'repeat': repeat, 'total_ns': wall, 'process_cpu_ns': cpu})
    accuracy = check_error(np, actual_array(result), reference)
    metadata = {
        'engine': args.engine, 'variant': args.variant, 'scale': args.scale, 'layout': args.layout,
        'M': m, 'N': n, 'K': k, 'batch': 1, 'requested_threads': args.threads,
        'affinity': affinity, 'available_cpus_before_pin': allowed, 'campaign': args.campaign,
        'thread_api_before': before, 'thread_api_after_set': after,
        'thread_api_after_gemm': {name: f() for name, f in getters.items()},
        'library': str(args.library.resolve()), 'loaded_libraries': mappings(),
        'samples': samples, 'accuracy': accuracy, 'edge_checks': edge_checks,
        'fortran_abi_check': fortran_check, 'ompt': diagnostic,
        'task_count_after': len(list(Path('/proc/self/task').iterdir())),
        'numpy_version': np.__version__, 'torch': torch_meta,
        'input_sha256': hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(),
        'warmup': args.warmup, 'repeat': args.repeat,
        'error_formula': 'max(abs(actual-reference)/max(abs(reference),1))',
        'wait_policy': os.environ.get('OMP_WAIT_POLICY'), 'kmp_blocktime': os.environ.get('KMP_BLOCKTIME'),
    }
    dump(args.worker_output, metadata)
    print('PASS ' + json.dumps(accuracy), flush=True)


def write_tsv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, delimiter='\t')
        writer.writeheader(); writer.writerows(rows)


def audit_imports(project, kml, compiler, observer, output):
    """Compare a reproducible loader failure with explicit OpenMP tool settings."""
    results = []
    for variant, policy in [('locking','default'), ('multi','default'),
                            ('multi','disabled'), ('multi','explicit_observer')]:
        for repeat in range(2):
            env = os.environ.copy()
            for name in ['LD_PRELOAD','OMP_TOOL','OMP_TOOL_LIBRARIES']:
                env.pop(name, None)
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', BLAS_NUM_THREADS='1',
                       OMP_WAIT_POLICY='PASSIVE', KMP_BLOCKTIME='0', PYTHONFAULTHANDLER='1',
                       LD_LIBRARY_PATH=':'.join(map(str,[compiler/'lib/aarch64-unknown-linux-gnu',
                         project/'hpckit-components/Hyper-MPI_26.1.RC1_aarch64_openEuler22.03_Bisheng_hns-release/release/hmpi/lib'])))
            if policy == 'disabled': env['OMP_TOOL'] = 'disabled'
            if policy == 'explicit_observer':
                env.update(OMP_TOOL='enabled', OMP_TOOL_LIBRARIES=str(observer))
            library = kml/variant/'libkblas.so.1.26.1.RC1'
            code = (
                'import ctypes,resource\nresource.setrlimit(resource.RLIMIT_CORE,(0,0))\n'
                f'lib=ctypes.CDLL({str(library)!r},mode=ctypes.RTLD_GLOBAL)\n'
                'print("KML loaded",flush=True)\nimport torch\n'
                'print("torch imported",torch.__version__,flush=True)\n'
                'torch.set_num_threads(1)\n'
                'x=torch.ones(128,64);y=torch.ones(64,32);z=torch.mm(x,y)\n'
                'assert bool((z==64).all())\nprint("GEMM PASS",flush=True)\n'
                'print("".join(line for line in open("/proc/self/maps") if "libkblas" in line))\n'
            )
            try:
                run = subprocess.run([str(project/'env-kml/bin/python'),'-c',code],
                                     env=env, capture_output=True, text=True, timeout=30)
                returncode, log = run.returncode, run.stdout+'\n'+run.stderr
            except subprocess.TimeoutExpired as exc:
                returncode, log = 'timeout', str(exc.stdout)+'\n'+str(exc.stderr)
            name = f'import_{variant}_{policy}_{repeat}.log'
            (output/'logs'/name).write_text(log)
            results.append({'variant':variant, 'omp_tool_policy':policy,
                            'repeat':repeat, 'returncode':returncode, 'log':name})
            print(f'IMPORT {variant} {policy} repeat={repeat} code={returncode}', flush=True)
    dump(output/'import_audit.json', results)


def coordinator(args):
    project = args.project.resolve()
    components = project/'hpckit-components'
    kml = components/'KunpengHPCKit-kml.26.1.RC1/bishenglib/sve/kblas'
    compiler = components/'BiShengCompiler-5.2.0-aarch64-linux-mini-hpc'
    output = args.output or ROOT/'results'/('kml_multi_probe_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'tasks').mkdir(); (output/'logs').mkdir()
    observer = ROOT/'bin/kml_probe_observer.so'
    observer.parent.mkdir(exist_ok=True)
    compile_cmd = ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra',
                   '-idirafter', str(compiler/'lib/clang/19/include'), str(ROOT/'kml_probe_observer.cpp'),
                   '-o', str(observer)]
    compiled = subprocess.run(compile_cmd, capture_output=True, text=True)
    (output/'compile.log').write_text(compiled.stdout+'\n'+compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(f'observer compile failed; see {output}/compile.log')
    available = sorted(os.sched_getaffinity(0))
    quota = None
    for quota_path in [Path('/sys/fs/cgroup/cpu.max'), Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')]:
        if quota_path.exists():
            value = quota_path.read_text().split()
            if quota_path.name == 'cpu.max' and value[0] != 'max':
                quota = int(value[0])/int(value[1])
            elif quota_path.name != 'cpu.max' and int(value[0]) > 0:
                quota = int(value[0])/int((quota_path.parent/'cpu.cfs_period_us').read_text())
    capacity = min(len(available), quota) if quota else len(available)
    provenance = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'available_cpus': available, 'cpu_capacity': capacity, 'quota_cores': quota,
                  'online': Path('/sys/devices/system/cpu/online').read_text().strip(),
                  'requested_threads': args.thread_list, 'scales': args.scales,
                  'campaigns': args.campaigns, 'warmup': args.warmup, 'repeat': args.repeat,
                  'compile_command': compile_cmd, 'worker_python': str(project/'env-kml/bin/python'),
                  'timing_scope': 'fresh outputs for torch; reused output for CBLAS; no OMPT in timed workers',
                  'scaling_limit': 'Requests beyond available CPU capacity are diagnostics, not multi-core scaling.',
                  'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   [Path(__file__), ROOT/'kml_probe_observer.cpp']},
                  'libraries_sha256': {v: hashlib.sha256((kml/v/'libkblas.so.1.26.1.RC1').read_bytes()).hexdigest()
                                       for v in ['locking', 'multi']}}
    dump(output/'provenance.json', provenance)
    (output/'lscpu.txt').write_text(subprocess.run(['lscpu'], capture_output=True, text=True).stdout)
    print('RESULTS '+str(output), flush=True)
    if args.audit_imports:
        audit_imports(project, kml, compiler, observer, output)
    tasks = [(engine, v, sc, layout, t, campaign, diag)
             for engine in args.engines for v in ['locking', 'multi'] for sc in args.scales
             for layout in (['NN'] if engine == 'cblas' else ['X0','X4'])
             for t in args.thread_list for campaign in range(args.campaigns)
             for diag in ([False, True] if campaign == 0 else [False])]
    random.Random(20261014).shuffle(tasks)
    failures, records = [], []
    failed_torch_variants = set()
    for idx, (engine, variant, scale, layout, threads, campaign, diagnostic) in enumerate(tasks, 1):
        if engine == 'torch' and variant in failed_torch_variants:
            failures.append({'engine': engine, 'variant': variant, 'scale': scale, 'layout': layout,
                             'threads': threads, 'campaign': campaign, 'diagnostic': diagnostic,
                             'status': 'skipped_after_import_failure'})
            continue
        name = f'{engine}_{variant}_{scale}_{layout}_T{threads}_c{campaign}'+('_diag' if diagnostic else '')
        env = os.environ.copy()
        # A single selected KML is dlopened before torch; do not rely on LD_PRELOAD.
        env.pop('LD_PRELOAD', None)
        env.pop('OMP_TOOL_LIBRARIES', None)
        env.update(OMP_NUM_THREADS=str(threads), BLAS_NUM_THREADS=str(threads),
                   OPENBLAS_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1',
                   OMP_WAIT_POLICY='PASSIVE', KMP_BLOCKTIME='0', OMP_TOOL='enabled' if diagnostic else 'disabled',
                   PYTHONFAULTHANDLER='1', PYTHONDONTWRITEBYTECODE='1')
        env['LD_LIBRARY_PATH'] = ':'.join(map(str, [compiler/'lib/aarch64-unknown-linux-gnu',
            components/'Hyper-MPI_26.1.RC1_aarch64_openEuler22.03_Bisheng_hns-release/release/hmpi/lib']))
        if diagnostic:
            env['OMP_TOOL_LIBRARIES'] = str(observer)
        cmd = [str(project/'env-kml/bin/python'), str(Path(__file__).resolve()), '--worker',
               '--engine', engine, '--variant', variant, '--scale', scale, '--layout', layout,
               '--threads', str(threads), '--campaign', str(campaign),
               '--warmup', str(args.warmup), '--repeat', str(args.repeat),
               '--library', str(kml/variant/'libkblas.so.1.26.1.RC1'),
               '--observer', str(observer), '--worker-output', str(output/'tasks'/f'{name}.json')]
        if diagnostic:
            cmd.append('--diagnostic')
            if engine == 'torch': cmd.append('--native-backtrace')
        try:
            run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
            code, log = run.returncode, run.stdout+'\n'+run.stderr
        except subprocess.TimeoutExpired as exc:
            code = 'timeout'
            log = str(exc.stdout)+'\n'+str(exc.stderr)
        (output/'logs'/f'{name}.log').write_text(log)
        if code != 0:
            failures.append({'task': name, 'engine': engine, 'variant': variant, 'returncode': code,
                             'command': cmd, 'status': 'failed'})
            if engine == 'torch' and 'importing torch' in log and 'torch imported' not in log:
                failed_torch_variants.add(variant)
            print(f'{idx}/{len(tasks)} FAIL {name} code={code}', flush=True)
            continue
        record = json.loads((output/'tasks'/f'{name}.json').read_text())
        record['task'] = name
        records.append(record)
        print(f'{idx}/{len(tasks)} PASS {name} effective={record["thread_api_after_gemm"]["BlasGetNumThreadsLocal"]}', flush=True)
    dump(output/'failures.json', failures)
    groups = {}
    diagnostics = {}
    raw = []
    for r in records:
        key = tuple(r[f] for f in ['engine','variant','scale','layout','requested_threads'])
        if r['ompt'] is not None:
            diagnostics[key] = r
            continue
        groups.setdefault(key, []).append(r)
        for sample in r['samples']:
            raw.append({**dict(zip(['engine','variant','scale','layout','requested_threads'],key)),
                        'campaign': r['campaign'], **sample})
    summary = []
    for key, rr in sorted(groups.items()):
        times = sorted(s['total_ns'] for r in rr for s in r['samples'])
        row = dict(zip(['engine','variant','scale','layout','requested_threads'],key))
        api = rr[0]['thread_api_after_gemm']; diag = diagnostics.get(key, {}).get('ompt', {})
        row.update(samples=len(times), median_ns=round(statistics.median(times)),
                   p95_ns=times[math.ceil(.95*len(times))-1],
                   blas_parallel_type=api['BlasGetParallel'], reported_blas_threads=api['BlasGetNumThreadsLocal'],
                   blas_num_procs=api['BlasGetNumProcs'], affinity=','.join(map(str,rr[0]['affinity'])),
                   max_normalized_error=max(r['accuracy']['max_normalized_error'] for r in rr),
                   median_cpu_equivalents=statistics.median(s['process_cpu_ns']/s['total_ns'] for r in rr for s in r['samples']),
                   diagnostic_ompt_initialized=diag.get('initialized','NA'),
                   diagnostic_max_actual_team=diag.get('max_actual_team','NA'),
                   resource_scope='within_cpu_capacity' if key[-1] <= capacity else 'request_exceeds_cpu_capacity',
                   complete_campaigns=len(rr)==args.campaigns)
        summary.append(row)
    for row in summary:
        baseline = next((r for r in summary if all(r[f]==row[f] for f in ['engine','variant','scale','layout']) and r['requested_threads']==1),None)
        # For >1 threads, a setter alone is insufficient: require a native GEMM
        # diagnostic to have observed a parallel team as well as CPU capacity.
        team = row['diagnostic_max_actual_team']
        confirmed_team = (row['requested_threads']==1 or
                          (row['engine']=='cblas' and isinstance(team,int) and team>1))
        valid = (row['requested_threads'] <= capacity and row['reported_blas_threads']==row['requested_threads']
                 and confirmed_team and row['complete_campaigns'] and baseline and baseline['complete_campaigns'])
        row['multi_core_speedup_vs_t1'] = baseline['median_ns']/row['median_ns'] if valid else 'NA'
    write_tsv(output/'raw.tsv', raw); write_tsv(output/'summary.tsv', summary)
    write_tsv(output/'thread_diagnostics.tsv', [
        {'task': r['task'], 'engine': r['engine'], 'variant': r['variant'],
         'requested_threads': r['requested_threads'], **r['ompt']} for r in diagnostics.values()])
    report = ['# KML SVE multi 兼容性与线程验证', '',
              f'可用 CPU：{available}；有效容量：{capacity} 核。',
              '请求数超过可用核数的记录仅用于兼容性诊断，多核扩展标为 N/A。',
              '端到端口径：CBLAS 含 ctypes 与后端调用，复用输出；PyTorch 为完整 B1 einsum，分配输出。',
              '等待设置：OMP_WAIT_POLICY=PASSIVE、KMP_BLOCKTIME=0；不直接与旧实验比较。', '',
              '| 接口 | 库 | 用例 | 请求线程 | KML 报告线程 | 中位数/ms | P95/ms | 归一化误差 | 多核扩展 |',
              '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in summary:
        report.append(f'| {r["engine"]} | {r["variant"]} | {r["scale"]}/{r["layout"]} | {r["requested_threads"]} | {r["reported_blas_threads"]} | {r["median_ns"]/1e6:.6f} | {r["p95_ns"]/1e6:.6f} | {r["max_normalized_error"]:.3e} | {r["multi_core_speedup_vs_t1"]} |')
    report += ['', f'失败/跳过任务：{len(failures)}。详见 failures.json 与 logs/。',
               '正确性使用独立 NumPy FP64 全输出参考；CBLAS 额外验证四种转置、padding、alpha/beta 和 Fortran sgemm_ ABI。',
               'OMPT 在独立诊断进程记录并行区实际 team；0 表示该区间未捕获并行 team，不把 /proc 活跃线程当作计算线程。']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    status = 'COMPATIBILITY_RUN_FINISHED_SCALING_RESOURCE_LIMITED' if capacity < max(args.thread_list) else 'RUN_FINISHED'
    if failures: status += '_WITH_FAILURES'
    (output/'status.txt').write_text(status+'\n')
    print('COMPLETE '+str(output)+' '+status, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path('/root/benchmark/pytorch-kunpeng'))
    p.add_argument('--output', type=Path)
    p.add_argument('--engines', nargs='+', choices=['cblas','torch'], default=['cblas','torch'])
    p.add_argument('--scales', nargs='+', choices=list(SCALES), default=list(SCALES))
    p.add_argument('--thread-list', nargs='+', type=int, default=[1,2,4,8,32])
    p.add_argument('--campaigns', type=int, default=3)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeat', type=int, default=50)
    p.add_argument('--timeout', type=int, default=90)
    p.add_argument('--audit-imports', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--engine', choices=['cblas','torch'])
    p.add_argument('--variant', choices=['locking','multi'])
    p.add_argument('--scale', choices=list(SCALES))
    p.add_argument('--layout', default='NN', choices=['NN','X0','X4'])
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--campaign', type=int, default=0)
    p.add_argument('--library', type=Path)
    p.add_argument('--observer', type=Path)
    p.add_argument('--worker-output', type=Path)
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--native-backtrace', action='store_true')
    args = p.parse_args()
    if min([args.repeat,args.campaigns,args.threads,*args.thread_list]) < 1 or args.warmup < 0:
        p.error('positive repeat/campaign/thread counts and nonnegative warmup required')
    worker(args) if args.worker else coordinator(args)


if __name__ == '__main__':
    main()
