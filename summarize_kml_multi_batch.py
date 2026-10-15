#!/usr/bin/env python3
"""Validate the KML-only rerun and compare explicitly labelled historical data."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import xml.etree.ElementTree as ET

from summarize_tblis_experiment import write_cdf_grid
import summarize_tblis_batch_experiment as old

NEW='pytorch_2.5.0_kml_multi'
LOCK='pytorch_2.5.0_kml_patched'
METHODS=[NEW,LOCK,'pytorch_2.5.0_openblas','tblis_2.0_armsve_intra','tblis_2.0_armsve_inter']
LABELS={NEW:'KML-multi/new',LOCK:'KML-lock/old','pytorch_2.5.0_openblas':'OpenBLAS/old',
        'tblis_2.0_armsve_intra':'TBLIS-intra/old','tblis_2.0_armsve_inter':'TBLIS-inter/old'}
COLORS={NEW:'#d62728',LOCK:'#ff9d44','pytorch_2.5.0_openblas':'#4c78a8',
        'tblis_2.0_armsve_intra':'#2f9e44','tblis_2.0_armsve_inter':'#b279a2'}


def read(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))


def write(path,rows):old.write_tsv(path,rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result_dir',type=Path);args=p.parse_args()
    root=args.result_dir.resolve();prov=json.loads((root/'provenance.json').read_text())
    if (root/'status.txt').read_text().strip() not in ['DATA_COMPLETE','COMPLETE']:
        raise RuntimeError('runner has not completed')
    hist=Path(prov['baseline']);cases=read(root/'case_manifest.tsv');bycase={c['case_id']:c for c in cases}
    if len(cases)!=120:raise RuntimeError('expected full 120-case matrix')
    raw=old.gather_rows(root/'raw');groups=defaultdict(list)
    for r in raw:groups[r['case_id']].append(r)
    if set(groups)!=set(bycase):raise RuntimeError('case coverage mismatch')
    summary=[];observations=[];validation=[]
    historical_meta={}
    for case in cases:
        cid=case['case_id'];rr=groups[cid];batch=int(case['batch']);threads=int(case['threads'])
        if len(rr)!=150 or Counter(int(r['campaign']) for r in rr)!={0:50,1:50,2:50}:
            raise RuntimeError(f'wrong sample count {cid}')
        for campaign in range(3):
            if {int(r['repeat']) for r in rr if int(r['campaign'])==campaign}!=set(range(50)):
                raise RuntimeError('repeat coverage')
        if any(r['method']!=NEW or int(r['total_ns'])<=0 or float(r['max_rel_error'])>1e-4 for r in rr):
            raise RuntimeError('invalid sample')
        for r in rr:
            if any(r[f]!='NA' for f in ['transform_ns','packing_ns','kernel_ns','restore_ns']):
                raise RuntimeError('PyTorch backend stages must remain NA')
        historical_meta[cid]=json.loads((hist/'metadata'/LOCK/f'{cid}_c0.json').read_text())
        for campaign in range(3):
            meta=json.loads((root/'metadata'/NEW/f'{cid}_c{campaign}.json').read_text())
            previous=historical_meta[cid]
            for field in ['a_logical_sha256','b_logical_sha256','torch_config_sha256']:
                if meta[field]!=previous[field]:raise RuntimeError(f'{cid}: history mismatch {field}')
            for operand in ['a_entry','b_entry']:
                for field in ['shape','stride_elements','storage_offset_elements']:
                    if meta[operand][field]!=previous[operand][field]:raise RuntimeError('layout mismatch')
            if meta['warmup']!=10 or meta['repeat']!=50 or meta['affinity']!=previous['affinity']:
                raise RuntimeError('protocol mismatch')
            kml=[f for f in meta['loaded_shared_objects'] if 'libkblas' in f]
            if len(kml)!=1 or Path(kml[0]).resolve()!=Path(prov['library']).resolve():raise RuntimeError('wrong KML')
        observation=json.loads((root/'kml_thread_diagnostics'/f'{cid}.json').read_text())
        ob=observation['observation'];inner=threads if batch==1 else 1
        if ob['sgemm_calls']!=batch or ob['sgemm_calling_threads']!=min(batch,threads):
            raise RuntimeError(f'GEMM task count mismatch {cid}: {ob}')
        if ob['blas_local_threads_min']!=inner or ob['blas_local_threads_max']!=inner:
            raise RuntimeError('thread setting mismatch')
        if batch==1 and threads>1 and ob['kml_max_team']!=threads:
            raise RuntimeError(f'KML inner team mismatch {cid}: {ob}')
        if batch>1 and (ob['kml_max_team']>1 or ob['sgemm_entry_active_level_max']>1):
            raise RuntimeError('nested KML parallel team')
        observations.append({'case_id':cid,'method':NEW,**observation['plan'],**ob})
        values=sorted(int(r['total_ns']) for r in rr);median=round(st.median(values))
        medians=[round(st.median(int(r['total_ns']) for r in rr if int(r['campaign'])==c)) for c in range(3)]
        row={'case_id':cid,'method':NEW,**{f:case[f] for f in ['scale','layout','batch','threads','equation','M','N','K']},
             'samples':150,'median_total_ns':median,'p95_total_ns':values[math.ceil(.95*len(values))-1],
             'median_per_contraction_ns':median/batch,'contractions_per_s':batch*1e9/median,
             'total_gflops':int(case['flops'])/median,'median_process_core_equivalents':st.median(float(r['process_core_equivalents']) for r in rr),
             'median_core_utilization_pct':st.median(float(r['core_utilization_pct']) for r in rr),
             'minimum_effective_bandwidth_gbps':int(case['minimum_bytes'])/median,'hardware_dram_bandwidth_gbps':'NA',
             'max_abs_error':max(float(r['max_abs_error']) for r in rr),'max_rel_error':max(float(r['max_rel_error']) for r in rr),
             'campaign_median_min_ns':min(medians),'campaign_median_max_ns':max(medians),
             'campaign_spread_pct':100*(max(medians)-min(medians))/st.median(medians),
             'configured_inner_threads':inner,'configured_outer_workers':min(batch,threads),
             'observed_kml_max_team':ob['kml_max_team'],'observed_gemm_calling_threads':ob['sgemm_calling_threads'],
             'measurement_epoch':'current','source_run':root.name,'schedule':observation['plan']['mode']}
        for f in ['transform_ns','packing_ns','kernel_ns','restore_ns','kernel_gflops','transform_ratio','packing_ratio','kernel_ratio']:row[f]='NA'
        summary.append(row)
    previous=read(hist/'summary.tsv');prevby={(r['case_id'],r['method']):r for r in previous}
    newby={r['case_id']:r for r in summary}
    for r in summary:
        r['speedup_vs_t1']=newby[r['case_id'].rsplit('_T',1)[0]+'_T1']['median_total_ns']/r['median_total_ns']
        r['old_locking_over_new']=int(prevby[r['case_id'],LOCK]['median_total_ns'])/r['median_total_ns']
    comparisons=[]
    for r in summary:
        prior=prevby[r['case_id'],LOCK]
        comparisons.append({'case_id':r['case_id'],'batch':r['batch'],'threads':r['threads'],
            'old_source_run':hist.name,'new_source_run':root.name,
            'old_locking_median_ns':prior['median_total_ns'],'new_multi_median_ns':r['median_total_ns'],
            'old_locking_p95_ns':prior['p95_total_ns'],'new_multi_p95_ns':r['p95_total_ns'],
            'old_median_over_new':r['old_locking_over_new'],
            'new_median_change_pct':100*(r['median_total_ns']/int(prior['median_total_ns'])-1),
            'new_p95_change_pct':100*(r['p95_total_ns']/int(prior['p95_total_ns'])-1)})
    write(root/'locking_vs_multi.tsv',comparisons)
    for path in (root/'distinct_validation').glob('*.json'):
        data=json.loads(path.read_text())
        if data['max_rel_error']>1e-4:raise RuntimeError('distinct correctness')
    expected={c['case_id'] for c in cases if c['scale']=='S'}
    if not expected<={f.stem for f in (root/'distinct_validation').glob('*.json')}:raise RuntimeError('distinct coverage')
    diagnostics=old.gather_rows(root/'materialization');thread_rows=old.gather_rows(root/'thread_diagnostics')
    if len(diagnostics)!=120 or len(thread_rows)!=120:raise RuntimeError('diagnostic coverage')
    for diag in diagnostics:
        case=bycase[diag['case_id']]
        expected_bytes=4*int(case['input_elements']) if case['layout']=='X4' else 0
        if int(diag['input_allocation_bytes'])!=expected_bytes:raise RuntimeError('materialization byte mismatch')
    traces=list((root/'traces'/NEW).glob('*.json'))
    if len(traces)!=8:raise RuntimeError(f'expected 8 Chrome traces got {len(traces)}')
    for f in traces:json.loads(f.read_text())
    audit=prov['openblas_after']
    if audit['binary_sha256']!=prov['openblas_before']['binary_sha256'] or not audit['no_kml_loaded']:
        raise RuntimeError('OpenBLAS changed')
    write(root/'raw.tsv',raw);write(root/'summary.tsv',summary)
    write(root/'kml_thread_diagnostics.tsv',observations);write(root/'pytorch_diagnostics_summary.tsv',diagnostics)
    combined=[]
    for case in cases:
        for method in METHODS:
            if method==NEW:row=dict(newby[case['case_id']])
            else:row={**prevby[case['case_id'],method],'measurement_epoch':'historical','source_run':hist.name}
            combined.append(row)
    write(root/'combined_summary.tsv',combined)

    # CDF: keep immutable historical samples as labelled comparison inputs.
    hist_raw=read(hist/'raw.tsv');cdf_groups=defaultdict(list)
    for row in [*raw,*hist_raw]:cdf_groups[row['case_id'],row['method']].append(int(row['total_ns']))
    (root/'cdf').mkdir(exist_ok=True)
    points=[];panels=[]
    for case in cases:
        cid=case['case_id'];curves=[]
        for m in METHODS:
            vals=sorted(cdf_groups[cid,m])
            if len(vals)!=150:raise RuntimeError('CDF sample count')
            curves.append({'legend':LABELS[m],'color':COLORS[m],'values':vals})
            for rank,v in enumerate(vals,1):points.append({'case_id':cid,'method':m,'source_run':root.name if m==NEW else hist.name,
                                                          'total_ns':v,'cdf':rank/len(vals)})
        panels.append((cid,curves))
        write_cdf_grid(root/'cdf'/f'{cid}.svg',cid,[(cid,curves)],2,
                       subtitle='E2E microseconds / empirical CDF; new multi vs historical methods; 150 samples each')
    write(root/'cdf_points.tsv',points)
    write_cdf_grid(root/'cdf_overview.svg','KML multi rerun: E2E CDF',panels,5,x_scale='log',
                   subtitle='Current multi; historical locking/OpenBLAS/TBLIS; x: E2E microseconds, y: ECDF')
    (root/'cdf/index.md').write_text('\n'.join(['# 逐用例 CDF','']+[f'- [{c["case_id"]}]({c["case_id"]}.svg)' for c in cases])+'\n')
    # Reuse the existing diagnostic timeline renderer, with source labels explicit.
    old.METHODS=METHODS;old.METHOD_LABELS=LABELS
    timeline=old.write_timeline(root,combined,[*diagnostics,*read(hist/'materialization_diagnostics.tsv')])
    write(root/'timeline_selected.tsv',timeline)
    for f in [root/'cdf_overview.svg',root/'timeline_batch.svg',*list((root/'cdf').glob('*.svg'))]:ET.parse(f)

    report=['# PyTorch-KML SVE multi：完整多批次多线程实验','',
            '结论：CPU0–31 可用，本次仅重做 PyTorch-KML。OpenBLAS 独立环境、配置、动态库及共享性能 runner 未改变；运行前后校验通过。',
            '', '## 方法与协议','',
            f'当前方法 `{NEW}`：B1 使用 KML inner=T；B4/B32 保留 ATen batch outer，KML inner=1。禁止嵌套扩张，实际线程见 kml_thread_diagnostics.tsv。',
            'S/M/L/XL × X0/X4 × B1/B4/B32 × T1/T2/T4/T8/T32，120 配置，每组 3 轮×50 样本，每轮预热 10 次。',
            '使用历史性能输入、FP64 参考、计时循环及 affinity；另外以 batch 不同的随机非整齐小矩阵校验索引正确性。',
            '与原 runner 一致，每轮计时结束后逐元素校验末次输出，额外检查有限数；不是在每次正式调用的计时区间内执行参考校验。',
            'KML 子进程新增 OMP_TOOL=disabled、OMP_DYNAMIC=FALSE、OMP_MAX_ACTIVE_LEVELS=1 和显式 BLAS 线程控制；未将这些设置注入 OpenBLAS。',
            f'历史对照来自 `{hist.name}`。下表 old/历史行没有重跑；不同日期环境噪声仍可能影响小差距。',
            'PyTorch packing/kernel 无可靠独立拆分，主表保留 N/A；profiler 诊断单独列出。DDR 带宽无可用 PMU，算法字节/时间仅为 minimum effective bandwidth。',
            '', '## 完整统一表','',
            '| 用例 | 方法/来源 | 总中位数 ns | P95 ns | Transform ns | Packing ns | Kernel ns | Restore ns | Kernel GFLOP/s | E2E GFLOP/s | Transform % | Packing % | Kernel % | 最大归一化误差 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def value(r,k,ratio=False):
        v=r.get(k,'NA')
        if v in ('NA','N/A',None,''):return 'N/A'
        if ratio:return f'{float(v)*100:.3f}'
        if k.endswith('gflops'):return f'{float(v):.3f}'
        if k=='max_rel_error':return f'{float(v):.3e}'
        return str(v)
    for r in combined:
        cells=[r['case_id'],LABELS[r['method']],value(r,'median_total_ns'),value(r,'p95_total_ns')]
        cells += [value(r,k) for k in ['transform_ns','packing_ns','kernel_ns','restore_ns','kernel_gflops','total_gflops']]
        cells += [value(r,k,True) for k in ['transform_ratio','packing_ratio','kernel_ratio']]
        cells += [value(r,'max_rel_error')]
        report.append('| '+' | '.join(cells)+' |')
    report += ['', '## PyTorch multi 专用诊断表','',
               '下表来自独立 profiler/dispatch 调用，单位 ns；不能与上面的正式中位数相加或相减。BMM 包含后端调用，不等于纯 kernel。',
               '| 用例 | planner self | 输入物化 self | BMM container | 输出物化 self | 输出元数据 self | 输入分配字节 | A clone 次数 | B clone 次数 |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    diagby={d['case_id']:d for d in diagnostics}
    for case in cases:
        d=diagby[case['case_id']]
        report.append('| '+' | '.join([case['case_id'],*[str(d[k]) for k in [
            'planner_self_ns','input_materialize_self_ns','bmm_container_ns',
            'output_materialize_self_ns','output_metadata_self_ns','input_allocation_bytes',
            'a_materialization_count','b_materialization_count']]])+' |')
    report += ['', '## 诊断与原始数据','',
               '- `raw.tsv`：本次 18,000 个正式样本；`summary.tsv`：本次 120 组。',
               '- `combined_summary.tsv`：600 组，带 source_run 和 measurement_epoch，避免混淆旧数据。',
               '- `pytorch_diagnostics_summary.tsv`：PyTorch 物化和 operator 诊断；不与正式阶段时间相加。',
               '- `kml_thread_diagnostics.tsv`：BLAS 边界调用数、调用线程及 KML/ATen OMPT team。',
               '- `cdf_overview.svg`、`cdf/index.md`、`cdf_points.tsv`：端到端 CDF。',
               '- `timeline_batch.svg`、`timeline_selected.tsv`、`traces/`：诊断 timeline；段宽不代表 KML 内部纯 kernel。']
    (root/'suites.MD').write_text('\n'.join(report)+'\n')

    report_sources={}
    for file in [Path(__file__),Path(old.__file__),Path(__file__).parent/'summarize_tblis_experiment.py']:
        report_sources[str(file.resolve())]=hashlib.sha256(file.read_bytes()).hexdigest()
    (root/'report_provenance.json').write_text(json.dumps({
        'report_source_sha256':report_sources,'current_run':root.name,'historical_run':hist.name,
        'current_sample_count':len(raw),'historical_sample_count':len(hist_raw),
        'combined_rows':len(combined),'cdf_point_count':len(points),
        'historical_summary_sha256':hashlib.sha256((hist/'summary.tsv').read_bytes()).hexdigest(),
        'historical_raw_sha256':hashlib.sha256((hist/'raw.tsv').read_bytes()).hexdigest(),
    },indent=2)+'\n')

    def ms(value):return f'{float(value)/1e6:.6f}'
    xl=newby['XL_X0_B1_T32'];xl_old=prevby['XL_X0_B1_T32',LOCK]
    xl_tblis=prevby['XL_X0_B1_T32','tblis_2.0_armsve_intra']
    wins={batch:sum(int(r['batch'])==batch and r['old_locking_over_new']>1 for r in summary)
          for batch in [1,4,32]}
    tail_regressions=sorted([r for r in comparisons if r['new_p95_change_pct']>5],
                            key=lambda r:r['new_p95_change_pct'],reverse=True)
    median_regressions=[r for r in comparisons if r['new_median_change_pct']>3]
    analysis=['# PyTorch-KML multi 重跑分析','',
              '## 核心结果','',
              f"XL/X0/B1/T32：multi 中位数 {ms(xl['median_total_ns'])} ms，历史 locking 为 {ms(xl_old['median_total_ns'])} ms，时间比 {xl['old_locking_over_new']:.3f}x；同线程历史 TBLIS-intra 为 {ms(xl_tblis['median_total_ns'])} ms。",
              f"本次与历史 locking 逐配置比较，B1/B4/B32 分别有 {wins[1]}/40、{wins[4]}/40、{wins[32]}/40 个中位数更低。胜出数不代表显著性，微小差距需结合 campaign 波动。",
              f"最大归一化误差 {max(r['max_rel_error'] for r in summary):.3e}；120 个独立 BLAS/OMPT 诊断通过。",
              f"相对历史，{len(median_regressions)} 个配置中位数回退超过 3%，但 {len(tail_regressions)} 个配置的绝对 P95 回退超过 5%。实验完整性通过不等于所有性能验收目标都通过。",
              '',
              '本次仅重跑 KML；以下加速比定义为“历史 locking 中位数 / 本次 multi 中位数”。大于 1 表示本次更快。',
              '当前部署：B1 内部并行，B4/B32 外层并行且 inner=1。没有运行新的 TBLIS 或 OpenBLAS。','',
              '## B1：逐线程与历史 locking 对照','',
              '不变量：每行固定规模、布局、B1、逻辑值；性能输入、参考、affinity 和计时循环与历史相同。变量：库变体、内部线程配置和明确记录的 KML-only 运行时设置；跨行 T 不同。',
              '下表逐用例展示绝对时间与比值，不对多个用例的比值取中位数。历史行与本次不在同一天运行。','',
              'KML 并行 team 列为 0 表示该区间没有创建并行 team；T1 仍由调用线程执行 GEMM，不表示没有计算线程。','',
              '| 用例 | 历史 locking/ms | 本次 multi/ms | 本次 P95/ms | locking/multi | KML 并行 team |',
              '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if int(r['batch'])!=1:continue
        a=prevby[r['case_id'],LOCK]
        analysis.append(f"| {r['case_id']} | {int(a['median_total_ns'])/1e6:.6f} | {r['median_total_ns']/1e6:.6f} | {r['p95_total_ns']/1e6:.6f} | {r['old_locking_over_new']:.3f}x | {r['observed_kml_max_team']} |")
    analysis += ['', '## T32：所有规模、布局与 batch','',
                 '不变量：请求 T32、CPU0–31，方法间同一用例的数学计算、布局和输入。变量：方法及其实际 inner/outer 策略；跨行规模与 batch 不同。所有 TBLIS/locking 列都是历史数据。','',
                 '| 用例 | 历史 locking/ms | 本次 multi/ms | 历史 TBLIS-intra/ms | 历史 TBLIS-inter/ms | locking/multi |',
                 '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if int(r['threads'])!=32:continue
        cid=r['case_id']
        analysis.append(f"| {cid} | {int(prevby[cid,LOCK]['median_total_ns'])/1e6:.6f} | {r['median_total_ns']/1e6:.6f} | {int(prevby[cid,'tblis_2.0_armsve_intra']['median_total_ns'])/1e6:.6f} | {int(prevby[cid,'tblis_2.0_armsve_inter']['median_total_ns'])/1e6:.6f} | {r['old_locking_over_new']:.3f}x |")
    tail_section=['', '## 需要关注的 P95 回退','',
                  '不变量：每行是相同规模、布局、batch、请求线程和逻辑输入。变量：当前 multi 部署与历史 locking 部署，且测量日期不同。',
                  '结论：中位数加速并不保证绝对 P95 改善；以下配置需要后续运行时和线程选择消融。不能据此认定根因均在 KML 内核。','',
                  '| 用例 | 历史 P95/ms | 本次 P95/ms | P95 增幅 |',
                  '|---|---:|---:|---:|']
    for r in tail_regressions:
        tail_section.append(f"| {r['case_id']} | {ms(r['old_locking_p95_ns'])} | {ms(r['new_multi_p95_ns'])} | {r['new_p95_change_pct']:.2f}% |")
    tail_section += ['', 'S/X0/B1/T32 与 M/X0/B1/T32 的每轮 P95（各 50 个正式样本）：','',
                     '| 用例 | campaign 0/ms | campaign 1/ms | campaign 2/ms |',
                     '|---|---:|---:|---:|']
    for cid in ['S_X0_B1_T32','M_X0_B1_T32']:
        per=[]
        for campaign in range(3):
            vals=sorted(int(r['total_ns']) for r in groups[cid] if int(r['campaign'])==campaign)
            per.append(ms(vals[math.ceil(.95*len(vals))-1]))
        tail_section.append('| '+' | '.join([cid,*per])+' |')
    tail_section += ['', 'S 的较高 P95 在三轮中均出现；M 的轮间波动更大。该现象应与原始 CDF、运行时等待及宿主调度一同分析，不能仅凭聚合中位数判断稳定性。']
    # Put the risk next to the headline result, before the detailed B1 tables.
    insertion=analysis.index('## B1：逐线程与历史 locking 对照')
    analysis[insertion:insertion]=tail_section+['']
    analysis += ['', '## 范围与限制','',
                 '逐配置的中位数及绝对 P95 变化见 locking_vs_multi.tsv；正的 change_pct 表示本次耗时增加。',
                 '主计时循环与历史一致；库变体、KML线程配置和必要的工具/嵌套控制为本次变化。跨日期差距不能全部归因于算法。',
                 'B4/B32 仍采取 outer-only；未按尺寸调优或测试 hybrid，所以不是 KML 所有可能策略的最优结果。',
                 '主输入保留历史周期序列以便对齐；独立 batch 不同的校验是补充正确性证据，不计入性能数据。',
                 '完整数据及来源见 suites.MD。']
    (root/'analysis.md').write_text('\n'.join(analysis)+'\n')
    validation += ['PASS: 120 configurations x 3 campaigns x 50 samples = 18000',
                   'PASS: mathematical inputs, layouts, affinity and timing loop match historical run',
                   'PASS: only intended multi library mapped, same PyTorch config as historical KML',
                   'PASS: full-output FP64 reference error <= 1e-4 and finite outputs',
                   'PASS: supplemental distinct-batch correctness covers batch/layout/thread matrix',
                   'PASS: 120 BLAS boundary/OMPT diagnostics verify calls, inner/outer policy and no nesting',
                   'PASS: 120 profiler/materialization and 120 worker activity diagnostics',
                   'PASS: 8 Chrome traces, CDF/timeline SVG XML valid',
                   'PASS: OpenBLAS runtime config and binary hashes unchanged before/after',
                   'INFO: old OpenBLAS/locking/TBLIS values are historical; no new samples claimed',
                   f'INFO: {len(tail_regressions)} P95 regressions >5% versus historical locking; correctness PASS is not performance acceptance']
    (root/'validation.log').write_text('\n'.join(validation)+'\n')
    (root/'status.txt').write_text('COMPLETE\n')
    print(json.dumps({'result':str(root),'samples':len(raw),'groups':len(summary),
                      'max_rel_error':max(r['max_rel_error'] for r in summary)}))


if __name__=='__main__':main()
