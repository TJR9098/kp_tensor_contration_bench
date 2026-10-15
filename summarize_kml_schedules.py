#!/usr/bin/env python3
"""Validate explicit KML schedules, report per-case and historical comparisons."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import xml.etree.ElementTree as ET

import summarize_tblis_batch_experiment as old
from summarize_tblis_experiment import write_cdf_grid

INTER='kml-multi-inter';INTRA='kml-multi-intra';NEW=[INTER,INTRA]
HIST=['pytorch_2.5.0_openblas','pytorch_2.5.0_kml_patched','tblis_2.0_armsve_intra','tblis_2.0_armsve_inter']
METHODS=[*NEW,*HIST]
LABELS={INTER:'KML-inter/new',INTRA:'KML-intra/new',HIST[0]:'OpenBLAS/old',
        HIST[1]:'KML-lock/old',HIST[2]:'TBLIS-intra/old',HIST[3]:'TBLIS-inter/old'}
COLORS={INTER:'#e45756',INTRA:'#006699',HIST[0]:'#888888',HIST[1]:'#f2a541',HIST[2]:'#2f9e44',HIST[3]:'#b279a2'}
read=old.read_tsv;write=old.write_tsv


def main():
    p=argparse.ArgumentParser();p.add_argument('result_dir',type=Path);args=p.parse_args()
    root=args.result_dir.resolve();prov=json.loads((root/'provenance.json').read_text());hist=Path(prov['baseline'])
    if (root/'status.txt').read_text().strip() not in ['DATA_COMPLETE','COMPLETE']:raise RuntimeError('run incomplete')
    for name,digest in prov['source_sha256'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'experiment source drift: {name}')
    for name,key in [('schedule_library','schedule_library_sha256'),('library','library_sha256')]:
        if hashlib.sha256(Path(prov[name]).read_bytes()).hexdigest()!=prov[key]:raise RuntimeError('library drift')
    cases={r['case_id']:r for r in read(root/'case_manifest.tsv')}
    if len(cases)!=120:raise RuntimeError('case coverage')
    raw=old.gather_rows(root/'raw');past_raw=read(hist/'raw.tsv')
    group=defaultdict(list)
    for r in raw:group[r['case_id'],r['method']].append(r)
    if set(group)!={(cid,m) for cid in cases for m in NEW}:raise RuntimeError('240 groups required')
    observed=[];maximum_error=0.
    for (cid,method),rr in group.items():
        case=cases[cid];batch=int(case['batch']);threads=int(case['threads'])
        inner=threads if method==INTRA else 1;outer=1 if method==INTRA else min(batch,threads)
        if len(rr)!=150 or Counter(int(r['campaign']) for r in rr)!={0:50,1:50,2:50}:raise RuntimeError('sample count')
        prev=json.loads((hist/'metadata'/HIST[1]/f'{cid}_c0.json').read_text())
        for campaign in range(3):
            if {int(r['repeat']) for r in rr if int(r['campaign'])==campaign}!=set(range(50)):raise RuntimeError('repeat coverage')
            meta=json.loads((root/'metadata'/method/f'{cid}_c{campaign}.json').read_text())
            for field in ['a_logical_sha256','b_logical_sha256','torch_config_sha256','affinity','warmup','repeat']:
                if meta[field]!=prev[field]:raise RuntimeError(f'{cid}/{method}: mismatch {field}')
            for operand in ['a_entry','b_entry']:
                for field in ['shape','stride_elements','storage_offset_elements']:
                    if meta[operand][field]!=prev[operand][field]:raise RuntimeError('layout mismatch')
            for token,path in [('libkblas',prov['library']),('libtorch_cpu.so',prov['schedule_library'])]:
                found=[x for x in meta['loaded_shared_objects'] if token in x]
                if len(found)!=1 or Path(found[0]).resolve()!=Path(path).resolve():raise RuntimeError('runtime linkage')
            if not math.isfinite(meta['max_rel_error']) or meta['max_rel_error']>1e-4:raise RuntimeError('accuracy')
            maximum_error=max(maximum_error,meta['max_rel_error'])
        diag=json.loads((root/'kml_thread_diagnostics'/method/f'{cid}.json').read_text());ob=diag['observation']
        if ob['sgemm_calls']!=batch or ob['sgemm_calling_threads']!=outer:raise RuntimeError('BLAS dispatch')
        if ob['blas_local_threads_min']!=inner or ob['blas_local_threads_max']!=inner:raise RuntimeError('BLAS thread config')
        if method==INTRA:
            if ob['sgemm_calls_in_outer_region'] or ob['sgemm_entry_active_level_max']!=0:raise RuntimeError('intra called in outer region')
            if threads>1 and ob['kml_max_team']!=threads:raise RuntimeError('intra team')
        elif ob['kml_max_team']>1 or ob['sgemm_entry_active_level_max']>1:raise RuntimeError('inter nested team')
        observed.append({'case_id':cid,'method':method,**diag['plan'],**ob})
    supplemental=list((root/'distinct_validation').glob('*/*.json'))
    if not {(c['case_id'],m) for c in cases.values() if c['scale']=='S' for m in NEW}<={(f.stem,f.parent.name) for f in supplemental}:
        raise RuntimeError('distinct-batch coverage')
    for f in supplemental:
        if json.loads(f.read_text())['max_rel_error']>1e-4:raise RuntimeError('supplemental accuracy')
    material=old.gather_rows(root/'materialization');activities=old.gather_rows(root/'thread_diagnostics')
    if len(material)!=240 or len(activities)!=240:raise RuntimeError('diagnostic count')
    for d in material:
        c=cases[d['case_id']];expected=4*int(c['input_elements']) if c['layout']=='X4' else 0
        if int(d['input_allocation_bytes'])!=expected:raise RuntimeError('materialization mismatch')
    traces=list((root/'traces').glob('*/*.json'))
    if len(traces)!=16:raise RuntimeError(f'expected16 traces got{len(traces)}')
    for f in traces:json.loads(f.read_text())
    if prov['openblas_before']['binary_sha256']!=prov['openblas_after']['binary_sha256']:raise RuntimeError('OpenBLAS drift')
    old.METHODS=METHODS;old.METHOD_LABELS=LABELS;old.METHOD_COLORS=COLORS
    combined=old.summarize_raw([*raw,*past_raw],cases,3)
    old.add_scaling_metrics(combined)
    old.attach_diagnostics(combined,[*activities,*read(hist/'thread_diagnostics.tsv')])
    obs={(r['case_id'],r['method']):r for r in observed}
    for r in combined:
        new=r['method'] in NEW;r['source_run']=root.name if new else hist.name
        r['measurement_epoch']='current' if new else 'historical'
        r['hardware_dram_bandwidth_gbps']='NA'
        r['bandwidth_note']='minimum_effective_bandwidth_gbps is logical-byte estimate, not measured DDR bandwidth'
        if r['method'] in [HIST[0],HIST[1]]:
            # Older PyTorch diagnostics called intra-op T "inner threads" even
            # though the selected BLAS was single-threaded. Preserve that raw
            # value separately and make the unified schedule fields explicit.
            r['historical_reported_inner_threads']=r['configured_inner_threads']
            r['historical_reported_outer_workers']=r['configured_outer_workers']
            r['configured_inner_threads']=1
            r['configured_outer_workers']=1 if r['method']==HIST[0] else min(r['batch'],r['threads'])
            r['schedule_evidence']='historical deployment/source audit; original reported fields retained separately'
        if new:
            o=obs[r['case_id'],r['method']]
            r['observed_gemm_calling_threads']=o['sgemm_calling_threads']
            r['observed_kml_parallel_team']=o['kml_max_team']
            r['schedule_evidence']='current process plan and separate BLAS/OMPT measurement'
    current=[r for r in combined if r['method'] in NEW]
    write(root/'raw.tsv',raw);write(root/'summary.tsv',current);write(root/'combined_summary.tsv',combined)
    write(root/'kml_thread_diagnostics.tsv',observed);write(root/'pytorch_diagnostics_summary.tsv',material)
    by={(r['case_id'],r['method']):r for r in combined}
    comparison=[]
    for cid in cases:
        a,b=[by[cid,m] for m in NEW]
        comparison.append({'case_id':cid,'inter_median_ns':a['median_total_ns'],'intra_median_ns':b['median_total_ns'],
                           'inter_p95_ns':a['p95_total_ns'],'intra_p95_ns':b['p95_total_ns'],
                           'inter_over_intra':a['median_total_ns']/b['median_total_ns'],
                           'best_median_schedule':'inter' if a['median_total_ns']<b['median_total_ns'] else 'intra'})
    write(root/'inter_vs_intra.tsv',comparison)
    same_strategy=[]
    for cid in cases:
        for method,tblis in [(INTER,HIST[3]),(INTRA,HIST[2])]:
            a,b=by[cid,method],by[cid,tblis]
            same_strategy.append({'case_id':cid,'kml_method':method,'tblis_method':tblis,
                'kml_source':root.name,'tblis_source':hist.name,
                'kml_median_ns':a['median_total_ns'],'tblis_median_ns':b['median_total_ns'],
                'kml_p95_ns':a['p95_total_ns'],'tblis_p95_ns':b['p95_total_ns'],
                'tblis_over_kml_median':b['median_total_ns']/a['median_total_ns'],
                'tblis_over_kml_p95':b['p95_total_ns']/a['p95_total_ns']})
    write(root/'same_strategy_comparison.tsv',same_strategy)
    # Full comparisons with explicit source labels; no historical timings called new.
    cdf_groups=defaultdict(list)
    for row in [*raw,*past_raw]:cdf_groups[row['case_id'],row['method']].append(int(row['total_ns']))
    panels=[];points=[];(root/'cdf').mkdir(exist_ok=True)
    (root/'cdf_kml_only').mkdir(exist_ok=True)
    for cid in cases:
        curves=[]
        for method in METHODS:
            values=sorted(cdf_groups[cid,method]);assert len(values)==150
            curves.append({'legend':LABELS[method],'color':COLORS[method],'values':values})
            points += [{'case_id':cid,'method':method,'total_ns':v,'cdf':(i+1)/len(values),
                        'source_run':root.name if method in NEW else hist.name} for i,v in enumerate(values)]
        panels.append((cid,curves))
        write_cdf_grid(root/'cdf'/f'{cid}.svg',cid,[(cid,curves)],2,
                       subtitle='x: entire-batch E2E us; y: ECDF. Both KML schedules new, other curves historical.')
        write_cdf_grid(root/'cdf_kml_only'/f'{cid}.svg',cid,[(cid,curves[:2])],1,
                       subtitle='New KML only. x: entire-batch E2E us; y: ECDF; 150 samples per curve.')
    write_cdf_grid(root/'cdf_overview.svg','Explicit KML inter/intra: E2E CDF',panels,5,x_scale='log',
                   subtitle='120 cases; 150 samples per curve; historical methods clearly labelled old')
    write_cdf_grid(root/'cdf_kml_only_overview.svg','KML inter vs intra: new samples only',
                   [(cid,curves[:2]) for cid,curves in panels],5,x_scale='log',
                   subtitle='x: entire-batch E2E us (log); y: ECDF; same build and same workload')
    write(root/'cdf_points.tsv',points)
    (root/'cdf/index.md').write_text('# 逐用例端到端 CDF\n\n'+'\n'.join(f'- [{cid}]({cid}.svg)' for cid in cases)+'\n')
    # Existing renderer recognizes PyTorch diagnostics by prefix; aliases are display-only.
    aliases={m:'pytorch_'+m for m in NEW}
    old.METHODS=[aliases.get(m,m) for m in METHODS]
    old.METHOD_LABELS={aliases.get(m,m):LABELS[m] for m in METHODS}
    timeline_summary=[{**r,'method':aliases.get(r['method'],r['method'])} for r in combined]
    timeline_diags=[{**r,'method':aliases.get(r['method'],r['method'])} for r in [*material,*read(hist/'materialization_diagnostics.tsv')]]
    timeline=old.write_timeline(root,timeline_summary,timeline_diags)
    for r in timeline:
        r['method']=next((m for m,a in aliases.items() if a==r['method']),r['method'])
    write(root/'timeline_selected.tsv',timeline)
    for f in [root/'cdf_overview.svg',root/'cdf_kml_only_overview.svg',root/'timeline_batch.svg',
              *list((root/'cdf').glob('*.svg')),*list((root/'cdf_kml_only').glob('*.svg'))]:ET.parse(f)
    def val(r,key,percent=False):
        v=r.get(key,'NA')
        if v in ('NA','N/A',None):return 'N/A'
        if percent:return f'{100*float(v):.3f}'
        if key.endswith('gflops'):return f'{float(v):.3f}'
        if key=='max_rel_error':return f'{float(v):.3e}'
        return str(v)
    lines=['# PyTorch KML multi：inter/intra 完整对照','',
           '不变量：S/M/L/XL、X0/X4、B1/B4/B32、T1/T2/T4/T8/T32，FP32、输入、affinity、完整 einsum 计时一致。',
           '变量：同一份隔离 PyTorch 库在 C++ BMM 层选择 batch 并行或串行，并相应配置 KML 内部线程。',
           'inter：始终按 batch 外层并行，KML inner=1；intra：始终顺序处理 batch，KML inner=T。B1 也分别测试这两个策略。',
           '本次每种方法 120 配置×3轮×50样本，共 36,000 个正式样本，每轮预热10次。没有用 Python 拆成多次 einsum。',
           'OpenBLAS 和原安装 PyTorch 二进制未修改；OpenBLAS/TBLIS/旧locking 使用历史数据，明确标为 old。',
           '统一TSV中历史PyTorch的线程配置按已核验的单线程BLAS/外层策略标明；原报告的框架线程字段另保存在historical_reported_*列，避免将intra-op T误当作BLAS内部线程。',
           '旧计时循环与输出生命周期沿用；正确性在每轮结束后全输出校验，并有 batch 值不同的补充验证。',
           'KML 内部阶段不可可靠拆分，PyTorch 主表阶段保持 N/A；profiler 诊断单列。硬件 DDR 带宽不可用，minimum byte rate 不是实测内存带宽。','',
           '## 完整性能表','',
           '| 用例 | 方法/来源 | 总中位数 ns | P95 ns | Transform ns | Packing ns | Kernel ns | Restore ns | Kernel GFLOP/s | E2E GFLOP/s | Transform % | Packing % | Kernel % | 最大归一化误差 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in combined:
        fields=[r['case_id'],LABELS[r['method']]]+[val(r,k) for k in ['median_total_ns','p95_total_ns','transform_ns','packing_ns','kernel_ns','restore_ns','kernel_gflops','total_gflops']]
        fields += [val(r,k,True) for k in ['transform_ratio','packing_ratio','kernel_ratio']]+[val(r,'max_rel_error')]
        lines.append('| '+' | '.join(fields)+' |')
    lines += ['', '## PyTorch 专用诊断表','',
              '以下仅来自独立 profiler/dispatch 诊断，不与正式中位数相加；BMM container 不是纯 kernel。','',
              '| 用例 | 方法 | Planner ns | 输入物化 ns | BMM ns | 输出物化 ns | 输出元数据 ns | 输入物化字节 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in material:
        lines.append('| '+' | '.join([r['case_id'],r['method'],*[r[k] for k in ['planner_self_ns','input_materialize_self_ns','bmm_container_ns','output_materialize_self_ns','output_metadata_self_ns','input_allocation_bytes']]])+' |')
    lines += ['', '## 数据文件','',
              '- summary.tsv：240个新方法组；raw.tsv：36000个新正式样本。',
              '- combined_summary.tsv：720组，逐行保存历史/本次来源。',
              '- inter_vs_intra.tsv：120个同配置策略对照。',
              '- same_strategy_comparison.tsv：KML与历史TBLIS在相同策略、相同请求线程下的240组对照。',
              '- kml_thread_diagnostics.tsv：BLAS调用线程、内部team及嵌套检查。',
              '- pytorch_diagnostics_summary.tsv、traces/、timeline_batch.svg：独立诊断。',
              '- cdf_overview.svg、cdf/index.md、cdf_points.tsv：完整端到端CDF。']
    lines += ['- cdf_kml_only_overview.svg、cdf_kml_only/：仅两种新KML策略的CDF，避免历史慢路径压缩图形比例。']
    (root/'suites.MD').write_text('\n'.join(lines)+'\n')
    analysis=['# KML inter/intra：实验分析','',
              '## T32：策略对照','',
              '不变量：每行固定规模、布局、batch、T32及逻辑值；变量为 inter/intra 调度。TBLIS 列是历史数据。','',
              '| 用例 | KML-inter/ms | KML-intra/ms | inter/intra | 历史 TBLIS-inter/ms | 历史 TBLIS-intra/ms |',
              '|---|---:|---:|---:|---:|---:|']
    for c in cases.values():
        if c['threads']!='32':continue
        cid=c['case_id'];a,b=by[cid,INTER],by[cid,INTRA]
        analysis.append(f"| {cid} | {a['median_total_ns']/1e6:.6f} | {b['median_total_ns']/1e6:.6f} | {a['median_total_ns']/b['median_total_ns']:.3f}x | {by[cid,HIST[3]]['median_total_ns']/1e6:.6f} | {by[cid,HIST[2]]['median_total_ns']/1e6:.6f} |")
    xl0a,xl0b=by['XL_X0_B4_T32',INTER],by['XL_X0_B4_T32',INTRA]
    xl4a,xl4b=by['XL_X4_B4_T32',INTER],by['XL_X4_B4_T32',INTRA]
    conclusion=[f"结论：XL/X0/B4/T32 的 inter/intra 中位数分别为 {xl0a['median_total_ns']/1e6:.6f}/{xl0b['median_total_ns']/1e6:.6f} ms，时间比 {xl0a['median_total_ns']/xl0b['median_total_ns']:.3f}x。",
                f"XL/X4/B4/T32 对应 {xl4a['median_total_ns']/1e6:.6f}/{xl4b['median_total_ns']/1e6:.6f} ms，时间比 {xl4a['median_total_ns']/xl4b['median_total_ns']:.3f}x。",
                '这是同一天、同一构建与相同数学工作量下的策略差异；库间比较仍需注意 TBLIS 是历史测量且输出分配口径不同。','']
    tail_example=by['M_X0_B1_T32',INTRA]
    conclusion += [f"注意：部分小规模高线程intra存在明显长尾。例如 M/X0/B1/T32 中位数 {tail_example['median_total_ns']/1e6:.6f} ms，但 P95 为 {tail_example['p95_total_ns']/1e6:.6f} ms。正确性通过不等于所有延迟指标都改善。",'']
    analysis[6:6]=conclusion
    analysis += ['', '## 各策略的中位数与 P95','',
                 '不变量：同一用例；变量为策略及其内部线程安排。中位数最快不等于 P95 最低。','',
                 f"结论：在120个固定线程配置中，intra有{sum(r['inter_over_intra']>1 for r in comparison)}个中位数低于inter。该数量仅描述本次排序，不代表每个微小差异都有统计显著性。",'',
                 '| 用例 | inter中位数/ms | intra中位数/ms | inter P95/ms | intra P95/ms |',
                 '|---|---:|---:|---:|---:|']
    for cid in cases:
        a,b=by[cid,INTER],by[cid,INTRA]
        analysis.append(f"| {cid} | {a['median_total_ns']/1e6:.6f} | {b['median_total_ns']/1e6:.6f} | {a['p95_total_ns']/1e6:.6f} | {b['p95_total_ns']/1e6:.6f} |")
    analysis += ['', '## 最佳已测策略和线程','',
                 '固定单项规模、布局、batch；在已测 T1/T2/T4/T8/T32 和 inter/intra 中选择最低中位数。只是有限候选中的最优，未做显著性检验。','',
                 '结论：选择必须结合规模与batch，以下逐用例列出实际获胜的策略和线程；不将所有工作负载压缩成一个加速比。','',
                 '| 规模/布局/batch | KML策略 | KML T | KML/ms | 历史TBLIS策略 | TBLIS T | TBLIS/ms |',
                 '|---|---|---:|---:|---|---:|---:|']
    for scale in ['S','M','L','XL']:
        for layout in ['X0','X4']:
            for batch in [1,4,32]:
                rr=[r for r in combined if (r['scale'],r['layout'],r['batch'])==(scale,layout,batch)]
                a=min([r for r in rr if r['method'] in NEW],key=lambda r:r['median_total_ns'])
                b=min([r for r in rr if r['method'] in HIST[2:]],key=lambda r:r['median_total_ns'])
                analysis.append(f"| {scale}/{layout}/B{batch} | {a['method']} | {a['threads']} | {a['median_total_ns']/1e6:.6f} | {b['method']} | {b['threads']} | {b['median_total_ns']/1e6:.6f} |")
    analysis += ['', '## 尾延迟较突出的新配置','',
                 '不变量：同一方法和用例的150个正式样本；比较中位数与P95两个统计量，不新增实验变量。',
                 '结论：策略评价还需检查高分位延迟。以下列出每种策略P95/中位数最高的五个配置，不能只看中位数排序。','',
                 '| 用例 | 方法 | 中位数/ms | P95/ms | P95/中位数 |','|---|---|---:|---:|---:|']
    for method in NEW:
        for r in sorted([r for r in current if r['method']==method],key=lambda r:r['p95_total_ns']/r['median_total_ns'],reverse=True)[:5]:
            analysis.append(f"| {r['case_id']} | {method} | {r['median_total_ns']/1e6:.6f} | {r['p95_total_ns']/1e6:.6f} | {r['p95_total_ns']/r['median_total_ns']:.3f} |")
    analysis += ['', 'KML 库的单 GEMM 内部时间仍不可分拆；不要把 BMM 或 BLAS边界时间称为纯内核时间。',
                 '两种策略使用同一隔离 libtorch_cpu；原 KML安装和官方OpenBLAS轮子未修改。历史输入周期性问题由独立 batch 不同的校验补充，性能输入保持一致。',
                 'OMP team=0 表示未创建并行team，单线程计算仍在调用线程执行。未测混合 inter×intra 或在线请求排队。']
    (root/'analysis.md').write_text('\n'.join(analysis)+'\n')
    validation=['PASS: 120 cases x 2 schedules x 3 campaigns x 50 = 36000 formal samples',
                'PASS: same mathematical inputs, layout strides, affinity and original full-einsum timing loop',
                'PASS: correct isolated torch and KML mappings for all 720 formal processes',
                f'PASS: full-output FP64 reference error <= 1e-4; max={maximum_error:.12e}',
                'PASS: distinct-batch supplementary correctness and finite-output checks',
                'PASS: 240 BLAS/OMPT diagnostics: exact batch calls, inter callers/inner=1 and intra serial caller/inner=T',
                'PASS: 240 profiler/materialization diagnostics, 16 Chrome traces, CDF/timeline SVGs',
                'PASS: installed PyTorch library and OpenBLAS binary hashes unchanged',
                'INFO: TBLIS/OpenBLAS/locking comparisons are historical, not new runs']
    (root/'validation.log').write_text('\n'.join(validation)+'\n')
    (root/'report_provenance.json').write_text(json.dumps({'current_run':str(root),'historical_run':str(hist),
        'new_samples':len(raw),'historical_samples':len(past_raw),'combined_groups':len(combined),
        'report_sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(old.__file__),Path(__file__).parent/'summarize_tblis_experiment.py']}},indent=2)+'\n')
    (root/'status.txt').write_text('COMPLETE\n')
    print(json.dumps({'result':str(root),'groups':len(current),'samples':len(raw),'max_rel_error':maximum_error}))


if __name__=='__main__':main()
