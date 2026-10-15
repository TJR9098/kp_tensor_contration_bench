#!/usr/bin/env python3
"""Merge the supplied historical reports with the completed KML-multi run."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
import xml.etree.ElementTree as ET

from summarize_tblis_experiment import write_cdf_grid
import summarize_tblis_batch_experiment as charts

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'results/tblis_batch_cpu0-31_20261011T074035858907Z'
NEW=ROOT/'results/pytorch_kml_multi_cpu0-31_20261014T101523721193Z'
ATTACH=Path('/root/.codex/attachments/45b6b399-1eba-4d2b-a1d6-423b006f62a2')
OB='pytorch_2.5.0_openblas';LOCK='pytorch_2.5.0_kml_patched';MULTI='pytorch_2.5.0_kml_multi'
INTRA='tblis_2.0_armsve_intra';INTER='tblis_2.0_armsve_inter'
METHODS=[OB,LOCK,MULTI,INTRA,INTER]
LABEL={OB:'PyTorch-OpenBLAS',LOCK:'kml-locking',MULTI:'kml-multi',INTRA:'TBLIS-intra',INTER:'TBLIS-inter'}
COLOR={OB:'#4c78a8',LOCK:'#e69532',MULTI:'#d62728',INTRA:'#2f9e44',INTER:'#b279a2'}
SCALES=['S','M','L','XL'];LAYOUTS=['X0','X4'];BATCHES=[1,4,32];THREADS=[1,2,4,8,32]


def read(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))


def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def rename(text):
    text=text.replace('PyTorch-KML','kml-locking')
    return re.sub(r'(?<![A-Za-z_-])KML(?![A-Za-z_-])','kml-locking',text)


def section(text,title):
    match=re.search(r'^## '+re.escape(title)+r'\n(.*?)(?=^## |\Z)',text,re.M|re.S)
    if not match:raise ValueError('missing template section '+title)
    return match[1].strip()


def abslinks(text,base):
    def sub(m):
        target=m[1]
        if target.startswith(('/','#','http:','https:')):return m[0]
        return ']('+str((base/target).resolve())+')'
    return re.sub(r'\]\(([^)]+)\)',sub,text)


def controls(fixed,variable):
    return f'不变量：{fixed}。变量：{variable}。\n\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-template',type=Path,default=ATTACH/'analysis.md')
    parser.add_argument('--suites-template',type=Path,default=ATTACH/'suites.MD')
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if not args.overwrite and any((out/n).exists() for n in ['analysis.md','suites.MD']):
        raise FileExistsError('output reports exist; use --overwrite after review')
    templates={n:p.read_text() for n,p in [('analysis',args.analysis_template),('suites',args.suites_template)]}
    template_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.analysis_template,args.suites_template]}
    assets=out/'results/kml_reports_merged_20261014';assets.mkdir(parents=True,exist_ok=True)
    summary=read(OLD/'summary.tsv')+read(NEW/'summary.tsv')
    raw=read(OLD/'raw.tsv')+read(NEW/'raw.tsv')
    cases=read(NEW/'case_manifest.tsv');by={(r['case_id'],r['method']):r for r in summary}
    assert len(summary)==600 and len(by)==600 and len(raw)==90000 and len(cases)==120
    groups=defaultdict(list)
    for row in raw:groups[row['case_id'],row['method']].append(row)
    for row in summary:
        cid,m=row['case_id'],row['method'];samples=groups[cid,m];values=sorted(int(r['total_ns']) for r in samples)
        assert len(samples)==150 and Counter(r['campaign'] for r in samples)=={'0':50,'1':50,'2':50}
        assert round(st.median(values))==int(row['median_total_ns'])
        assert values[math.ceil(.95*len(values))-1]==int(row['p95_total_ns'])
        batch,t=int(row['batch']),int(row['threads'])
        row['speedup_vs_t1']=int(by[cid.rsplit('_T',1)[0]+'_T1',m]['median_total_ns'])/int(row['median_total_ns'])
        row['throughput_vs_b1']=batch*int(by[f"{row['scale']}_{row['layout']}_B1_T{t}",m]['median_total_ns'])/int(row['median_total_ns'])
    for name,rows in [('summary.tsv',summary),('raw.tsv',raw)]:
        charts.write_tsv(assets/name,[{**r,'method':LABEL[r['method']],'source_method':r['method'],
                          'source_run':NEW.name if r['method']==MULTI else OLD.name,
                          'measurement_epoch':'new' if r['method']==MULTI else 'historical'} for r in rows])
    diag=read(OLD/'materialization_diagnostics.tsv')+read(NEW/'pytorch_diagnostics_summary.tsv')
    charts.write_tsv(assets/'materialization_diagnostics.tsv',[{**r,'method':LABEL[r['method']],
                     'source_method':r['method']} for r in diag])
    charts.write_tsv(assets/'kml_thread_diagnostics.tsv',[{**r,'method':'kml-multi'} for r in read(NEW/'kml_thread_diagnostics.tsv')])
    # Recreate all figures with the same two KML labels as the documents.
    points=[];cdf_dir=assets/'cdf';cdf_dir.mkdir(exist_ok=True)
    for case in cases:
        cid=case['case_id'];curves=[]
        for method in METHODS:
            vals=sorted(int(r['total_ns']) for r in groups[cid,method])
            curves.append({'legend':LABEL[method],'color':COLOR[method],'values':vals})
            points += [{'case_id':cid,'method':LABEL[method],'total_ns':v,'cdf':i/150} for i,v in enumerate(vals,1)]
        write_cdf_grid(cdf_dir/f'{cid}.svg',cid,[(cid,curves)],2,
                       subtitle='E2E microseconds / ECDF; kml-multi new, other methods historical; 150 samples each')
    charts.write_tsv(assets/'cdf_points.tsv',points)
    panels=[]
    for sc in SCALES:
        for la in LAYOUTS:
            for t in THREADS:
                curves=[]
                for b in BATCHES:
                    for m in METHODS:
                        curves.append({'legend':f'{LABEL[m]}/B{b}','color':COLOR[m],
                            'dash':{1:'',4:'7,4',32:'2,3'}[b],
                            'values':[int(r['total_ns']) for r in groups[f'{sc}_{la}_B{b}_T{t}',m]]})
                panels.append((f'{sc}/{la}/T{t}',curves))
    write_cdf_grid(assets/'cdf_batch_all_overview.svg','五方法端到端 CDF',panels,5,x_scale='log',
                   subtitle='x: E2E microseconds (log); y: ECDF; 90,000 samples; kml-multi new, other methods historical')
    write_cdf_grid(assets/'cdf_batch_t32_overview.svg','T32 五方法端到端 CDF',
                   [(name,curves) for name,curves in panels if name.endswith('/T32')],2,x_scale='log',
                   subtitle='x: E2E microseconds (log); y: ECDF; kml-locking and kml-multi are distinct measurements')
    (cdf_dir/'index.MD').write_text('# 逐用例 CDF\n\n'+'\n'.join(f'- [{c["case_id"]}]({cdf_dir/c["case_id"]}.svg)' for c in cases)+'\n')
    charts.METHODS=METHODS;charts.METHOD_LABELS=LABEL
    timeline=charts.write_timeline(assets,summary,diag)
    charts.write_tsv(assets/'timeline_selected.tsv',[{**r,'method':LABEL[r['method']]} for r in timeline])

    def get(sc,la,b,t,m):return by[f'{sc}_{la}_B{b}_T{t}',m]
    def ns(row):return int(row['median_total_ns'])
    def best(sc,la,b,m):return min([get(sc,la,b,t,m) for t in THREADS],key=ns)
    wins={b:Counter() for b in BATCHES};bestwins={b:Counter() for b in BATCHES}
    for case in cases:
        winner=min(METHODS,key=lambda m:ns(by[case['case_id'],m]));wins[int(case['batch'])][winner]+=1
    for sc in SCALES:
        for la in LAYOUTS:
            for b in BATCHES:bestwins[b][min(METHODS,key=lambda m:ns(best(sc,la,b,m)))]+=1
    allwins=sum(wins.values(),Counter());allbest=sum(bestwins.values(),Counter())
    tail=[r for r in summary if int(r['p95_total_ns'])/ns(r)>=2]
    spread=[r for r in summary if float(r['campaign_spread_pct'])>=10]
    changes=read(NEW/'locking_vs_multi.tsv')
    regress=sorted([r for r in changes if float(r['new_p95_change_pct'])>5],
                   key=lambda r:float(r['new_p95_change_pct']),reverse=True)
    t32_rows=[]
    for sc in SCALES:
        for la in LAYOUTS:
            for b in BATCHES:
                values={m:ns(get(sc,la,b,32,m)) for m in METHODS}
                t32_rows.append([f'{sc}/{la}/B{b}',*[f'{values[m]/1e6:.3f}' for m in METHODS],LABEL[min(values,key=values.get)]])
    t32_table=table(['规模/布局/Batch',*[f'{LABEL[m]}（ms）' for m in METHODS],'最快方法'],t32_rows)
    win_table=table(['Batch',*[LABEL[m] for m in METHODS],'配置数'],
                    [[f'B{b}',*[wins[b][m] for m in METHODS],40] for b in BATCHES]+
                    [['全部',*[allwins[m] for m in METHODS],120]])
    best_rows=[]
    for m in METHODS:
        choices=[best(sc,la,b,m) for sc in SCALES for la in LAYOUTS for b in BATCHES]
        count=Counter(int(r['threads']) for r in choices)
        best_rows.append([LABEL[m],*[count[t] for t in THREADS],f'{st.median(float(r["speedup_vs_t1"]) for r in choices):.3f}x',f'{allbest[m]}/24'])
    best_table=table(['方法',*[f'最优T{t}' for t in THREADS],'最佳线程相对T1中位加速（附表口径）','全局胜出工作负载'],best_rows)
    tail_table=table(['用例','kml-locking P95/ms','kml-multi P95/ms','P95 增幅'],
        [[r['case_id'],f'{int(r["old_locking_p95_ns"])/1e6:.6f}',f'{int(r["new_multi_p95_ns"])/1e6:.6f}',f'{float(r["new_p95_change_pct"]):.2f}%'] for r in regress])
    best_b1=table(['规模/布局','kml-locking 最优T/耗时ms','kml-multi 最优T/耗时ms','TBLIS-intra 最优T/耗时ms','intra/multi'],
        [[f'{sc}/{la}',*[f'T{best(sc,la,1,m)["threads"]} / {ns(best(sc,la,1,m))/1e6:.6f}' for m in [LOCK,MULTI,INTRA]],
          f'{ns(best(sc,la,1,INTRA))/ns(best(sc,la,1,MULTI)):.3f}x'] for sc in SCALES for la in LAYOUTS])
    protocol='''- S/M/L/XL × X0/X4 × B1/B4/B32 × T1/T2/T4/T8/T32，共 120 配置、5 方法、600 方法组。
- 每组 3 个 campaign，每轮预热 10 次、正式记录 50 次；合并 90,000 个样本，其中 kml-multi 的 18,000 个为本次新增。
- 表达式 `pabkl,pklcd->pabcd`，FP32；M=长度(a)×长度(b)，N=长度(c)×长度(d)，K=长度(k)×长度(l)。p 是 batch 索引，P 是 batch 数；增加 P 时单项尺寸不变。
- S/M/L/XL 的 M/N/K 分别为 256/256/64、512/512/128、1024/1024/256、2048/2048/512。
- X0 物理轴序 A=abkl、B=klcd；X4 为 A=akbl、B=kcld；p 轴在最外层。布局间逻辑 shape、值和计算一致。
- T1/T2/T4/T8/T32 允许的 CPU 分别为 {3}、{2,3}、{0,…,3}、{0,…,7}、{0,…,31}。当前 32 vCPU，无 cgroup CPU 配额限制。
- P95 为 nearest-rank；总时间是完整 batch 调用，FLOPs=P×2×M×N×K；总时间/P 是摊销时间，不是逐项响应延迟。
- FP64 参考和性能输入沿用原实验。原序列有周期，不保证不同 batch 数值互异；新增独立的 batch 不同随机校验用于检查索引错误，不混入性能样本。
- 正确性指标为 max(abs(error)/max(abs(reference),1))，门限 1e-4；每轮计时后逐元素校验末次输出，新增有限数检查。
- PyTorch profiler/物化/线程诊断另行采集；内部 packing/kernel 无可靠独立计时，保持 N/A。TBLIS-intra 含本地阶段插桩，inter 并发阶段为 N/A。
- 核心等效数含自旋等 CPU 消耗，不等于 GEMM 计算线程数。硬件 DDR 带宽无可用 PMU；算法字节/时间只能作为最低有效带宽估算。'''
    methods_text='''- `PyTorch-OpenBLAS`：官方 PyTorch 2.5.0 aarch64 CPU wheel，BLAS_INFO=open，OpenBLAS 内部线程为 1；沿用历史样本。
- `kml-locking`：原 KML 框架路径，使用 KML SVE locking；单 GEMM 内部单线程，ATen 按 batch 并行；沿用历史样本。
- `kml-multi`：新增 KML SVE multi 路径。B1 使用 KML inner=T；B4/B32 保留 ATen outer，KML inner=1。设置 OMP_TOOL=disabled、OMP_DYNAMIC=FALSE、OMP_MAX_ACTIVE_LEVELS=1；不是 shape-tuned/hybrid 最优策略。
- `TBLIS-intra`：batch 顺序执行，每项使用 T 个内部线程；沿用历史样本。
- `TBLIS-inter`：min(P,T) 个持久 outer worker，每项内部线程为 1；线程池创建不计入调用，分发和同步计入；沿用历史样本。'''
    source_note=f'''原附件的四方法数据来自 `{OLD.name}`；新 `kml-multi` 数据来自 `{NEW.name}`。
OpenBLAS 配置、库哈希及公共计时 runner 校验未改变，因此本轮只重跑 KML。
合并表是跨日期结果对照，不是五方法在同一轮重跑；小差距不能排除运行环境波动。
所有图表使用统一显示名 `kml-locking` 和 `kml-multi`，底层原始方法 ID 留在配套 TSV 的 source_method 列。'''

    # Preserve every old performance row exactly except the requested KML label.
    perf=rename(section(templates['suites'],'完整性能表')).replace('最大相对误差','最大归一化误差')
    merged_lines=[];preserved=0;added=0
    for line in perf.splitlines():
        merged_lines.append(line)
        if re.match(r'^\| (S|M|L|XL)_X[04]_B\d+_T\d+ \|',line):preserved+=1
        cells=[c.strip() for c in line.strip('|').split('|')]
        if len(cells)==12 and cells[1]=='kml-locking':
            r=by[cells[0],MULTI]
            merged_lines.append(f"| {r['case_id']} | kml-multi | {r['batch']} | {r['threads']} | {r['median_total_ns']} | {r['p95_total_ns']} | {float(r['median_per_contraction_ns']):.1f} | {float(r['total_gflops']):.3f} | {float(r['speedup_vs_t1']):.3f} | {float(r['throughput_vs_b1']):.3f} | {float(r['median_process_core_equivalents']):.3f} | {float(r['max_rel_error']):.3e} |")
            added+=1
    assert preserved==480 and added==120
    checks=read(NEW/'kml_thread_diagnostics.tsv')
    omp_table=table(['用例','方法','请求T','计划Inner','计划Outer','sgemm调用','调用GEMM线程','KML并行team','ATen并行team'],
        [[r['case_id'],'kml-multi',r['requested_threads'],r['configured_blas_inner_threads'],r['configured_outer_gemm_tasks'],
          r['sgemm_calls'],r['sgemm_calling_threads'],r['kml_max_team'],r['torch_max_team']] for r in checks])
    diag_by={(r['case_id'],r['method']):r for r in diag}
    py_table=table(['用例','方法','planner self/ns','输入物化 self/ns','BMM container/ns','输出物化 self/ns','输入物化字节','A clone','B clone'],
        [[c['case_id'],LABEL[m],*[diag_by[c['case_id'],m][f] for f in ['planner_self_ns','input_materialize_self_ns','bmm_container_ns',
          'output_materialize_self_ns','input_allocation_bytes','a_materialization_count','b_materialization_count']]] for c in cases for m in [LOCK,MULTI]])
    tail_rank=sorted(summary,key=lambda r:int(r['p95_total_ns'])/ns(r),reverse=True)[:15]
    tail_rank_table=table(['用例','方法','P95/中位数','campaign 跨度'],
        [[r['case_id'],LABEL[r['method']],f'{int(r["p95_total_ns"])/ns(r):.3f}x',f'{float(r["campaign_spread_pct"]):.2f}%'] for r in tail_rank])
    phase_low=section(templates['suites'],'稳定性警示').split('TBLIS-intra 阶段合计覆盖端到端时间最低的方法组：',1)[1].strip()
    index_text=f'''- [analysis.md]({out/'analysis.md'})：五方法分析；[suites.MD]({out/'suites.MD'})：完整性能表。
- [summary.tsv]({assets/'summary.tsv'})：600 个方法组；[raw.tsv]({assets/'raw.tsv'})：90,000 个正式样本，均带来源。
- [全部 CDF]({assets/'cdf_batch_all_overview.svg'})、[T32 CDF]({assets/'cdf_batch_t32_overview.svg'})、[逐用例索引]({cdf_dir/'index.MD'})、[CDF 点]({assets/'cdf_points.tsv'})。
- [物化诊断]({assets/'materialization_diagnostics.tsv'})、[kml-multi 线程诊断]({assets/'kml_thread_diagnostics.tsv'})。
- [timeline]({assets/'timeline_batch.svg'})、[timeline 数据]({assets/'timeline_selected.tsv'})。
- [历史入口布局审计]({OLD/'layout_audit.tsv'})、[新运行元数据]({NEW/'provenance.json'})。
- [合并验证]({assets/'validation.log'})、[合并来源]({assets/'provenance.json'})；原实验数据不被覆盖。'''
    suite_parts=[('# 大规模多批次多线程张量缩并实验（kml-locking / kml-multi 合并版）',''),
      ('## 实验目标','在原四方法报告中补充 KML multi，比较单项内部并行与 batch 外层并行。\n\n'+source_note),
      ('## 方法实现',methods_text),('## 实验协议',protocol),
      ('## T32 方法对比',controls('每行规模、布局、P、T32 与逻辑输入','方法及其内外层策略')+
       '结论：B1 的 T32 中位数由 kml-multi 领先；XL/B4 仍由历史 TBLIS-intra 领先。下表最快方法已按五方法重新计算。\n\n'+t32_table),
      ('## 完整性能表','保留原 480 行数值，并在对应 kml-locking 后插入 120 行 kml-multi；方法名同时标识测量来源。\n\n'+'\n'.join(merged_lines)),
      ('## TBLIS-intra 阶段统计','以下 120 行沿用历史数据；packing/kernel 为各线程累计耗时最大值，不是严格互斥的墙钟分解。\n\n'+section(templates['suites'],'TBLIS-intra 阶段统计')),
      ('## TBLIS 调度与线程诊断','以下沿用历史诊断，CPU 活跃 worker 可能包含等待，不直接等于有效 GEMM 计算线程。\n\n'+section(templates['suites'],'TBLIS 调度与线程诊断').replace('实际计算worker','CPU活跃worker')),
      ('## kml-multi 实际线程诊断','新增独立 BLAS 边界/OMPT 调用。KML 并行 team=0 表示没有创建并行区，串行调用仍使用一个线程；旧 kml-locking 没有同口径数据，不补造观测值。\n\n'+omp_table),
      ('## PyTorch KML 专用诊断表','下表是独立 profiler/dispatch 诊断，不与正式中位数相加。BMM 包含后端开销，不能称为纯 kernel。\n\n'+py_table),
      ('## 端到端耗时 CDF',f'横轴为完整 batch 端到端耗时，纵轴为经验 CDF；图中新增 kml-multi，原 KML 统一标为 kml-locking。\n\n![T32五方法CDF]({assets/"cdf_batch_t32_overview.svg"})\n\n[逐用例索引]({cdf_dir/"index.MD"})'),
      ('## Timeline',f'![五方法诊断Timeline]({assets/"timeline_batch.svg"})\n\nPyTorch 来自独立 profiler，TBLIS-intra 为阶段统计，inter 仅显示总时间；用于解释路径，不是共同的纯 kernel 计时。'),
      ('## 自动摘要',win_table+'\n\n'+best_table+'\n\n原附件中的 TBLIS inter/intra 对照未重跑，数值不变。T32 下两种 TBLIS 调度的'+section(templates['suites'],'自动摘要').split('T32 下两种 TBLIS 调度的',1)[1]),
      ('## 稳定性警示',f'{len(tail)}/600 个方法组 P95/中位数≥2，{len(spread)}/600 个方法组 campaign 跨度≥10%。\n\nkml-multi 相对 kml-locking 有 8 个配置 P95 增加超过 5%，尽管没有配置中位数回退超过 3%。\n\n'+tail_table+'\n\n五方法的长尾排序：\n\n'+tail_rank_table+
       '\n\n历史 TBLIS-intra 阶段覆盖最低的组（阶段之和不是严格 wall-clock 分解，差额不能直接当作已证实的等待时间）：\n\n'+phase_low),
      ('## 结果文件',index_text)]
    suites='\n\n'.join(title+('\n\n'+body if body else '') for title,body in suite_parts)+'\n'
    analysis=build_analysis(templates['analysis'],summary,raw,by,groups,cases,best,wins,bestwins,
                            win_table,best_table,best_b1,tail_table,tail,spread,assets,out,source_note,methods_text,protocol,index_text)
    (out/'suites.MD').write_text(suites);(out/'analysis.md').write_text(analysis)

    # Verify the full performance table against both the attachment and machine data.
    full=section(suites,'完整性能表')
    rows=[line for line in full.splitlines() if re.match(r'^\| (S|M|L|XL)_X',line)]
    assert len(rows)==600
    for line in section(templates['suites'],'完整性能表').splitlines():
        if re.match(r'^\| (S|M|L|XL)_X',line):assert rename(line) in rows
    inverse={v:k for k,v in LABEL.items()}
    for line in rows:
        cell=[c.strip() for c in line.strip('|').split('|')];r=by[cell[0],inverse[cell[1]]]
        assert int(cell[4])==ns(r) and int(cell[5])==int(r['p95_total_ns'])
        assert cell[9]==f'{float(r["throughput_vs_b1"]):.3f}'
    for path in [out/'analysis.md',out/'suites.MD']:
        content=path.read_text();assert 'PyTorch-KML' not in content
        assert sum(l.startswith('```') for l in content.splitlines())%2==0
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content):
            if target.startswith(('http:','https:','#')):continue
            clean=re.sub(r':\d+$','',target)
            if Path(clean) in [assets/'provenance.json',assets/'validation.log']:continue
            assert Path(clean).exists(),(path,target)
    for p in [*assets.glob('*.svg'),*cdf_dir.glob('*.svg')]:
        ET.parse(p);assert 'PyTorch-KML' not in p.read_text()
    assert template_hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.analysis_template,args.suites_template]}
    provenance={'input_documents_sha256':template_hashes,'historical_run':str(OLD),'new_run':str(NEW),
                'display_names':LABEL,'groups':600,'samples':90000,'preserved_old_performance_rows':480,
                'added_multi_rows':120,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (assets/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    (assets/'validation.log').write_text('\n'.join([
        'PASS: original attachment files untouched',
        'PASS: all 480 historical performance rows preserved except KML display name',
        'PASS: 120 kml-multi rows added; 600 method groups and 90000 samples',
        'PASS: median and nearest-rank P95 recomputed from raw for all 600 groups',
        'PASS: winners, best threads, layout ratios, tails and campaign statistics recomputed',
        'PASS: five-method CDF/timeline labels are kml-locking and kml-multi',
        'PASS: Markdown links exist and generated SVGs parse',
        'INFO: 4-vCPU consistency section is historical and excludes kml-multi'])+'\n')
    print(json.dumps({'analysis':str(out/'analysis.md'),'suites':str(out/'suites.MD'),'groups':600,'samples':90000}))


def build_analysis(template,summary,raw,by,groups,cases,best,wins,bestwins,win_table,best_table,
                   best_b1,tail_table,tail,spread,assets,out,source_note,methods_text,protocol,index_text):
    """Updated narrative, with the historical 4-vCPU and TBLIS audits retained."""
    def ns(r):return int(r['median_total_ns'])
    def get(sc,la,b,t,m):return by[f'{sc}_{la}_B{b}_T{t}',m]
    def scoperows(m,b=None,t=None):
        return [r for r in summary if r['method']==m and (b is None or int(r['batch'])==b)
                and (t is None or int(r['threads'])==t)]
    conclusions='''1. **新增 kml-multi 改变了单 batch 的中位数排名。** 各方法选择最佳已测线程后，B1 的八个规模/布局用例均由 kml-multi 领先；旧报告中 TBLIS-intra 的优势是相对 kml-locking。XL/X0/B1/T32 从 49.127433 ms 降为 2.222337 ms，约快 22.106x；历史 TBLIS-intra 为 5.030086 ms。
2. **多 batch 中位数总体接近 kml-locking。** B4/B32 的 80 个配置中，75 个变化在 ±3% 内。本次 B4/B32 使用 outer、inner=1，没有测试大型 B4 的 inner 或 hybrid 最优策略；XL/B4 仍由 TBLIS-intra 领先。
3. **中位数改善不代表 P95 改善。** kml-multi 有 8 个配置的 P95 比 kml-locking 高出 5% 以上；S/X0/B1/T32 从 0.203245 ms 升至 0.851684 ms，M/X0/B1/T32 从 1.067261 ms 升至 3.553626 ms。
4. **X4 的整输入物化仍存在。** 两种 KML 的 X4 都在 BMM 之前物化 A/B。B1 计算加速后布局成本相对更明显；M/B1/T32 的 X4/X0 从约 1.062x 变为 1.632x，绝对差却仍约 0.055–0.056 ms。
5. **TBLIS 的 inter/intra 结论保留，但不能推广为整体方法排名。** 历史 B32/T32 中 inter 在全部规模优于 intra；加入 multi 后，单次大缩并不能再笼统推荐 TBLIS 优于所有 KML 部署。
6. **4-vCPU 一致性结果仅描述历史四方法。** 旧 69/72 的最快方法一致统计不含 kml-multi；没有 kml-multi 的 4-vCPU 对应实验，不能将该一致性结论扩展到它。'''
    legacy4=abslinks(rename(section(template,'与 4 vCPU 实验的一致性')),OLD)
    legacy4=legacy4.replace('这说明差异不是数学输入、布局物化或单次 kernel 变慢，而是 32 vCPU 环境下\nTBLIS-intra T2 小任务路径的非平稳运行时等待。',
        '逻辑输入与路径审计未发现对应差异，现象支持运行时额外开销的解释；但没有足够证据将其唯一归因于某一种等待机制。')
    scaling_rows=[]
    for b in BATCHES:
        for sc in SCALES:
            for la in LAYOUTS:
                scaling_rows.append([f'{sc}/{la}/B{b}',
                    *[f"{ns(get(sc,la,b,1,m))/1e6:.6f} → {ns(get(sc,la,b,32,m))/1e6:.6f} ({ns(get(sc,la,b,1,m))/ns(get(sc,la,b,32,m)):.3f}x)" for m in METHODS]])
    scaling_table=table(['规模/布局/Batch',*[f'{LABEL[m]}：T1→T32 ms（自身加速）' for m in METHODS]],scaling_rows)
    aggregate_scaling=table(['方法','B1','B4','B32'],
        [[LABEL[m],*[f'{st.median(float(r["speedup_vs_t1"]) for r in scoperows(m,b,32)):.3f}x' for b in BATCHES]] for m in METHODS])
    throughput_rows=[]
    for sc in SCALES:
        for la in LAYOUTS:
            row=[f'{sc}/{la}']
            for m in METHODS:
                row.append(' / '.join(f'{float(get(sc,la,b,32,m)["throughput_vs_b1"]):.3f}x' for b in [4,32]))
            throughput_rows.append(row)
    throughput_table=table(['规模/布局，T32',*[f'{LABEL[m]}：B4/B1、B32/B1 吞吐' for m in METHODS]],throughput_rows)
    agg_through=[]
    for m in METHODS:
        ratios=[ns(get(sc,la,32,32,m))/ns(get(sc,la,1,32,m)) for sc in SCALES for la in LAYOUTS]
        agg_through.append([LABEL[m],*[f'{st.median(float(r["throughput_vs_b1"]) for r in scoperows(m,b,32)):.3f}x' for b in [4,32]],f'{st.median(ratios):.3f}x'])
    gf=table(['规模',*[LABEL[m] for m in METHODS]],
        [[sc,*[' / '.join(f'{st.mean(float(get(sc,la,b,32,m)["total_gflops"]) for la in LAYOUTS):.1f}' for b in BATCHES) for m in METHODS]] for sc in SCALES])
    scheduling=table(['规模','X0 inter/intra','X4 inter/intra','X0 inter/kml-locking','X4 inter/kml-locking','X0 inter/kml-multi','X4 inter/kml-multi'],
        [[sc,*[f'{ns(get(sc,la,32,32,INTER))/ns(get(sc,la,32,32,m)):.3f}x' for m in [INTRA,LOCK,MULTI] for la in LAYOUTS]] for sc in SCALES])
    layout=table(['规模','两种 KML 的 X4 输入物化',*[LABEL[m] for m in METHODS]],
        [[sc,f'{size:.1f} MiB',*[f'{ns(get(sc,"X4",32,32,m))/ns(get(sc,"X0",32,32,m)):.3f}x' for m in METHODS]] for sc,size in zip(SCALES,[4,16,64,256])])
    b1layout=table(['规模，B1/T32','kml-locking X4/X0','kml-multi X4/X0','locking 差值/ms','multi 差值/ms'],
        [[sc,*[f'{ns(get(sc,"X4",1,32,m))/ns(get(sc,"X0",1,32,m)):.3f}x' for m in [LOCK,MULTI]],
          *[f'{(ns(get(sc,"X4",1,32,m))-ns(get(sc,"X0",1,32,m)))/1e6:.6f}' for m in [LOCK,MULTI]]] for sc in SCALES])
    distribution=[];campaign=[]
    for b in BATCHES:
        for m in METHODS:
            rows=scoperows(m,b);ratio=[int(r['p95_total_ns'])/ns(r) for r in rows]
            samples=[r for r in raw if r['method']==m and int(r['batch'])==b]
            outliers=sum(int(r['total_ns'])>2*ns(by[r['case_id'],m]) for r in samples)
            distribution.append([f'B{b}',LABEL[m],f'{st.median(ratio):.3f}',f'{max(ratio):.3f}',f'{outliers}/{len(samples)} ({100*outliers/len(samples):.2f}%)'])
            worst=max(rows,key=lambda r:float(r['campaign_spread_pct']))
            campaign.append([f'B{b}',LABEL[m],f'{st.median(float(r["campaign_spread_pct"]) for r in rows):.2f}%',f'{float(worst["campaign_spread_pct"]):.2f}%',worst['case_id']])
    tail_evidence=[]
    for cid in ['S_X0_B1_T32','M_X0_B1_T32']:
        q=[]
        for c in range(3):
            v=sorted(int(r['total_ns']) for r in groups[cid,MULTI] if int(r['campaign'])==c)
            q.append(f'{v[math.ceil(.95*len(v))-1]/1e6:.6f}')
        tail_evidence.append([cid,*q])
    tblis_tail=section(template,'端到端 CDF 与尾延迟')
    tblis_tail=tblis_tail[tblis_tail.index('### TBLIS-intra 的 T2 长尾'):]
    tblis_tail=tblis_tail.replace('### TBLIS-intra 的','### 历史 TBLIS-intra 的')
    tblis_tail=tblis_tail.replace('上表中 97%-99% 的未归因占比说明 T32 小任务退化不是 kernel 变慢。',
        '上表的低覆盖比例提示还存在未被这些指标描述的端到端开销；由于 packing/kernel 是各线程累计值的最大值，不能将差额视为精确的等待时间，也不足以排除所有计算路径影响。')
    limits=rename(section(template,'口径限制')).replace('kml-locking 内部 packing','两种 KML 内部 packing')
    limits+='\n9. kml-multi 来自后续独立运行，其他方法沿用历史；虽然输入、计时、affinity、参考一致，小差距仍受日期及宿主状态影响。\n10. kml-multi 的 B1 采用 inner，B4/B32 采用 outer；没有验证 shape-tuned/hybrid 最优策略。相对 B1 吞吐的基准与 kml-locking 不同。\n11. 原性能序列的周期导致部分 batch 值相同，不能充分发现索引串位；新增独立随机校验弥补这一覆盖，但不替换历史性能输入。'
    recommendations='''1. **单 batch 的方法选择应纳入 kml-multi。** 已测八个规模/布局在最佳线程中位数上均由它领先；这改变了旧报告中相对 kml-locking 的定位。
2. **同时约束 P95。** S/X0/B1/T32 和 M/X0/B1/T32 中位数显著变快，但绝对 P95 退化；固定满线程不能作为普适最优策略。
3. **多 batch 保留两种 KML 对照。** B4/B32 中位数多数接近，B32/T32 的七个规模/布局由 kml-multi 领先，M/X4 仍由 TBLIS-inter 小幅领先；小差距不作显著性承诺。
4. **XL/B4 仍有线程策略空间。** 本次 kml-multi 保守使用 outer-only，不能用它的结果否定 B4 的 inner/hybrid 候选。
5. **布局优化和阶段归因仍然必要。** X4 整输入物化没有因换库消失；kernel 指标不能代替端到端，CPU 活跃线程也不能替代 BLAS/OMPT 计算路径证据。
6. **4-vCPU 一致性结论限于历史子集。** 不再用“整体方向完全一致”概括加入 multi 后的五方法排名；在线服务还需另外计入排队和 batching 等待。'''
    parts=[('# 大规模多批次多线程张量缩并实验结果分析（kml-locking / kml-multi 合并版）',''),
      ('## 分析范围',source_note+'\n\n固定表达式：\n\n```text\nC[p,a,b,c,d] = sum(k,l) A[p,a,b,k,l] * B[p,k,l,c,d]\n```\n\n'+protocol),
      ('## 方法与调度',methods_text+'\n\n实际线程诊断已验证 kml-multi 的 B1 内部 team，以及 B4/B32 的单线程 BLAS 调用。旧 kml-locking 没有同口径 OMPT，不虚构历史观测值。'),
      ('## TBLIS 的 intra 和 inter 设置',section(template,'TBLIS 的 intra 和 inter 设置')+'\n\n上述两种调度来自同一 TBLIS 构建，保持历史数据；方法定位需结合新增 kml-multi 一起比较。'),
      ('## 结论摘要',conclusions),
      ('## 与 4 vCPU 实验的一致性（历史子集）',controls('旧四方法的共同 T1/T2/T4 用例','旧 4-vCPU 与旧 32-vCPU 资源环境')+
       '结论：原来的 69/72 方法排序一致仍保留，但仅针对 PyTorch-OpenBLAS、kml-locking、TBLIS-intra/inter；kml-multi 没有 4-vCPU 配对数据。\n\n'+legacy4),
      ('## 总体胜出分布',controls('每个用例的规模、布局、P、请求 T 和逻辑输入','五种方法，且来源跨测量日期')+
       '结论：固定线程配置下，kml-multi 胜 87/120，kml-locking 胜 21/120，OpenBLAS 胜 7/120，TBLIS-intra 胜 4/120，inter 胜 1/120。胜出仅按中位数排序，不能替代 P95 或显著性判断。\n\n'+win_table+
       '\n\n### 各方法选择最佳线程数\n\n不变量：每个工作负载的规模、布局、P。变量：方法与各自选择的最佳已测 T。\n\n结论：24 个工作负载中，kml-multi 胜 20 个、kml-locking 胜 1 个、intra 胜 2 个、inter 胜 1 个。\n\n'+best_table+
       '\n\n### B1：逐用例最佳线程对照\n\n结论：八个 B1 的中位数排名均由 kml-multi 领先；表中不同方法可能使用不同线程，且未把 P95 作为优化目标。\n\n'+best_b1),
      ('## Batch 与线程扩展','### T1 到 T32 的端到端加速\n\n'+controls('每行规模、布局、P及方法','T1→T32，CPU集合随请求线程变化')+
       '结论：kml-locking/inter 的 B1 基本不扩展；kml-multi 和 TBLIS-intra 可以并行单项。intra 在小任务 T32 可能退化，单个汇总倍率无法表达这种差异。下面逐用例展示，所有时间单位为 ms。\n\n'+scaling_table+
       '\n\n附表：沿用原附件的八个规模/布局比值中位数，补入 kml-multi；仅作汇总参考，主要判断以上表为准。\n\n'+aggregate_scaling+
       '\n\n### T32 下随 batch 增长的吞吐\n\n'+controls('单项规模、布局、方法及 T32','P=1/4/32，总工作量随 P 增加')+
       '结论：kml-multi 的 B1 被内部并行显著加速，因此 B32/B1 相对吞吐比会变小；它还在 B1 与 B4/B32 间切换调度策略。这不等于 B32 绝对性能退化。\n\n相对吞吐=P×B1时间/BP时间；不是强扩展加速，也不是每项响应延迟。\n\n'+throughput_table+
       '\n\n附表：原附件汇总口径，补齐第五方法。\n\n'+table(['方法','B4/B1 吞吐','B32/B1 吞吐','B32/B1 总耗时'],agg_through)+
       '\n\n### T32 端到端性能\n\n固定 T32；每格依次为 B1/B4/B32 的 GFLOP/s，按原附件取 X0/X4 算术平均。\n\n结论：新增 multi 的主要跃升出现在 B1；B4/B32 使用相同外层策略，其数值总体接近 locking。\n\n'+gf),
      ('## 两种 TBLIS 调度的边界',controls('P=32、T32，每行规模/布局','intra/inter 及与两种 KML 的方法对照')+
       '结论：历史 inter 在全部八组中快于 intra；相对两种 KML，M/X4 的 inter 小幅领先，其余七组由 KML 路径领先。\n\n比值分子均为 inter，分母由列名指定。\n\n'+scheduling),
      ('## 布局、物化与 batch 工作集',controls('每种规模内的 P=32、T32、逻辑值与计算量','X0→X4，方法分别配对')+
       '结论：两种 KML 的 X4 整输入物化量相同；multi 并未消除上游 reshape 复制。以下是 X4/X0 的中位耗时比。\n\n'+layout+
       '\n\n### B1 加速后，布局差值的相对影响\n\n不变量：每种规模内 P=1、T32、逻辑计算和值。变量：X0/X4 与 locking/multi。\n\n结论：计算变快后，即使绝对布局差值近似不变，相对惩罚也会增加，不能据此认定复制字节增加。\n\n'+b1layout),
      ('## 端到端 CDF 与尾延迟',controls('各具体用例及每方法 150 个正式样本','方法和观察的分位点')+
       f'结论：中位数排名不能代替高分位排名。合并后 {len(tail)}/600 组 P95/中位数≥2；新增 multi 中存在显著尾延迟回退。\n\n'
       f'总览覆盖 120 配置、5 方法、90,000 样本；每行规模/布局，五列 T1/T2/T4/T8/T32，线型区分 batch，横轴为端到端 µs 对数刻度。\n\n![五方法CDF总览]({assets/"cdf_batch_all_overview.svg"})\n\n'
       f'[逐用例 CDF]({assets/"cdf/index.MD"})，每条曲线直接来自 150 个样本。\n\n### 分布稳定性\n\n'+
       table(['Batch','方法','P95/中位数中位值（汇总）','最大值','>2x中位数样本'],distribution)+
       '\n\n### kml-multi 相对 kml-locking 的 P95 回退\n\n结论：8 个配置 P95 回退超过 5%，其中 S/X0/B1/T32 的高 P95 在三轮均出现；M/X0 的轮间变化较大，根因未完全定位。\n\n'+tail_table+
       '\n\n每轮 P95（ms，每轮 50 个正式样本）：\n\n'+table(['用例','campaign 0','campaign 1','campaign 2'],tail_evidence)+'\n\n'+tblis_tail),
      ('## Campaign 复现性',controls('每个用例的方法、布局、规模、P、T与输入','三次独立 campaign')+
       '结论：需区分单轮内部长尾和跨轮中心位置变化。波动定义为三轮中位数极差除以三轮中位数的中位值；以下保留原汇总统计并加入 multi。\n\n'+
       table(['Batch','方法','波动中位值','最大值','最大值用例'],campaign)),
      ('## 正确性与证据完整性','''- 120 个唯一配置、600 个方法组，各组 150 个正式样本；历史 72,000 个与新增 18,000 个明确区分。
- 原 480 行性能数据数值保留，新增 120 行 kml-multi；中位数、P95 均从原始样本复算。
- 新旧 KML 的输入 SHA256、入口 shape/stride/offset、参考及 affinity 一致；OpenBLAS 运行前后配置与库哈希不变。
- 新增 120 个 BLAS/OMPT 诊断验证 inner/outer 分工，无失控嵌套；另有 batch 不同的小矩阵正确性测试。
- 最大归一化误差 2.205e-05 < 1e-4；每轮计时后校验末次输出，并非每个计时区间内执行参考校验。
- kml-locking 的历史 CPU 活跃 worker 不能当作 GEMM 内部 team；新增 observer 明确区分这些概念。
- CDF 图例、正文和 TSV 的显示名统一为 kml-locking/kml-multi，原始方法 ID 另存 source_method。'''),
      ('## 口径限制',limits),('## 最终判断与建议',recommendations),('## 数据索引',index_text)]
    return '\n\n'.join(title+('\n\n'+body if body else '') for title,body in parts)+'\n'


if __name__=='__main__':main()
