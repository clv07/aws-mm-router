"""Validate Phase 1 CSV metrics against summaries and the unchanged routing analyzer."""
import importlib.util
from pathlib import Path
import sys
import math
import analyze as a


def close(actual, expected, digits=None):
    """Compare exact quantiles or runner-rounded throughput and cost."""
    if digits is not None:
        assert round(actual,digits)==round(float(expected),digits),(actual,expected)
    else:
        assert math.isclose(actual,float(expected),abs_tol=1e-8),(actual,expected)


def main():
    """Check all seven runs, latest-run placement/prediction, and historical failures."""
    batch=Path('results/routing'); rows=a.read(batch/'requests.csv')
    runs=a.groups(rows,('run_id',))
    assert len(runs)==7
    price=sum(c['price_per_hour'] for c in a.lg.load_nodes(batch/'nodes.json').values())
    for s in a.read(batch/'summary.csv'):
        rs=runs[(s['run_id'],)]
        span=max(float(r['sent_s'])+float(r['e2e_s']) for r in rs)
        members=[r for r in rs if s['group']=='all' or r['modality']==s['group'] or r['node']==s['group']]
        ok=[r for r in members if r['status']=='ok']
        assert len(ok)==int(s['success']) and len(members)-len(ok)==int(s['failed'])
        for field,prefix in [('e2e_s','e2e'),('ttft_s','ttft')]:
            xs=[float(r[field]) for r in ok if r[field]]
            for name,value in [('median',a.median(xs)),('p95',a.lg.percentile(xs,.95)),('p99',a.lg.percentile(xs,.99))]:
                col=f'{name}_{prefix}_s'
                if s.get(col): close(value,s[col])
        close(len(ok)/span,s['req_s'],3)
        if s['usd_per_1k_requests']: close(price*span/3600/len(ok)*1000,s['usd_per_1k_requests'],3)
    # Import definitions without running the old analyzer's plotting main or changing its file.
    spec=importlib.util.spec_from_file_location('routing_baseline','analysis/routing/analyze_routing.py')
    baseline=importlib.util.module_from_spec(spec); argv=sys.argv; sys.argv=[argv[0]]
    spec.loader.exec_module(baseline); sys.argv=argv
    d=baseline.load()
    selected=[r for r in rows if r['status']=='ok' and r['run_id']==max(q['run_id'] for q in rows if (q['trace'],q['policy'])==(r['trace'],r['policy']))]
    for _,r in baseline.latency_table(d).iterrows():
        xs=[float(q['e2e_s']) for q in selected if (q['trace'],q['policy'])==(r.trace,r.policy) and (r.modality=='all' or q['modality']==r.modality)]
        close(a.median(xs),r['median']); close(a.lg.percentile(xs,.95),r.p95)
    for _,r in baseline.placement_table(d).iterrows():
        qs=[q for q in selected if (q['trace'],q['policy'],q['modality'])==(r.trace,r.policy,r.modality)]
        close(sum(q['node']==r.node for q in qs)/len(qs),r.share)
    for _,r in baseline.prediction_table(d).iterrows():
        qs=[q for q in selected if (q['trace'],q['policy'])==(r.trace,r.policy)]
        err=[float(q['e2e_s'])-float(q['predicted_e2e_s']) for q in qs]
        close(a.median(err),r.median_err_s)
        close(a.median([abs(e)/float(q['e2e_s']) for e,q in zip(err,qs)]),r.median_abs_rel)
    failed=[r for r in runs[('20261105T012551Z',)] if r['status']=='error']
    assert len(failed)==42 and len(runs[('20261105T012551Z',)])==122
    normalized=a.normalize(batch,[Path('traces')],a.lg.load_costs(a.lg.load_nodes(batch/'nodes.json'))[0])
    for key,qs in a.groups(failed,('node',)).items():
        expected=min(float(q['sent_s'])+float(q['e2e_s']) for q in qs)
        close(min(r['finish'] for r in normalized if r['run_id']=='20261105T012551Z' and r['node']==key[0] and r['status']=='error'),expected)
    print('PASS: seven summaries, latest placement/latency/prediction, 42 failures and first failures')


if __name__=='__main__': main()
