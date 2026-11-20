"""Validate Phase 1 CSV metrics against summaries and the unchanged routing analyzer."""
import importlib.util
from pathlib import Path
import sys
import math
import tempfile
from types import SimpleNamespace
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
    price=sum(c['price_per_hour'] for c in a.router.load_nodes(batch/'nodes.json').values())
    for s in a.read(batch/'summary.csv'):
        rs=runs[(s['run_id'],)]
        span=max(float(r['sent_s'])+float(r['e2e_s']) for r in rs)
        members=[r for r in rs if s['group']=='all' or r['modality']==s['group'] or r['node']==s['group']]
        ok=[r for r in members if r['status']=='ok']
        assert len(ok)==int(s['success']) and len(members)-len(ok)==int(s['failed'])
        for field,prefix in [('e2e_s','e2e'),('ttft_s','ttft')]:
            xs=[float(r[field]) for r in ok if r[field]]
            for name,value in [('median',a.median(xs)),('p95',a.runner.percentile(xs,.95)),('p99',a.runner.percentile(xs,.99))]:
                col=f'{name}_{prefix}_s'
                if s.get(col): close(value,s[col])
        close(len(ok)/span,s['req_s'],3)
        if s['usd_per_1k_requests']: close(price*span/3600/len(ok)*1000,s['usd_per_1k_requests'],3)
    # Import definitions without running the old analyzer's plotting main or changing its file.
    spec=importlib.util.spec_from_file_location('routing_baseline','analysis/routing/analyze_routing.py')
    baseline=importlib.util.module_from_spec(spec); argv=sys.argv; sys.argv=[argv[0]]
    spec.loader.exec_module(baseline); sys.argv=argv
    d=baseline.load()
    # Exercise the analysis entry point and compare the files a user actually reads.
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        a.analyze(SimpleNamespace(batches=[batch], traces=[Path('traces')], nodes=None,
                                  trim_start=60, trim_end=60, out=out))
        def latest(table, keys, **filters):
            """Index emitted Phase 1 comparisons at the baseline's grouping level."""
            return a.unique([r for r in a.read(out/(table+'.csv')) if r['source']=='aws' and r['latest']=='True'
                             and all(r[k]==v for k,v in filters.items())], keys)

        by_policy = ('trace','policy')
        heads = latest('headline', by_policy)
        assert {r['run_id'] for r in heads.values()} == set(d.run_id)
        latency = latest('per_modality', (*by_policy,'modality'))
        for _,r in baseline.latency_table(d).iterrows():
            if r.modality != 'all':
                row = latency[(r.trace,r.policy,r.modality)]
                close(float(row['full_send_p50']),r['median'])
                close(float(row['full_send_p95']),r.p95)
        placement = latest('placement', (*by_policy,'modality','node'), size='all')
        for _,r in baseline.placement_table(d).iterrows():
            row = placement[(r.trace,r.policy,r.modality,r.node)]
            close(float(row['share']),r.share)
            assert int(row['full_success']) == r.n
        prediction = latest('prediction', by_policy, node='all', modality='all')
        for _,r in baseline.prediction_table(d).iterrows():
            row = prediction[(r.trace,r.policy)]
            close(float(row['signed_error_s_p50']),r.median_err_s)
            close(float(row['median_abs_rel']),r.median_abs_rel)
        node_rows = a.unique([r for r in a.read(out/'nodes.csv') if r['source']=='aws'], ('run_id','node'))
    failed=[r for r in runs[('20261105T012551Z',)] if r['status']=='error']
    assert len(failed)==42 and len(runs[('20261105T012551Z',)])==122
    for (node,), qs in a.groups(failed,('node',)).items():
        expected = min(float(q['sent_s'])+float(q['e2e_s']) for q in qs)
        close(float(node_rows[('20261105T012551Z',node)]['first_failure_s']),expected)
    print('PASS: seven summaries, latest placement/latency/prediction, 42 failures and first failures')


if __name__=='__main__': main()
