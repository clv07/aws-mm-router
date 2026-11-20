"""Generate a Phase 2 trace, replay 50 requests in isolation and check exported metrics."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import analyze as a


def main():
    """Check both deadline rules and mean per-call duration for every simulated policy."""
    with tempfile.TemporaryDirectory(prefix='phase2-sim-') as directory:
        root=Path(directory); trace=root/'check.csv'
        import tracegen
        old=sys.argv[:]; old_out=a.lg.OUT_DIR
        try:
            sys.argv=['tracegen','make','--nodes','results/routing/nodes.json','--load','0.9','--seed','1','--out',str(trace)]
            tracegen.main()
            a.lg.OUT_DIR=root
            sys.argv=['loadgen','sim','--trace',str(trace),'--nodes','results/routing/nodes.json','--requests','50']
            a.lg.main()
            rs=a.read(root/'sim_requests.csv'); ds=a.read(root/'sim_decisions.csv')
            qs=a.unique(a.read(trace),('id',))
            for s in a.read(root/'sim_summary.csv'):
                members=[r for r in rs if r['policy']==s['policy'] and (s['group']=='all' or s['group']==r['modality'])]
                assert len(members)==int(s['requests'])
                met=sum(float(r['finish_s'])<=float(r['sim_deadline_s']) for r in members)/len(members)
                assert abs(met-float(s['slo_attainment']))<1e-12
                assert all(r['predicted_finish_s']=='' for r in members)
                if s['group']=='all':
                    durations=[float(d['duration_us']) for d in ds if d['policy']==s['policy']]
                    assert abs(sum(durations)/len(durations)-float(s['decision_us']))<1e-9
            normalized=a.normalize(root,[root],a.lg.load_costs(a.lg.load_nodes('results/routing/nodes.json'))[0])
            for r in normalized:
                assert r['met']==(r['finish']-r['arrival']<=float(qs[(r['id'],)]['deadline_s']))
                if r['policy'] in a.lg.QUEUE_POLICIES: assert r['node_wait_s']==0
                else: assert r['router_wait_s']==0
            print('PASS: exported overall/per-modality sim attainment, trace attainment and decision_us')
        finally:
            sys.argv=old; a.lg.OUT_DIR=old_out


if __name__=='__main__': main()
