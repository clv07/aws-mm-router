"""Hand-made request tests for wait attribution, clipped utilization and window boundaries."""
from pathlib import Path
import tempfile
import unittest
import analyze as a


class RowTests(unittest.TestCase):
    """Check independent expected values for temporal metrics."""
    def test_overlap(self):
        """Two overlapping slots occupy 8 of 10 seconds and 12 slot-seconds."""
        rows=[dict(busy=-2,finish=6),dict(busy=2,finish=8)]
        self.assertEqual(a.intervals(rows,0,10,2),(.8,.6))
        self.assertEqual(a.intervals(rows,0,10,None),(.8,None))
        self.assertEqual(a.intervals(rows,10,10,2),(None,None))
        self.assertEqual(a.intervals([dict(busy=None,finish=5)],0,10,2),(None,None))

    def test_boundary(self):
        """Arrival cohorts exclude the right endpoint; completions include earlier arrivals."""
        rows=[]
        for arrival,finish,status in [(50,70,'ok'),(60,80,'ok'),(119,130,'error'),(120,125,'ok')]:
            rows.append(dict(arrival=arrival,finish=finish,status=status,latency=finish-arrival,
                             slowdown=(finish-arrival)/10,deadline=15,met=status=='ok' and finish-arrival<=15))
        m=a.metrics(rows,60,120,1)
        self.assertEqual(m['requests'],2); self.assertEqual(m['success'],1)
        self.assertEqual(m['throughput'],2/60); self.assertEqual(m['attainment'],0)
        self.assertEqual(m['latency_p95'],20); self.assertEqual(m['goodput'],0)
        self.assertEqual(a.metrics(rows,60,60,1),{})

    def test_wait(self):
        """The same service gap is router wait for queue policies and node wait otherwise."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a.write(root/'trace.csv',[dict(id='1',deadline_s=15,slo_factor=3)])
            rows=[dict(source='sim',run_id='test',trace='trace',policy=p,id='1',run='00',modality='chat',arrival_s=2,start_s=12,finish_s=22,node='n',node_slots=2) for p in (a.router.POLICIES[0],a.router.QUEUE_POLICIES[0])]
            a.write(root/'sim_requests.csv',rows)
            normalized=a.normalize(root,[root],{'n':{'00':5}})
            for r in normalized:
                self.assertEqual(r['latency'],20); self.assertEqual(r['slowdown'],4)
                self.assertFalse(r['met'])
                if r['policy'] in a.router.QUEUE_POLICIES:
                    self.assertEqual(r['router_wait_s'],10); self.assertEqual(r['node_wait_s'],0)
                else:
                    self.assertEqual(r['router_wait_s'],0); self.assertEqual(r['node_wait_s'],10)


if __name__=='__main__': unittest.main()
