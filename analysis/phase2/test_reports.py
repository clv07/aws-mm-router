"""Regression checks for idle-node reporting and steady-state breakdown CSV output."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import analyze as a


class ReportTests(unittest.TestCase):
    """Check report values through the analysis entry point."""

    def test_idle_nodes_and_breakdown_window(self):
        """Show idle capacity and exclude pre-window arrivals from breakdown means."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = [dict(id=str(i), arrival_s=t, run='00', modality='chat')
                     for i, t in [(1, 0), (2, 100)]]
            a.write(root/'trace.csv', trace)
            (root/'trace.json').write_text(json.dumps({'params': {'duration': 200, 'seed': 1}}))
            requests = [dict(source='sim', run_id='test', trace='trace', policy='ect',
                             id=str(i), run='00', modality='chat', arrival_s=t,
                             start_s=t, finish_s=t+duration, node='active', node_slots=2)
                        for i, t, duration in [(1, 0, 1000), (2, 100, 1)]]
            a.write(root/'sim_requests.csv', requests)
            args = SimpleNamespace(batches=[root], traces=[root], nodes='unused',
                                   trim_start=60, trim_end=60, out=root/'out')
            nodes = {'active': {'slots': 2}, 'idle': {'slots': 2}}
            with patch.object(a.router, 'load_nodes', return_value=nodes), \
                 patch.object(a.router, 'load_costs', return_value=({'active': {'00': 1}}, [])):
                a.analyze(args)
                reported = a.unique(a.read(args.out/'nodes.csv'), ('node',))
                idle = reported[('idle',)]
                for field in ('throughput', 'busy_fraction', 'slot_utilization'):
                    self.assertEqual(float(idle[field]), 0)
                breakdown = a.read(args.out/'breakdown.csv')[0]
                self.assertEqual(float(breakdown['service_s']), 1)
                self.assertEqual(int(breakdown['breakdown_requests']), 1)
                args.trim_end = 150
                a.analyze(args)
                breakdown = a.read(args.out/'breakdown.csv')[0]
                self.assertEqual(breakdown['service_s'], '')
                self.assertEqual(breakdown['breakdown_requests'], '')


if __name__ == '__main__':
    unittest.main()
