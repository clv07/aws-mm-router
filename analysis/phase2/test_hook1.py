"""Test JSON byte parity and fresh-sample buffering locally, without AWS servers."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import analyze
import requests
from phase2_measurement import request_bytes, Measurements


class HookTests(unittest.TestCase):
    """Exercise local parts of the AWS measurement hook."""
    def test_bytes(self):
        """Match PreparedRequest bytes including Unicode escaping and whitespace."""
        for body in ({'text':'héllo','array':[1,2]}, {'image':'a'*10000}):
            self.assertEqual(request_bytes(body),len(requests.Request('POST','http://local',json=body).prepare().body))
        with self.assertRaises(ValueError): request_bytes({'bad':float('nan')})

    def test_fresh(self):
        """Skip old samples and record a new sample once despite repeated polls."""
        with tempfile.TemporaryDirectory() as directory:
            monitor=SimpleNamespace(samples=[{'cpu_pct':1,'ram_gb':2}])
            m=Measurements(Path(directory),'test',{'node':monitor},0)
            m.observe(); self.assertEqual(m.samples,[])
            monitor.samples.append({'cpu_pct':3,'ram_gb':4})
            m.observe(); m.observe(); self.assertEqual(len(m.samples),1)
            self.assertEqual(m.samples[0]['cpu_pct'],3)
            m.thread.start(); m.record('1',10,2,3); m.close()
            self.assertEqual(len(analyze.read(Path(directory)/'requests_extra.csv')),1)
            self.assertEqual(len(analyze.read(Path(directory)/'node_samples.csv')),1)


if __name__=='__main__': unittest.main()
