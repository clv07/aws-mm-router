# Archive

Early test runs from before the baseline and routing benchmarks were set up. They are kept for reference and are not used in the report or the analysis.

| File                                  | What it is |
| ------------------------------------- | ---------- |
| `results/benchmark.csv`               | A first test of `benchmark.py` (Oct 29): 9 workloads, one request each, on a c8g.4xlarge, in an older results format. |
| `roundrobin.py`                       | An early closed-loop round-robin baseline over two nodes over SSH tunnels: at most `--inflight` requests outstanding, request *i* goes to node *i* mod 2. Replaced by `loadgen.py` (`--policy rr`, open loop). |
| `results/roundrobin.csv`, `results/roundrobin_requests.csv` | Its results (Oct 29): 50 requests over a c7a.2xlarge and a c7i.2xlarge with 4 in flight, per group and per request. |

`roundrobin.py` imports `benchmark.py` and the dataset scripts from `src/`, so run it from the repository root with `PYTHONPATH=src python archive/roundrobin.py`; it writes to `results/roundrobin*.csv`.
