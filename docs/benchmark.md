# Qwen2.5-Omni Phase 1 benchmark

One Python program runs on your **benchmark client**, not on your inference instances. It sends HTTP requests to both EC2 inference nodes and samples `llama-server` CPU/RSS via SSH.

## Setup

Keep `benchmark.py`, `coco.py`, `cnn_dailymail.py`, and `librispeech.py` in the same folder. Install:

```bash
pip install requests datasets transformers soundfile
```

Download datasets once using the existing scripts:

```bash
python cnn_dailymail.py --download
python coco.py --download
python librispeech.py --download
```

Edit `NODES` at the top of `benchmark.py` with the two inference HTTP addresses and SSH targets. Your client must be able to reach the EC2 hosts over HTTP and SSH (e.g. same private VPC, or SSH tunneling). Configure SSH keys using your usual `~/.ssh/config` or ssh-agent. Only allow access from the benchmark client; don't expose port 8080 publicly.

Run:

```bash
python benchmark.py --run 01  # quick single workload, both nodes
python benchmark.py           # all 9 workloads on both nodes
```

The script fixes repetitions at 10, one warm-up per workload, output cap at 128, single request at a time, and deterministic sampling parameters. No long CLI configuration is needed.

Results: `results/summary.csv` and `results/{small,large}_runNN_raw.csv`.

## Columns

- `median_e2e_s`, `p95_s`: client-observed completion time.
- `req_s`: successful requests divided by **sum of request durations** (sequential inference throughput; excludes SSH setup/dataset loading).
- `ttft_s`: median request-to-first-streamed-content delay.
- `decode_tok_s`: median server-reported decode rate when available; otherwise a client-side estimate if completion token counts are reported. Blank when unavailable.
- `peak_ram_gb`: maximum sampled process RSS (GiB), not instance-wide memory.
- `cpu_pct`: median per-request CPU utilization of `llama-server` normalized to the node's total logical CPUs (0–100%).
- `success`, `failed`: number of completed and failed requests.

## Important

Run the same server model, quantization, context settings appropriate for each input, and compatible multimodal support on each node. The 8192-token input requires an appropriately large model context plus output budget. Input lengths from `cnn_dailymail.py` refer to content tokens, not full templated tokens.

Streaming server events may omit token usage/timing data; in that case `decode_tok_s` can be blank. Audio schema support varies by llama.cpp version; verify a one-request audio smoke test before the full benchmark. SSH sampling at 0.2s intervals measures a sampled peak and may miss short-lived spikes. If SSH authentication fails, latency still runs but CPU and memory results are blank. With only 10 repetitions, p95 is exploratory rather than a high-confidence tail estimate.