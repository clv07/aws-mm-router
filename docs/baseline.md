# Dataset setup
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

# llama-server setup

`run.sh` runs on each inference instance. It builds llama.cpp (commit `8642541b2`, CPU only) and downloads the model if missing, starts `llama-server` in a tmux session, and returns once the model is loaded. If the server fails to start, it prints the last 20 log lines.

## Setup

```bash
./run.sh <model_name>
tmux attach -t llama         # live server; leave with Ctrl-b d (Ctrl-C stops the server)
tail -f ~/llama-server.log   # log stream
```
`model_name` option:
1. `qwen`: Qwen2.5-Omni-3B Q8_0 (5.2 GB)
2. `gemma`: Gemma 4 E2B Q8_0 (5.5 GB)

Run it on every node with the same model. Switching models is the same command with the other name. The server restarts only if the model or flags changed, which aborts requests in flight. Rerun it after an instance stop/start.

Defaults (top of `run.sh`): `-c 8192 -np 2`, i.e. 2 requests at a time with 4,096 tokens each (input + output); `-t` = number of vCPUs; listens on `127.0.0.1:8080` (localhost only, so connect through an SSH tunnel).

## Important

The first run needs `sudo` and a few minutes to build and download. The 4,096-token limit is a server setting, not the model's; for longer inputs, edit `CTX` at the top of `run.sh` and rerun it (more context uses more memory).

# Single-request benchmark

One Python program runs on the client. It sends HTTP requests to both EC2 inference nodes and samples `llama-server` CPU/RSS via SSH.

## Setup

Edit `NODES` at the top of `benchmark.py` with the two inference HTTP addresses and SSH targets. Client must be able to reach the EC2 hosts over HTTP and SSH (e.g. same private VPC, or SSH tunneling). Configure SSH keys using usual `~/.ssh/config` or ssh-agent. Only allow access from the benchmark client; don't expose port 8080 publicly.

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

# Round Robin Baseline

`roundrobin.py` runs on the client. Round robin fixes each request's node in advance by its position in the list (A, B, A, B, ...), regardless of the request type or how busy the node is. The script sends 50 text/image/audio requests in order, 4 at a time across all nodes: each time one request finishes, the next request is sent. A request sent to a busy node waits in that node's queue. It also samples `llama-server` CPU/RSS via SSH.

## Setup

Start the same model on every node with `run.sh` and download the datasets (see above). Edit `NODES` at the top of `roundrobin.py`: entry order is the turn order, and each node needs its own local port and SSH alias. One entry gives the fixed-node baseline.

Open one tunnel per node, then run from the repo root:

```bash
ssh -N -L 8080:localhost:8080 <first node>
ssh -N -L 8081:localhost:8080 <second node>

python roundrobin.py --requests 10 --inflight 2   # quick test
python roundrobin.py                              # 50 requests, 4 in flight, seed 0
```

The mix is 3:1:1 text/image/audio with sizes 512/1024/3072 tokens, 1/2/4 images and 5/10/20 s. Each request uses temperature 0, at most 128 output tokens, no prompt cache, thinking mode off, one unrecorded warm-up runs per node and modality.

Results are appended every run and linked by `timestamp_utc`: `results/roundrobin_requests.csv` (one row per request) and `results/roundrobin.csv` (one summary row per group: `all`, each modality, each node).

## Columns

Both files record the run: `timestamp_utc`, `policy`, `seed`, `inflight`.

Per request:

- `index`: send order.
- `node`, `model`: where the request ran and the model reported by `llama-server`.
- `modality`, `size`: tokens / images / seconds.
- `sent_s`: send time since the start of the run.
- `e2e_s`: send to end of stream, including queueing on the node.
- `ttft_s`: send to first answer token.
- `decode_tok_s`, `output_tokens`: server-reported decode rate and tokens generated (≤ 128).
- `status`, `error`: `ok` or the error message.

Summary:

- `median_e2e_s`, `p95_s`: over the group's requests.
- `ttft_s`: median over the group's requests.
- `req_s`: the group's successful requests divided by the **whole run's wall time**; on the `all` row, run time = `success / req_s`.
- `peak_ram_gb`, `cpu_pct`: node rows only; maximum sampled RSS (GiB) and median CPU (0–100% of vCPUs) of `llama-server` during the run.
- `success`, `failed`: number of completed and failed requests.

## Important

Every run sends the same 50 inputs; `--seed` only changes their order, and with it the node assignment. Thinking is off because Gemma 4 would otherwise spend all 128 tokens reasoning and return no answer; Qwen does not think, so it is unaffected. When comparing models, note that replies can stop before 128 tokens and text sizes are counted with the Qwen tokenizer, so look at `output_tokens` and `decode_tok_s` alongside `e2e_s`.

Results are written only when a run finishes (Ctrl-C records nothing). If SSH sampling fails, CPU and memory results are blank.
