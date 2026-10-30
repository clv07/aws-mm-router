# Phase 1 Benchmark

Two Python scripts run on the benchmark client, sending HTTP requests to AWS EC2 inference nodes and collecting CPU/RAM metrics via SSH: `benchmark.py` sends one request at a time, and `roundrobin.py` runs the round-robin baseline under load. `run.sh` sets up the inference server on each node.

## 1. Setup

All scripts are in the project root:

```
benchmark.py
roundrobin.py
run.sh
cnn_dailymail.py
coco.py
librispeech.py
dataset/
results/
```

Create a virtual environment and install dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip install requests datasets transformers soundfile
```

Download datasets:

```
python cnn_dailymail.py --download
python coco.py --download
python librispeech.py --download
```

## 2. Inference Server

On each EC2 node, start `llama-server` with the same model:

```
./run.sh <model_name>
```
`model_name` option:
1. `qwen`: Qwen2.5-Omni-3B Q8_0 (5.2 GB)
2. `gemma`: Gemma 4 E2B Q8_0 (5.5 GB)

`run.sh` builds llama.cpp (commit `8642541b2`, CPU only) and downloads the model if missing, starts `llama-server` in tmux session `llama`, and returns once the model is loaded. Rerun it to switch models (the restart aborts requests in flight) or after an instance stop/start. The first run needs `sudo` and a few minutes.

Monitor the server:

```
tmux attach -t llama          # leave with Ctrl-b d; Ctrl-C stops the server
tail -f ~/llama-server.log
```

Server settings: `-c 8192 -np 2` (2 requests at a time, 4096 tokens each for input + output), `-t` = number of vCPUs, listening on `127.0.0.1:8080`, so the client connects through SSH tunnels. For longer inputs, edit `CTX` at the top of `run.sh`.

## 3. SSH Configuration

Edit `~/.ssh/config` on local machine:

```
Host qwen-large
    HostName <EC2_PUBLIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/key.pem

Host qwen-small
    HostName <EC2_PUBLIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/key.pem
```

Set key permissions:

```
chmod 400 ~/.ssh/key.pem
```

Create SSH tunnels in separate terminals:

```
ssh -N -L 8080:127.0.0.1:8080 qwen-large
ssh -N -L 8081:127.0.0.1:8080 qwen-small
```

Verify:

```
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8081/health
```

Keep both tunnels running during benchmarking.

## 4. Benchmark Configuration

Edit `benchmark.py`:

```
MODEL = "qwen"

NODES = {
    "large": {
        "url": "http://127.0.0.1:8080",
        "ssh": "qwen-large",
    },
    "small": {
        "url": "http://127.0.0.1:8081",
        "ssh": "qwen-small",
    },
}
```

Default: 1 measured requests, 1 warm-up, 128 max output tokens, concurrency = 1.

## 5. Workloads

| Run   | Modality | Size                     | Dataset       |
| ----- | -------- | ------------------------ | ------------- |
| 01–03 | Text     | 512 / 1024 / 3072 tokens | CNN/DailyMail |
| 04–06 | Image    | 1 / 2 / 4 images         | COCO          |
| 07–09 | Audio    | 5 / 10 / 20 seconds      | LibriSpeech   |

Text sizes assume a 4096-token context window.

## 6. Run

Both scripts need the tunnels from section 3 running.

### Single node (`benchmark.py`)

Each node is measured on its own, one request at a time, for the workloads in section 5:

```
python benchmark.py --run 01   # single workload
python benchmark.py            # all workloads
```

Results are appended to `results/benchmark.csv`.

| Benchmark Metric                                 | Description                           |
| ------------------------------------------------ | ------------------------------------- |
| `model`, `node`, `modality`, `size`, `size_unit` | Experiment configuration              |
| `median_e2e_s`, `p95_s`                          | Request latency                       |
| `req_s`                                          | Sequential throughput                 |
| `ttft_s`                                         | Time to first token                   |
| `decode_tok_s`                                   | Decode throughput                     |
| `peak_ram_gb`                                    | Peak sampled process RSS              |
| `cpu_pct`                                        | CPU usage normalized to node capacity |
| `success`, `failed`                              | Request outcomes                      |

CPU/RAM are sampled remotely via SSH every 0.2 seconds. P95 with 1 repetition is exploratory. Keep the model, quantization, and runtime settings consistent across nodes.

## 7. Round-Robin Baseline

Round robin fixes each request's node in advance by its position in the list (A, B, A, B, ...), regardless of the request type or how busy the node is. `roundrobin.py` by default sends 50 requests in order, 4 at a time across all nodes: each time one finishes, the next is sent. A request sent to a busy node waits in that node's queue.

Edit `NODES` at the top of `roundrobin.py` (same format as `benchmark.py`). Entry order is the turn order; a single entry gives the fixed-node baseline. With the tunnels from section 3 running:

```
python roundrobin.py --requests 10 --inflight 2   # quick test: 10 requests, 2 in flight across all nodes
python roundrobin.py                              # full run: 50 requests, 4 in flight, seed 0
```
Default 50 requests use the section 5 sizes in a 3:1:1 text/image/audio mix (30 / 10 / 10), with temperature 0, at most 128 output tokens, no prompt cache, and thinking off. One unrecorded warm-up runs per node and modality.

Results are **appended** and linked by `timestamp_utc`:

```
results/roundrobin_requests.csv   # one row per request
results/roundrobin.csv            # one row per run and group: all, each modality, each node
```

| Per-request metric                       | Description                                    |
| ---------------------------------------- | ---------------------------------------------- |
| `timestamp_utc`, `policy`, `seed`, `inflight` | Run configuration                         |
| `index`                                  | Send order                                     |
| `node`, `model`                          | Node used and model reported by `llama-server` |
| `modality`, `size`                       | Request type and size                          |
| `sent_s`                                 | Send time since the start of the run           |
| `e2e_s`                                  | Send to end of stream, including queueing      |
| `ttft_s`                                 | Time to first token                            |
| `decode_tok_s`, `output_tokens`          | Decode throughput and tokens generated (≤ 128) |
| `status`, `error`                        | Request outcome                                |

| Summary metric                                            | Description                                                          |
| --------------------------------------------------------- | -------------------------------------------------------------------- |
| `timestamp_utc`, `policy`, `requests`, `inflight`, `seed` | Run configuration                                                    |
| `model`                                                   | Model reported by `llama-server` (node rows: that node's model)      |
| `group`                                                   | Requests the row summarizes: `all`, a modality, or a node            |
| `median_e2e_s`, `p95_s`                                   | Request latency in the group                                         |
| `ttft_s`                                                  | Median time to first token in the group                              |
| `req_s`                                                   | Successful requests / whole run's wall time (`all` row = throughput) |
| `peak_ram_gb`, `cpu_pct`                                  | Node rows only, as in section 6                                      |
| `success`, `failed`                                       | Request outcomes                                                     |

Every run sends the same 50 inputs; `--seed` only changes their order, and with it the node assignment. Thinking is off because Gemma 4 would otherwise spend all 128 tokens reasoning and return no answer; Qwen is unaffected. Replies can stop before 128 tokens and text sizes are counted with the Qwen tokenizer, so compare `output_tokens` and `decode_tok_s` alongside `e2e_s` across models. Results are written only when a run finishes.
