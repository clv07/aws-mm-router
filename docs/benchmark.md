# Phase 1 Benchmark

Phase 1 measures each node on its own: every teammate runs the same workloads with the same settings on their EC2 instance, so results differ only by the instance (or, for one row, the quantization). `run.sh` sets up the inference server, and `benchmark.py` runs **on the same instance**, next to the server, so that network delay and image/audio upload time from a laptop are not part of the measurements and a laptop going to sleep cannot stop the run. `roundrobin.py` is the Phase 2 round-robin baseline over several nodes (section 6).

## 1. Instance

| Setting       | Value                                                                    |
| ------------- | ------------------------------------------------------------------------ |
| AMI           | Ubuntu Server 24.04 LTS, 64-bit (x86) for c6i/c7i/c7a/m7i/r7i, 64-bit (Arm) for c8g |
| Storage       | 30 GiB gp3 (llama.cpp build, models and datasets do not fit in the default 8 GiB) |
| Inbound rules | SSH from My IP only                                                      |

Peak `llama-server` memory with Q8_0 was 9.4–10.3 GB on c8g and c7i but 6.7 GB on c7a (same model and settings). On 8 GB instances (c7i.xlarge, c8g.xlarge) run one short test first and check `tail ~/llama-server.log` and `free -g` before the full run.

## 2. Inference Server

On the instance, clone this repository and start `llama-server`:

```
git clone https://github.com/clv07/aws-mm-router
cd aws-mm-router
./run.sh qwen            # Qwen2.5-Omni-3B Q8_0, the default for every node
./run.sh qwen Q4_K_M     # quantization rows only: Q4_K_M or f16
```

`run.sh` builds llama.cpp (commit `8642541b2`, CPU only) and downloads the model if missing, starts `llama-server` in tmux session `llama`, and returns once the model is loaded. The image/audio encoder (`mmproj-...-Q8_0.gguf`) is the same for every quantization, so only the language model's weights change. Rerun it to switch models (the restart aborts requests in flight) or after an instance stop/start. The first run needs `sudo` and a few minutes; if the build fails it prints the end of `~/llama-build.log`.

Monitor the server:

```
tmux attach -t llama          # leave with Ctrl-b d; Ctrl-C stops the server
tail -f ~/llama-server.log
```

Server settings: `-c 8192 -np 2` (2 requests at a time, 4096 tokens each for input + output), `-t` = number of vCPUs (on Intel these include hyperthreads: 8 vCPU = 4 physical cores), listening on `127.0.0.1:8080`. For longer inputs, edit `CTX` at the top of `run.sh`.

## 3. Client Setup

On the same instance, in the repository folder:

```
python3 -m venv .venv
source .venv/bin/activate
pip install requests datasets transformers soundfile
```

Download datasets (a few minutes inside AWS, about 3 GB):

```
python dolly.py --download
python cnn_dailymail.py --download
python coco.py --download
python librispeech.py --download
```

## 4. Workloads

| Run   | Modality | Size                     | Dataset       | Prompt                                   |
| ----- | -------- | ------------------------ | ------------- | ---------------------------------------- |
| 00    | Chat     | short question (~40 tokens) | Dolly-15k (open_qa, general_qa) | the question                |
| 01–03 | Text     | 512 / 1024 / 3072 tokens | CNN/DailyMail | Summarize the following news article     |
| 04–06 | Image    | 1 / 2 / 4 images         | COCO          | Describe the image or images concisely   |
| 07–08 | Audio    | 10 / 40 seconds          | LibriSpeech   | Transcribe this speech exactly           |

Run 00 is the short-prompt case (decode-heavy); runs 01–03 are long-input (prefill-heavy). Text sizes assume a 4096-token context window.

llama.cpp pads Qwen audio to whole 30-second windows (the clip plus 30 s of silence, cut into 30 s windows, a partial last window dropped), so any clip under 30 s costs one window: 5, 10 and 20 s took the same time. 10 s (one window) and 40 s (two windows, joined from consecutive LibriSpeech recordings) show that step.

Each workload sends 1 unrecorded warm-up and then 5 measured requests, each with a different input (question, article, image or recording), at temperature 0, at most 128 output tokens, no prompt cache.

## 5. Run

Run inside tmux so the run continues if SSH disconnects:

```
tmux new -s bench
source .venv/bin/activate
python benchmark.py                    # all workloads, 1 request at a time (the main Phase 1 run)
python benchmark.py --concurrency 2    # optional: 2 at a time (= the server's 2 slots)
python benchmark.py --run 04           # a single workload
```

Detach with Ctrl-b d and reattach with `tmux attach -t bench`. The node label defaults to the EC2 instance type; add a suffix for a variant of the same instance, e.g. `--node c7i.2xlarge-noamx`. With `--concurrency 2`, each workload sends 2 × 5 = 10 measured requests, keeping 2 outstanding. `--url` and `--ssh` point the client at a server on another machine, for example through an SSH tunnel, but then network time is included again.

Results are **appended** to one folder per node and model, so teammates' results can all be committed without conflicts:

```
results/phase1/<node>__<model>/summary.csv    # one row per workload and concurrency
results/phase1/<node>__<model>/requests.csv   # one row per request
```

e.g. `results/phase1/c7i.2xlarge__Qwen2.5-Omni-3B-Q8_0/`. Copy the folder back with `scp -r <host>:aws-mm-router/results/phase1/<folder> results/phase1/` and commit it. `results/benchmark.csv` holds the earlier single-request c8g.4xlarge run in the previous format.

| Summary metric                                   | Description                                                   |
| ------------------------------------------------ | ------------------------------------------------------------- |
| `model`, `llama_build`                           | Model file and llama.cpp build reported by `llama-server`     |
| `node`, `run`, `modality`, `size`, `size_unit`   | Experiment configuration                                      |
| `concurrency`, `requests`                        | Requests in flight at once, measured requests                 |
| `median_e2e_s`, `p95_s`                          | Request latency                                               |
| `ttft_s`                                         | Median time to first token                                    |
| `prompt_tokens`, `output_tokens`                 | Median input tokens (how many tokens the images/audio became) and output tokens |
| `req_s`, `out_tok_s`                             | Successful requests and output tokens per second of wall time |
| `decode_tok_s`                                   | Median per-request decode throughput                          |
| `peak_ram_gb`                                    | Peak sampled `llama-server` RSS                               |
| `cpu_pct`                                        | Median CPU usage normalized to node capacity                  |
| `success`, `failed`                              | Request outcomes                                              |

`requests.csv` has the same configuration columns plus, per request: `index`, `input` (dataset position), `sent_s`, `status`, `error`, `e2e_s`, `ttft_s`, `prompt_tokens`, `output_tokens`, `decode_tok_s`, and `server_prompt_ms` / `server_predicted_ms` (llama-server's own prompt-processing and generation times; prompt processing includes image/audio encoding and matched `ttft_s` within about 0.1 s in a test).

CPU/RAM are sampled every 0.2 seconds while a workload's measured requests run. Keep the model, quantization, and runtime settings consistent across nodes.

## 6. Round-Robin Baseline (Phase 2)

Round robin fixes each request's node in advance by its position in the list (A, B, A, B, ...), regardless of the request type or how busy the node is. `roundrobin.py` by default sends 50 requests in order, 4 at a time across all nodes: each time one finishes, the next is sent. A request sent to a busy node waits in that node's queue.

`roundrobin.py` runs on the local machine and reaches the nodes through SSH tunnels. Edit `~/.ssh/config` on the local machine:

```
Host node-c7a
    HostName <EC2_PUBLIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/key.pem

Host node-c7i
    HostName <EC2_PUBLIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/key.pem
```

Set key permissions with `chmod 400 ~/.ssh/key.pem`, then create one tunnel per node in separate terminals and keep them running:

```
ssh -N -L 8080:127.0.0.1:8080 node-c7a
ssh -N -L 8081:127.0.0.1:8080 node-c7i
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8081/health
```

Edit `NODES` at the top of `roundrobin.py`: one entry per node with its tunnel `url` and `ssh` host (used to sample CPU/RAM). Entry order is the turn order; a single entry gives the fixed-node baseline. With the tunnels running:

```
python roundrobin.py --requests 10 --inflight 2   # quick test: 10 requests, 2 in flight across all nodes
python roundrobin.py                              # full run: 50 requests, 4 in flight, seed 0
```
Default 50 requests use the sizes in `SIZES` at the top of `roundrobin.py` (text 512 / 1024 / 3072 tokens, 1 / 2 / 4 images, 5 / 10 / 20 s audio) in a 3:1:1 text/image/audio mix (30 / 10 / 10), with temperature 0, at most 128 output tokens, no prompt cache, and thinking off. One unrecorded warm-up runs per node and modality.

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
| `peak_ram_gb`, `cpu_pct`                                  | Node rows only, as in section 5                                      |
| `success`, `failed`                                       | Request outcomes                                                     |

Every run sends the same 50 inputs; `--seed` only changes their order, and with it the node assignment. Thinking is off because Gemma 4 would otherwise spend all 128 tokens reasoning and return no answer; Qwen is unaffected. Replies can stop before 128 tokens and text sizes are counted with the Qwen tokenizer, so compare `output_tokens` and `decode_tok_s` alongside `e2e_s` across models. Results are written only when a run finishes.
