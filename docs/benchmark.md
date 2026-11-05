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

Server settings: `-np 2` slots by default (2 requests processed at once) with `-c` = 4096 × slots, so every slot has 4096 tokens for input + output; `-t` = number of vCPUs (on Intel these include hyperthreads: 8 vCPU = 4 physical cores), listening on `127.0.0.1:8080`. `SLOTS=4 ./run.sh qwen` restarts the server with 4 slots (`-c 16384`); `CTX=...` overrides the total context. `benchmark.py` reads the slot count from the server and refuses a `--concurrency` above it, unless `--allow-queue` is given: then the extra requests wait in `llama-server`'s queue and `queue_s` measures that wait.

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

Each workload sends 1 unrecorded warm-up and then `--repeats` measured requests (default 5), each with a different input (question, article, image or recording), at temperature 0, at most 128 output tokens, no prompt cache. `--repeats` is requests per workload, not repetitions of the whole experiment; to repeat the experiment, run the command again (rows are appended with a new `run_id`).

**Output modes.** `natural` stops when the answer ends, so output length differs between requests and can differ between nodes for the same input. `fixed` sends `ignore_eos`, so every request generates exactly 128 tokens: compare `e2e_s` across nodes on `fixed` rows, and use `natural` rows for realistic latency. `ttft_s`, `prefill_tok_s`, `decode_tok_s` and `tpot_ms` do not depend on output length in either mode.

## 5. Run

Run inside tmux so the run continues if SSH disconnects. The whole sweep is one command:

```
tmux new -s bench
./experiment.sh qwen                   # REPEATS=5 and QUEUE_REPEATS=2 by default
NODE=c7i.2xlarge-noamx ./experiment.sh qwen   # label a variant of the same instance type
```

| Server (`run.sh`) | Concurrency | Waiting in the queue | Output modes | Repeats | Measured requests |
| ----------------- | ----------- | -------------------- | ------------ | ------- | ----------------- |
| `SLOTS=2`         | 1, then 2   | 0                    | natural, fixed | 5     | (45 + 90) × 2 = 270 |
| `SLOTS=2`         | 5, then 10  | 3, then 8            | fixed        | 2       | 90 + 180 = 270    |
| `SLOTS=4`         | 4           | 0                    | natural, fixed | 5     | 180 × 2 = 360     |

900 requests in total. Each workload sends repeats × concurrency measured requests, so the queue steps use fewer repeats (`QUEUE_REPEATS`): at concurrency 10 that is still 20 requests per workload. They use fixed output only, so every request needs the same work and the wait reflects the queue, not answer lengths.

In a queue step, `benchmark.py` keeps `concurrency` requests outstanding (closed loop): `slots` of them are processed and the rest wait, and a new one is sent each time one finishes. The queue therefore always holds concurrency − slots requests. Compare `queue_s`, `ttft_s`, `median_e2e_s` and `req_s` across concurrency 2, 5 and 10: throughput should stay at its concurrency-2 value while waiting grows with the queue.

It activates `.venv`, restarts `llama-server` for each slot count, logs everything to `<results folder>/logs/experiment-<run_id>.log`, and ends by printing the request count per slots / concurrency / output mode against the expected count. The server is left running with 4 slots. If a step fails the script stops; everything written so far is kept.

Single passes, by hand:

```
source .venv/bin/activate
python benchmark.py --repeats 10                             # 1 request at a time
python benchmark.py --repeats 10 --concurrency 1 2 --output natural fixed
python benchmark.py --run 04                                 # a single workload
python benchmark.py --repeats 2 --concurrency 5 10 --output fixed --allow-queue   # queue, 2 slots
```

Detach with Ctrl-b d and reattach with `tmux attach -t bench`. The node label defaults to the EC2 instance type; add a suffix for a variant of the same instance, e.g. `--node c7i.2xlarge-noamx`. With `--concurrency 2`, each workload sends 2 × repeats measured requests, keeping 2 outstanding. `--url` and `--ssh` point the client at a server on another machine, for example through an SSH tunnel, but then network time is included again.

Results are **appended**, never overwritten, to one folder per node and model, so teammates' results can all be committed without conflicts:

```
results/phase1/<node>__<model>/summary_v2.csv              # one row per workload, concurrency and output mode
results/phase1/<node>__<model>/requests_v2.csv             # one row per request, written as each request finishes
results/phase1/<node>__<model>/system_info-<run_id>-np<slots>.txt   # llama-server CPU features
results/phase1/<node>__<model>/logs/experiment-<run_id>.log
```

e.g. `results/phase1/c7i.2xlarge__Qwen2.5-Omni-3B-Q8_0/`. Older `summary.csv` / `requests.csv` files (previous column layout) are left untouched. Results live on the instance's disk: copy the folder back with `scp -r <host>:aws-mm-router/results/phase1/<folder> results/phase1/` and commit it **before terminating the instance** (stopping it keeps them). `results/benchmark.csv` holds the earlier single-request c8g.4xlarge run in the previous format.

| Summary metric                                   | Description                                                   |
| ------------------------------------------------ | ------------------------------------------------------------- |
| `run_id`                                         | One label per invocation (`experiment.sh` uses one for the whole sweep) |
| `model`, `llama_build`                           | Model file and llama.cpp build reported by `llama-server`     |
| `slots`, `ctx_per_slot`                          | Server slots (`-np`) and context per slot, reported by `llama-server` |
| `node`, `run`, `modality`, `size`, `size_unit`   | Experiment configuration                                      |
| `output_mode`                                    | `natural` or `fixed` (`ignore_eos`, always 128 output tokens)  |
| `concurrency`, `requests`                        | Requests in flight at once, measured requests                 |
| `median_e2e_s`, `p95_s`                          | Request latency (depends on output length in `natural` mode)  |
| `ttft_s`                                         | Median time to first token                                    |
| `prefill_tok_s`                                  | Median `prompt_tokens / server_prompt_ms` × 1000 (includes image/audio encoding) |
| `decode_tok_s`, `tpot_ms`                        | Median per-request decode throughput, and time per output token = 1000 / `decode_tok_s` |
| `queue_s`                                        | Median `e2e_s` − server processing time: time waiting for a slot + HTTP overhead; ≈ 0 when concurrency ≤ slots, the queue wait with `--allow-queue` |
| `prompt_tokens`, `output_tokens`                 | Median input tokens (how many tokens the images/audio became) and output tokens |
| `req_s`, `out_tok_s`                             | Successful requests and output tokens per second of wall time |
| `peak_ram_gb`                                    | Peak sampled `llama-server` RSS                               |
| `cpu_pct`                                        | Median CPU usage normalized to node capacity                  |
| `success`, `failed`                              | Request outcomes                                              |

`requests_v2.csv` has the same configuration columns plus, per request: `index`, `input` (dataset position; the same on every node), `sent_s`, `status`, `error`, `finish_reason` (`length` = hit 128 tokens, `stop` = answer ended), `e2e_s`, `ttft_s`, `queue_s`, `prompt_tokens`, `output_tokens`, `prefill_tok_s`, `decode_tok_s`, `tpot_ms`, and `server_prompt_ms` / `server_predicted_ms` (llama-server's own prompt-processing and generation times; prompt processing includes image/audio encoding and matched `ttft_s` within about 0.1 s in a test). Match rows across nodes by `run`, `input`, `concurrency` and `output_mode`.

CPU/RAM are sampled every 0.2 seconds while a workload's measured requests run. Keep the model, quantization, and runtime settings consistent across nodes.

## 6. Answer Quality (once per quantization)

`benchmark.py` measures speed only. Quantization trades answer quality for speed and memory, so the quantization row also measures quality. Quality depends on the weights, not the machine, so this runs **once per quantization on one node**, not on every instance. With the server and datasets from sections 2–3 (start with whichever model is loaded):

```
python quality.py                    # collect answers from the model llama-server is serving
./run.sh qwen Q4_K_M && python quality.py
./run.sh qwen Q8_0 && python quality.py
./run.sh qwen f16 && python quality.py
python quality.py --score            # compare all collected models with the f16 answers
./perplexity.sh                      # perplexity of every downloaded quantization (stops llama-server)
```

| Task         | Inputs                                         | Score                                                  |
| ------------ | ---------------------------------------------- | ------------------------------------------------------ |
| `chat`       | 10 Dolly questions (run 00)                    | ROUGE-L vs the f16 answer, share identical to f16      |
| `summarize`  | 10 512-token CNN/DailyMail texts (run 01)      | ROUGE-L vs the f16 answer, share identical to f16      |
| `image`      | 10 single COCO photos (run 04)                 | ROUGE-L vs the f16 answer, share identical to f16      |
| `transcribe` | 10 whole LibriSpeech recordings of 5–25 s      | word error rate vs the human transcript, plus ROUGE-L vs f16 |
| perplexity   | 20 × 512 tokens of wikitext-2 test             | perplexity (lower is better), the standard llama.cpp quantization measure |

f16 is the unquantized model, so ROUGE-L vs f16 (word overlap, 1.0 = same words in the same order) measures how much quantization changed the answer, not whether it is correct: one different early token can send a long description in another direction. Word error rate and perplexity are the measures against ground truth. Transcription uses whole recordings (not the cut clips of `benchmark.py`) so the transcript matches the audio, asks for the transcript only, and allows 256 output tokens; the other tasks keep `benchmark.py`'s prompts and 128 tokens. All requests use temperature 0 and no prompt cache.

Results:

```
results/quality/answers.csv      # every answer, with the model, task, input and (transcription) human transcript
results/quality/scores.csv       # one row per model and task
results/quality/perplexity.csv   # one row per model; perplexity-<quant>.log holds the full output
```

`--inputs` and `--tasks` change the number of inputs per task and which tasks are collected (e.g. `--tasks transcribe` to redo one); scoring uses the latest answer per model, task and input.

## 7. Round-Robin Baseline (Phase 2)

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
