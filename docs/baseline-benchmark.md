# Phase 1 Baseline Benchmarks: Multimodal Inference on CPU Instances

The baseline benchmarks measure how the choice of CPU instance affects a multimodal model's latency, throughput and cost, one node at a time. The routing benchmarks (`routing-benchmark.md`) use these measurements to route requests across a cluster of different nodes.

## 1. Setup

**Model.** Qwen2.5-Omni-3B, a 3-billion-parameter model that takes text, images and audio and answers in text. It runs on CPU in llama.cpp's `llama-server` (built from commit `8642541b2`), as GGUF weights: Q8_0 (8-bit) unless a run varies the quantization, with the image/audio encoder (`mmproj`) always in Q8_0. The server processes 2 requests at once (2 slots) with 4,096 tokens of context each.

**Workloads.** Four request types (modalities), in nine sizes, each from a public dataset:

| Run   | Modality | Size                        | Dataset                          | Prompt                                 |
| ----- | -------- | --------------------------- | -------------------------------- | -------------------------------------- |
| 00    | Chat     | short question (~40 tokens) | Dolly-15k (open_qa, general_qa)  | the question                           |
| 01–03 | Text     | 512 / 1024 / 3072 tokens    | CNN/DailyMail                    | Summarize the following news article   |
| 04–06 | Image    | 1 / 2 / 4 images            | COCO 2017 validation             | Describe the image or images concisely |
| 07–08 | Audio    | 10 / 40 seconds             | Mini LibriSpeech                 | Transcribe this speech exactly         |

Chat is a short prompt with a long answer, so its time is mostly spent generating (decode). Long text, images and audio are mostly spent processing the input (prefill, and encoding for images and audio). llama.cpp pads Qwen audio to whole 30-second windows, so any clip under 30 s costs one window: 10 s is one window and 40 s is two.

Every request uses temperature 0 (greedy decoding), at most 128 output tokens and no prompt cache, so the same input gives the same answer on every node.

**Infrastructure.** AWS EC2 in us-west-2, Ubuntu 24.04 or later (Arm image for Graviton), 30 GiB gp3 disk. The client runs **on the same instance** as the server, so network delay and image/audio upload time are not part of the measurements.

## 2. Methodology

**One variable at a time.** The baseline is a c7i.2xlarge (Intel Sapphire Rapids with AMX, 8 vCPUs = 4 physical cores, 16 GB) serving Q8_0. Every other configuration changes exactly one thing from it (in bold), so differences can be attributed to that one variable:

| Variable     | Instance    | CPU                       | vCPUs (physical cores) | RAM       | Quantization      | ~$/h | Results folder (`results/phase1/…`) |
| ------------ | ----------- | ------------------------- | ---------------------- | --------- | ----------------- | ---- | ----------------------------------- |
| Baseline     | c7i.2xlarge | Intel + AMX               | 8 (4)                  | 16 GB     | Q8_0              | 0.36 | `c7i.2xlarge__…-Q8_0`               |
| Architecture | c7a.2xlarge | **AMD Genoa**             | 8 (8)                  | 16 GB     | Q8_0              | 0.41 | `c7a.2xlarge__…-Q8_0`               |
|              | c8g.2xlarge | **Arm Graviton4**         | 8 (8)                  | 16 GB     | Q8_0              | 0.32 | `c8g.2xlarge__…-Q8_0`               |
|              | c7i.2xlarge | **Intel, AMX off**        | 8 (4)                  | 16 GB     | Q8_0              | 0.36 | `c7i.2xlarge-noamx__…-Q8_0`         |
|              | c6i.2xlarge | **Intel Ice Lake, no AMX** | 8 (4)                 | 16 GB     | Q8_0              | 0.34 | `c6i.2xlarge__…-Q8_0`               |
| vCPUs        | c7i.xlarge  | Intel + AMX               | **4 (2)**              | 8 GB      | Q8_0              | 0.18 | `c7i.1xlarge-q8__…-Q8_0`            |
|              | c7i.4xlarge | Intel + AMX               | **16 (8)**             | 32 GB     | Q8_0              | 0.71 | `c7i.4xlarge-q8__…-Q8_0`            |
| RAM          | m7i.2xlarge | Intel + AMX               | 8 (4)                  | **32 GB** | Q8_0              | 0.40 | `m7i.2xlarge__…-Q8_0`               |
|              | r7i.2xlarge | Intel + AMX               | 8 (4)                  | **64 GB** | Q8_0              | 0.53 | `r7i.2xlarge__…-Q8_0`               |
| Quantization | c7i.2xlarge | Intel + AMX               | 8 (4)                  | 16 GB     | **Q4_K_M (4-bit)** | 0.36 | `c7i.2xlarge__…-Q4_K_M`            |
|              | c7i.2xlarge | Intel + AMX               | 8 (4)                  | 16 GB     | **f16 (16-bit)**  | 0.36 | `c7i.2xlarge__…-f16`                |

The vCPU rows change RAM along with vCPUs, because EC2 ties them within a family.

**Same type, different machine.** Teammates ran on their own instances, so the baseline was measured on three c7i.2xlarge machines (`c7i.2xlarge`, `c7i.2xlarge-abhinav`, `c7i.2xlarge-amx`). They differed by up to 27% in prefill time and under 3% in decode speed. Each comparison uses runs from one machine where possible: AMX on and off ran on the same instance (`-amx` / `-noamx`), m7i and r7i are compared with `-abhinav`, and the quantizations ran on one instance.

**Concurrency and queueing.** On a subset of nodes (c6i, c7a, c7i with Q8_0 and Q4_K_M, c8g) we also vary how many requests are in flight: 1, 2 (both slots busy), and 4 with a 4-slot server. At 5 and 10 there are more requests than slots, so we deliberately let a queue build up in the server and measure each request's time waiting in it (`queue_s`) along with latency and throughput. This shows how much a second slot helps, and what queueing does to latency and throughput once a node has more requests than slots.

**Answer quality.** Quantization trades quality for speed, so each quantization's answers are also scored, once on one node, since quality depends on the weights and not on the machine.

## 3. How the Benchmark Works

`benchmark.py` runs on the instance next to `llama-server`. For each workload it sends one unrecorded warm-up request, then `--repeats` measured requests (default 5), each with a different input, keeping `--concurrency` requests in flight (closed loop: a new request is sent when one finishes). Responses are streamed, so the time of the first output token is known. A background sampler reads `llama-server`'s CPU and memory use every 0.2 s.

**Output modes.** `natural` stops when the answer ends, so output length differs between requests and between quantizations. `fixed` keeps generating to exactly 128 tokens (`ignore_eos`), so end-to-end times are comparable across nodes. Prefill and decode speeds do not depend on the mode.

## 4. What We Measure

Per request (`requests_v2.csv`) and summarized per workload as medians (`summary_v2.csv`):

| Metric                                | Meaning                                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------------- |
| `e2e_s` (`median_e2e_s`, `p95_s`)     | End-to-end latency: request sent to last token received                                   |
| `ttft_s`                              | Time to first token: waiting, encoding images/audio and prefill                           |
| `prefill_tok_s`                       | Input tokens processed per second (images and audio count as the tokens they become)      |
| `decode_tok_s`, `tpot_ms`             | Output tokens generated per second, and time per output token                             |
| `queue_s`                             | Time waiting for a free slot (≈ 0 unless concurrency > slots)                             |
| `req_s`, `out_tok_s`                  | Throughput: completed requests and output tokens per second                               |
| `prompt_tokens`, `output_tokens`      | Input size after tokenization (how many tokens an image or clip became), and answer length |
| `peak_ram_gb`, `cpu_pct`              | Peak memory of `llama-server`, and its median CPU use as a share of all vCPUs             |

Each row also records the configuration: `node`, `model`, `llama_build`, `slots`, `ctx_per_slot`, `output_mode`, `concurrency`, `run_id`, and, per request, `input` (the dataset position, the same on every node) and `finish_reason`.

**Answer quality** (`quality.py`, `perplexity.sh`), per quantization:

| Task         | Inputs                                       | Score                                                 |
| ------------ | -------------------------------------------- | ----------------------------------------------------- |
| `chat`       | 10 Dolly questions                           | ROUGE-L vs the f16 answer, share identical to f16     |
| `summarize`  | 10 512-token CNN/DailyMail articles          | ROUGE-L vs the f16 answer, share identical to f16     |
| `image`      | 10 single COCO photos                        | ROUGE-L vs the f16 answer, share identical to f16     |
| `transcribe` | 10 whole LibriSpeech recordings of 5–25 s    | word error rate vs the human transcript               |
| perplexity   | 20 × 512 tokens of wikitext-2 test           | perplexity (lower is better)                          |

ROUGE-L against f16 measures how much quantization changed the answer, not whether it is correct; word error rate and perplexity measure against ground truth.

## 5. Reproducing the Results

### 5.1 Instance and server

Launch an instance from the table in section 2 (Ubuntu, 30 GiB gp3, inbound SSH from your IP only), then on it:

```
git clone https://github.com/clv07/aws-mm-router && cd aws-mm-router
./src/run.sh qwen              # builds llama.cpp, downloads Q8_0 + mmproj, starts llama-server (2 slots) in tmux
./src/run.sh qwen Q4_K_M       # quantization rows: Q4_K_M or f16
SLOTS=1 ./src/run.sh qwen      # 8 GB instances (c7i.xlarge): see below
```

`run.sh` returns once the model is loaded; `tmux attach -t llama` shows the server and `~/llama-server.log` its output. Rerun it after an instance stop/start.

**Memory.** Q8_0 peaked at 9–12 GB with 2 slots and up to 14.5 GB with 4 slots. On the 8 GB c7i.xlarge, one request at a time peaked at 5.9 GB, but two 4-image requests at once ran out of memory and the kernel killed `llama-server` (`sudo dmesg | grep -i oom`), so use one slot there.

### 5.2 Client and datasets

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/loaders/dolly.py --download && python src/loaders/cnn_dailymail.py --download && python src/loaders/coco.py --download && python src/loaders/librispeech.py --download
```

The datasets (~3 GB) go to `dataset/`, which git ignores.

### 5.3 Running the benchmarks

Run inside tmux (`tmux new -s bench`), so the run continues if SSH disconnects. `--node` sets the label used in the results folder name; the default is the EC2 instance type. Each run writes to `results/phase1/<label>__<model>/`.

**One request at a time (section 2 matrix).**

```bash
python src/benchmark.py --node <label>
```

For the AMX-off row, first disable AMX (**TODO: how AMX was disabled**), then run with `--node c7i.2xlarge-noamx`.

Results: `summary.csv` (one row per workload) and `requests.csv` (one row per request) in the folder listed for each row in section 2. These were collected with the first version of `benchmark.py` (commit `ceb04b5`); the current version measures the same thing and writes `summary_v2.csv` / `requests_v2.csv` instead. The c8g.2xlarge has only the `_v2` files: its one-at-a-time numbers are the concurrency-1 rows of the sweep below.

**Concurrency sweep.**

```bash
./src/experiment.sh qwen            # Q8_0
./src/experiment.sh qwen Q4_K_M     # Q4_K_M
```

`experiment.sh` runs the whole sweep in one command (~5 h, 900 measured requests):
- a 2-slot server at concurrency 1 and 2, natural and fixed output, 5 repeats;
- concurrency 5 and 10, fixed output, 2 repeats, with the extra requests waiting in the server's queue;
- a 4-slot server at concurrency 4;
- at the end, it prints the request count per step against the expected count.

The c7a.2xlarge and c8g.2xlarge ran a shorter version with `benchmark.py` directly:

```bash
python src/benchmark.py --concurrency 1 2 --output natural fixed      # c7a
python src/benchmark.py --repeats 2 --concurrency 5 --output fixed --allow-queue
python src/benchmark.py --concurrency 1 2 --output fixed               # c8g
```

Results: `summary_v2.csv` and `requests_v2.csv` (one row per workload × concurrency × output mode), plus the server logs in `logs/`, in these folders:

| Node                 | Results folder (`results/phase1/…`) |
| -------------------- | ----------------------------------- |
| c6i.2xlarge          | `c6i.2xlarge__…-Q8_0`               |
| c7i.2xlarge, Q8_0    | `c7i.2xlarge__…-Q8_0`               |
| c7i.2xlarge, Q4_K_M  | `c7i.2xlarge__…-Q4_K_M`             |
| c7a.2xlarge, partial | `c7a.2xlarge__…-Q8_0`               |
| c8g.2xlarge, partial | `c8g.2xlarge__…-Q8_0`               |

**Labels.** A suffix such as `-q8` or `-amx` marks a variant or a different machine of the same type; `c7i.1xlarge-q8` is the c7i.xlarge. The c7i.xlarge has no audio results: its server crashed, most likely out of memory (see 5.1).

### 5.4 Answer quality

On one node, with each quantization in turn:

```
./src/run.sh qwen f16 && python src/quality.py
./src/run.sh qwen Q8_0 && python src/quality.py
./src/run.sh qwen Q4_K_M && python src/quality.py
python src/quality.py --score      # compare every model with the f16 answers
./src/perplexity.sh                # perplexity of every downloaded quantization (stops llama-server)
```

### 5.5 Where results go

Results are only ever **appended**, to one folder per node label and model, so teammates' results never collide:

```
results/phase1/<node>__<model>/summary_v2.csv     # one row per workload, concurrency and output mode
results/phase1/<node>__<model>/requests_v2.csv    # one row per request, written as each one finishes
results/phase1/<node>__<model>/logs/              # experiment.sh console output
results/quality/{answers,scores,perplexity}.csv
```

Copy a folder off the instance (`scp -r <host>:aws-mm-router/results/phase1/<folder> results/phase1/`) before terminating it; stopping it keeps the disk.

Early test runs that predate this setup (a first c8g.4xlarge run and a closed-loop round-robin test) are in [`archive/`](../archive/) and are not used in the report.
