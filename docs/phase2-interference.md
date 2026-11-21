# Phase 2: Interference Between Modalities (`src/corun.py pairs`)

When two requests share a node (one per llama-server slot), each slows the other down. How much may depend on what they are: a long text prefill and a short chat answer may share the CPU well, while two image encodes may not. `corun.py pairs` measures this for every pair of workloads, Phase 1 style: on one node, one workload or pair at a time. It runs on the router machine and reaches the node over the network (`--url`, an SSH tunnel), so every request's time also includes the router-node hop; `summarize` therefore also gives the slowdown from llama-server's own times. (`corun.py` replaced `interference.py`, which ran on the node; `corun.py groups` measures k streams at once.)

The result is a **slowdown matrix** per node: slowdown(A | B) = A's latency while B runs in the other slot ÷ A's latency alone. It is an input the router can use: a request of A sent to a node whose other slot is busy with B should take about slowdown(A | B) × A's solo time. `ect` today assumes one fixed factor per node (`slot_gain`), the same for every pair.

The Phase 1 sweep only ran one workload at a time (`benchmark.py --concurrency 2` runs two requests of the same workload), so it covers the diagonal of this matrix but nothing off it.

First results, from a test run on a lab server rather than AWS: [`analysis/interference/`](../analysis/interference/README.md).

## How it works

For each workload alone, then for each pair (A, B) on a llama-server with 2 slots:

- **Two closed-loop streams, one per slot.** Stream A sends workload A again and again, with a new input each time and one request in flight. Stream B does the same with workload B. Both keep going until each has finished `--repeats` requests. The shorter workload therefore runs many times alongside one long request.
- **Random pauses** of up to `--overlap` s (default 0.5) before each request shift the two streams against each other. A request therefore meets the other workload in every phase: prefill, image or audio encoding, decoding.
- **Timestamps:** every request's start, first token and end are logged on a clock shared by both streams, along with the arrival time of every output token (`token_s`), from which `summarize` derives the gaps and stalls.
- **Fixed output** by default (`--output fixed`): every request generates 128 tokens, so latency doesn't depend on how long the answer is.
- **Inputs are built before measuring,** so building them takes no CPU while a pair runs. `--pool` (40) distinct inputs per workload are prebuilt; a self-pair's two streams take alternate ones.
- **Server CPU and memory** are sampled for each pair over SSH (`--ssh`), as in `benchmark.py`.
- **Encoding time:** with the patched llama-server built by `run.sh`, each request also records `encoding_ms`, so `summarize` shows which stage slows down.

`summarize` keeps a paired request only if the other stream was busy for at least `--min-overlap` (0.8) of its lifetime. This drops, for example, the last short request that finishes after the long one has ended. It then compares the kept requests with the same workload alone.

By default the workloads are one size per modality: chat, 1,024-token text, 1 image and 10 s audio (`--runs 00 02 04 07`). That gives 4 solo measurements and 10 pairs, including each workload with itself.

## Running it

The routing cluster's nodes already have llama.cpp, the model and the datasets on disk (stopping an instance keeps them). On each node to profile (the c7a and m8g; see **Nodes** below for the c7i), the two nodes in parallel:

```bash
# 1. on the node (ssh mm-c7a, see docs/cluster-access.md): latest code, server with 2 slots
cd ~/aws-mm-router && git pull
CACHE_RAM=0 ./src/run.sh qwen           # patched llama-server, 2 slots; waits until ready

# 2. on the router, inside tmux so it survives an SSH disconnect (~15-25 min with the defaults);
#    18080 is the router's tunnel to the c7a's llama-server
tmux new -s interference
python src/corun.py pairs --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a
#    detach: Ctrl-b d; reattach: tmux attach -t interference

# 3. matrix
python src/corun.py summarize results/interference/c7a.2xlarge-p2__Qwen2.5-Omni-3B-Q8_0
```

Copy the results back to your laptop before stopping the instances, then commit them:

```bash
rsync -a mm-router:aws-mm-router/results/interference/ results/interference/
```

Stop the instances when done. If a node's dataset folder is missing (e.g. a new instance), set it up first as in [baseline-benchmark.md](baseline-benchmark.md) §5.2.

**Measuring each workload alone again.** The Phase 1 profiles already have each workload alone (concurrency 1, fixed output, 2 slots, the same inputs). `run` still measures them again first, which adds about 3–5 minutes. That way, a slowdown compares two measurements from the same instance, server and hour: a stopped and restarted instance can land on a different physical machine, and the solo runs also give the baseline for the pauses between tokens, which Phase 1 didn't record. Check that the new solo times are close to Phase 1's as a sanity check.

## Choosing modalities and sizes

Combinations are chosen by workload, using the ids from `benchmark.py` `RUNS`. Each id is one modality at one size:

| Id | Workload | Id | Workload |
| --- | --- | --- | --- |
| `00` | chat | `05` | 2 images |
| `01` | text, 512 tokens | `06` | 4 images |
| `02` | text, 1,024 tokens | `07` | audio, 10 s |
| `03` | text, 3,072 tokens | `08` | audio, 40 s |
| `04` | 1 image | | |

- **`--runs`** chooses the workloads. Each is measured alone, then every pair of them is measured, including each with itself. The default `--runs 00 02 04 07` (one size per modality) therefore covers every modality combination: chat+text, chat+image, chat+audio, text+image, text+audio, image+audio, and each modality with itself.
- **`--pairs`** runs only the listed pairs. Both workloads of each pair must also be in `--runs`, since each is measured alone for comparison.
- **`--no-self`** leaves out each workload paired with itself.

On the node, in tmux, with llama-server started with 2 slots (`./src/run.sh qwen`):

```bash
# every modality combination (4 alone + 10 pairs)
python src/corun.py pairs --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a

# only text+image and image+audio
python src/corun.py pairs --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a --runs 02 04 07 --pairs 02+04 04+07

# only image and audio: each alone, image+image, image+audio, audio+audio
python src/corun.py pairs --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a --runs 04 07

# the largest sizes, without self-pairs
python src/corun.py pairs --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a --runs 03 06 08 --no-self

# how image size changes interference with text: 1, 2 and 4 images against 1,024-token text
python src/corun.py pairs --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a --runs 02 04 05 06 --pairs 02+04 02+05 02+06

# the slowdown matrix
python src/corun.py summarize results/interference/c7a.2xlarge-p2__Qwen2.5-Omni-3B-Q8_0
```

**Why there are no percentage mixes here.** A node has 2 slots, so at most two requests ever run together. This script measures each pair directly rather than mixes like 50% text / 50% image. Mixes belong in the cluster test: `tracegen.py make --mix text=50,image=50` makes the trace and `main.py run` replays it (see "Using it in the router"). That checks whether the pair measurements predict what happens under mixed traffic.

## Commands

| Option | Default | Meaning |
| --- | --- | --- |
| `--runs` | `00 02 04 07` | Workloads (`benchmark.py` `RUNS`) measured alone and in pairs |
| `--pairs` | every pair of `--runs` | Only these pairs, e.g. `02+04`; both workloads must be in `--runs` |
| `--no-self` | off | Leave out each workload paired with itself |
| `--repeats` | 5 | Finished requests per stream |
| `--overlap` | 0.5 s | Longest random pause before each paired request |
| `--output` | `fixed` | `fixed` (128 tokens) or `natural` |
| `--pool` | 40 | Distinct inputs prebuilt per workload |
| `--node` (required), `--url`, `--ssh`, `--seed`, `--where` | | Node label, its llama-server, the SSH host to sample its CPU/RAM on |

`summarize` takes `--min-overlap` (0.8) and `--run-id` (only those runs). It prints the slowdown, server-side slowdown, encode slowdown (patched server), stall, pair-gain and request-count matrices and writes `matrix.csv`.

**Time:** each pair runs until the longer workload has finished `--repeats` requests alongside the other one. The default set takes roughly 10–20 minutes per node, longer with the largest sizes (3,072-token text and 40 s audio take about 30–100 s each when alone).

**Nodes:** the c7i.xlarge (8 GB) runs with 1 slot, so it can't be measured as is. With 2 slots and `--cache-ram 0` it may fit the small sizes. Check `peak_ram_gb` in `pairs.csv` before trying 4 images or 40 s audio there.

## Output

Results go to `results/interference/<node>__<model>/`, one folder per node and model as in `results/phase1/`.

| File | One row per | Contents |
| --- | --- | --- |
| `requests.csv` | request | Pair, stream (A/B), workload, `other_run` (empty when alone), input, `request_bytes`, `router`, `start_s`/`first_s`/`end_s` on the pair's clock, latency, TTFT, prefill and decode speed, every token's arrival time (`token_s`), the server's own timings including `encoding_ms` |
| `pairs.csv` | solo or pair measurement | Wall time, requests per stream, failures, `peak_ram_gb`, `cpu_pct` |
| `matrix.csv` (from `summarize`) | workload × what ran alongside (`solo` for alone) | The metrics below |

| Metric | Meaning |
| --- | --- |
| `slowdown` | Median latency alongside ÷ median latency alone |
| `server_slowdown` | The same from llama-server's prompt + decode time: without the router-node network |
| `ttft_slowdown`, `encode_slowdown`, `prefill_slowdown`, `decode_slowdown` | The same for time to first token, encoder time (patched server only), prompt time minus encoding, and decode speed. These show which stage suffers. Prompt time also holds any time the slot was blocked by the other one's image or audio. |
| `max_gap_s` | Longest pause between two output tokens over the requests (`solo_max_gap_s`: alone). A long pause while the other slot starts on an image or audio clip would show that its encoding holds up the other slot. |
| `stall_s` | Mean time per request spent in pauses longer than 5× the request's median gap between tokens |
| `pair_gain` | 1/slowdown(A\|B) + 1/slowdown(B\|A), the "weighted speedup" used for co-scheduling jobs on shared processors. 1 means sharing the node gains nothing over running the two one after the other; 2 means neither slows the other; below 1 means they would be faster one at a time. |
| `peak_ram_gb` | Peak llama-server memory during the pair, for the router's memory limits |
| `request_bytes` (in `requests.csv`) | Size of the request body, for the network measurements |

## Using it in the router

`corun.load_matrix(folder)` returns `{(run, other_run): slowdown}`. For workloads that weren't profiled, it also has entries keyed by modality (the median over that modality's measured sizes), e.g. `('text', 'image')`. For a 2-slot node whose other slot is running B, a new request A would take about `slowdown(A | B) × cost(A)`, and B's remaining time stretches by `slowdown(B | A)`. An interference-aware `ect` would use these in place of the single `slot_gain`, and prefer the node whose current work pairs best with the new request.

To check at cluster level that the matrix predicts what happens under real traffic, run two-modality traces made with `tracegen.py`. For example, `tracegen.py make --mix text=50,image=50 --load 0.9 --nodes nodes.json` replayed with and without the interference-aware policy.

## Limits

- Pairs only: with 2 slots a node never runs more than two requests at once. More slots would need triples.
- A slowdown is a median over the phases the two requests happened to overlap in. The timestamps in `requests.csv` let the analysis split it by phase (e.g. A's prefill during B's decoding) if the medians hide too much.
- The client shares the node's CPU, as in `benchmark.py`. Payloads are prebuilt, but streaming two responses still costs a little CPU.
