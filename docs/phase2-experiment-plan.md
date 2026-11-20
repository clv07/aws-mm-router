# Phase 2: Stages, Data Movement and Split (plan)

This plan covers the **stage breakdown** (encoding vs prompt processing vs decoding), **data movement**, and **splitting a request** across nodes, and how their results feed the routing policies. [phase2-profiling-plan.md](phase2-profiling-plan.md) covers pairwise interference, concurrency and the bursty replay; experiments 4 and 5 below refer to it.

**Goal:** route multimodal requests over a few different CPU nodes for latency and deadlines (SLOs), using what each node is good at. Phase 1 routed each request to one node. Phase 2 adds three things:
1. **Stages:** where a request's time goes (image/audio encoding, prompt processing, decoding).
2. **Splitting a request across nodes:** encode and process the prompt on one node, then move its KV cache to another node that generates the answer.
3. **The cost of moving data:** requests, responses and KV cache, over the network.

**What we split, and what we don't:**
- We split each request **at the stage boundary**: encoding + prompt processing on one node, decoding on another, with the KV cache moved in between ("EP+D"). llama-server supports this through slot save/restore.
- We **don't** split encoding from prompt processing (that would move embeddings, and llama-server can't take them).
- We **don't** split the model's layers across nodes. That only helps when a model doesn't fit in memory, and it makes one request slower.

**The router sits on top:** splitting becomes one more per-request decision for the router (where to process the prompt, and whether to hand the decoding to another node). Experiments 1–4 measure each piece in isolation; experiment 5 puts them into the router and compares policies under load.

## 1. Experiments

| # | What we run | What it tests | Question it answers | What we quantify | Tool |
| --- | --- | --- | --- | --- | --- |
| **1** | **Stage breakdown.** Each workload alone on each node (chat; text 512 / 1,024 / 3,072 tokens; 1 / 2 / 4 images; 10 / 40 s audio), timing encoding, prompt processing and decoding separately | Where a request's time goes, per modality and per CPU | **Is encoding the bottleneck?** For which modalities, sizes and nodes? Which node is fastest at each stage? | Encode s, prompt-processing s and decode s per request. Share of total time spent encoding. Tokens per image / audio second. Speed per stage per node (tokens/s; images or audio-seconds/s) | `patches/llama-stage-profiling.patch` (built by `run.sh`): llama-server reports each request's `encoding_ms`, so every tool records it, alone and under load; `benchmark_v3.py` for the Phase 1-style per-node profile |
| **2** | **Data movement.** Network round-trip and bandwidth between every pair of machines, through the router's tunnels; request and response bytes per modality; KV save, copy and restore per modality | The cost of every byte the system moves | How much does moving requests, streamed tokens and KV cost, next to compute? Does a KV file saved on x86 load on Arm? | Round-trip ms, MB/s per path. Bytes up/down per request. KV MB per request (≈36 KB per prompt token). Save / copy / restore s | `ping`, `iperf3`, `movement.py payload`; KV: `split.py` |
| **3** | **Split vs no split, one request at a time.** Each workload three ways: all on node A, all on node B, split A → B (c7a → m8g, c7a → c7i, c7i → m8g). Repeated with the link limited to 1 Gbps, 100 Mbps and 20 Mbps (`tc`) | Whether running each stage on its best node beats the KV-move cost | **When does splitting pay?** For which modalities, node pairs and link speeds? | Latency of each option, time saved by splitting, break-even bandwidth per modality, answer agreement | `split.py` (tested on s41), `tc` |
| **4** | **Interference.** Every pair of workloads together on each 2-slot node; the prompt batch size (`-b`/`-ub` 2,048 → 128) for chat paired with text, image and audio | What a request costs another request on the same node | How much does co-location slow each request? Can configuration stop chat from stalling, or must the router avoid it? Is the encoder the cause? | Slowdown matrix per node (end-to-end, server-side, per stage), longest pause in chat's output, pair gain, per batch size | `corun.py pairs` + `summarize` |
| **5** | **Routing under traffic.** Traces (steady and bursty, 60% and 90% load, default mix + held-out mixes) replayed with each policy: each node alone, `rr`, `least`, `ect`, `ect` + interference, `ect` + split | Whether the knowledge from 1–4 improves routing when requests overlap | How much does each added piece of knowledge improve tail latency and deadlines met? What does splitting cost in network traffic? | p50/p95/p99 latency per modality, % of deadlines met (end-to-end, and TTFT + TPOT), throughput, goodput, $ per 1,000 requests, bytes moved per policy (requests, streams, KV), router decision time | `tracegen.py`, `main.py run` with `router.py` policies (**split execution and new policies: to add**), `analysis/phase2/`, `movement.py netdev` |
| *6 (optional)* | *Multi-turn sessions:* for each new turn, keep it on the node holding its KV, move the KV, or recompute | The cost of accumulated context | When should a session move to another node? | Bytes per turn, latency per turn for each option | Session traces + runner support (**to add**) |

**How the results feed the router (experiment 5):**
- Experiment 1 gives the time of each stage on each node.
- Experiments 2 and 3 give the cost of moving the KV, and when splitting wins.
- Experiment 4 gives the penalty for sharing a node.

`ect` + split uses all three: for each request it estimates the finish time of every option (one node, or a split A → B) and picks the earliest.

## 2. Encoding: what we know and what experiment 1 adds

- **From the class readings:** ModServe found encoding is 79% of time to first token for Llama-3.2-11B and 25–54% for other vision-language models. TriInfer (HydraInfer) found encoding stalls other requests' decoding.
- **From our runs:**
  - Images and audio take much longer to process than text with the same number of tokens. On the c7a, prompt processing runs at 77 tokens/s for an image vs 154 for text (Phase 1 profiles).
  - First direct measurements: `llama-mtmd-cli` on s41 with 8 cores: **encoding is 52–53% of prompt time for 1, 2 and 4 images, and 39–43% for 10 s and 40 s audio** (28–41% of the whole request). The patched llama-server on a test node: 64% of prompt time for 4 images. The rest of the prompt time is the language model processing the media's tokens (~480 per image, 75 per audio second).
- **In our llama-server build, encoding can't be chunked.** An image or audio chunk is encoded and then fully prompt-processed in one call, which blocks the node's other slot. This is the likely cause of chat's long pauses next to images and audio.
- **Experiment 1 measures:**
  - the encoding share for every modality, size and node;
  - how encoding grows with images per request and audio length;
  - which CPU encodes fastest (Phase 1 suggested the c7a's 8 physical cores, by more than 2×).

## 3. Build-up: stages over two weeks

| Stage | When | What | Output | Decision at the end |
| --- | --- | --- | --- | --- |
| **0. Develop on the lab servers** | Days 1–2 (no AWS cost) | Stage profiling (patched llama-server), extra logging and analysis; split execution in `main.py` / `router.py`; dry-run experiments 1, 3 and 4 | Working scripts; rough numbers | – |
| **1. Bring up the cluster** | Day 3 | Start the nodes and the router (§4, [cluster-access.md](cluster-access.md)); network baseline (experiment 2, first half) | Network numbers | – |
| **2. Measure the pieces** | Days 3–6, in parallel | Experiments 1, 2 (KV half), 3 and 4 on AWS | Stage times, KV costs, split vs no split, slowdown matrices | **Does splitting pay on AWS?** Yes → `ect` + split goes into experiment 5. No → experiment 5 runs without it, and experiment 3's result is the finding. |
| **3. Router policies** | Days 5–8 | `ect` + interference and `ect` + split, using the numbers from stage 2; test each on a short trace | Policies that run end to end | Which configurations to run in full |
| **4. Main runs** | Days 8–11 | Experiment 5: 2 loads × 2 arrival patterns × 6 policies ≈ 24 runs × 17 min ≈ 7 cluster-hours, plus held-out mixes and repeat seeds for the final configurations | Results for the report | – |
| **5. Analysis and report** | Days 11–14 | Figures, tables, write-up; optional experiment 6 if stage 4 finishes early | Report | – |

### Splitting up the work

| Person | Stage 0–2 | Stage 3–5 |
| --- | --- | --- |
| 1 | Cluster bring-up, network and KV (experiment 2), split vs no split (experiment 3) | Bandwidth sweep; network section of the report |
| 2 | Stage breakdown (experiment 1): script + runs | Encoding section of the report |
| 3 | Interference and concurrency (experiment 4, [profiling plan](phase2-profiling-plan.md)) + batch sizes | Interference-aware policy |
| 4 | `main.py` / `router.py`: split execution path; logging (hop timestamps, bytes, deadline met, decision time); `only:<node>` | Split policy; experiment 5 runs |
| 5 | Traces (`tracegen.py`), analysis scripts | Experiment 5 analysis, report |

## 4. Cluster

All four machines are in us-west-2d (22 of the account's 24 vCPUs); access is set up once as in [cluster-access.md](cluster-access.md).

| Role | Instance | vCPU | SSH alias (on every machine) |
| --- | --- | --- | --- |
| Router and client | c7i.large | 2 | `router` |
| Node | c7a.2xlarge | 8 | `node-c7a` |
| Node | c7i.2xlarge | 8 | `node-c7i` |
| Node | m8g.xlarge | 4 | `node-m8g` |

- **Router hop:** the router reaches each node's llama-server through SSH tunnels (ports 18080–18082 → the node's 8080), inside the VPC, so the hop is under a millisecond rather than the internet round trip from a lab server.
- **KV copies go node to node** (`split.py --transport direct`), with the shared cluster key; the aliases work as both `ssh` and `peer` in the nodes file.
- **Network baseline:** `movement.py payload` router → each node (through the tunnel) and node → node.
- **Router CPU:** the router builds every request body (base64 images and audio) on 2 vCPUs; `main.py` now samples its CPU, to check it is not the bottleneck under bursts.

## 5. Scripts: who runs what, and what the routers read

Each experiment has its own script and writes its own results folder. The routing policies only read those results; they never re-measure.

| Experiment | Script | Runs on | Writes | What a router reads from it |
| --- | --- | --- | --- | --- |
| Node profiles | `benchmark.py` (ready) | each node | `results/phase1/<node>-p2__<model>/` | Time per workload alone (`router.load_costs`, as today) |
| 1. Stage breakdown | patched llama-server (`run.sh`); `benchmark_v3.py` | each node | `encoding_ms` in every tool's request rows; `results/phase2/<node>__<model>/requests_stages.csv`, `summary_stages.csv` | Encode, prompt-processing and decode time per workload |
| 2. Network | `movement.py payload` / `netdev` (ready) | router (+ `sink` on each node) | `results/movement/<path>/` | Round-trip time and MB/s per path |
| 3. Split | **`split.py`** (ready, tested on s41) | router | `results/split/<label>/alone.csv`, `split.csv`, `costs.json` | `split.SplitCosts(costs.json)`: `.alone(node, run)`, `.split(a, b, run, mb_s=None)`, `.options(run)` |
| 4. Interference | `corun.py pairs` + `summarize` | router | `results/interference/<node>__<model>/matrix.csv` | `corun.load_matrix(folder)`: slowdown(A \| B) |
| 5. Routing | `tracegen.py` (TTFT/TPOT deadlines), `main.py run` + `router.py` (**split and interference policies to add**); `analysis/phase2/analyze.py`, `figures.py` | router | `results/routing/`; analysis tables and the SLO grids | – (consumes the four above) |

**To carry out a split during a routing run,** a policy calls `split.split_request(src_url, dst_url, body, filename, copy, cleanup)`. It does the prefill on src, the save, copy and restore, and the answer on dst, and returns every step's time.

**Server setup:**
- `run.sh` starts llama-server with `--slot-save-path ~/slots` and takes `CACHE_RAM=` (`CACHE_RAM=0` for `split.py`, and on 8 GB nodes).
- `run.sh` builds llama-server with `patches/llama-stage-profiling.patch`, so responses carry `encoding_ms`.

## 6. Launch checklist (stages and split)

Do the access setup in [cluster-access.md](cluster-access.md) first, then on the router: tunnels on 18080–18082, venv, datasets (as in [phase2-profiling-plan.md](phase2-profiling-plan.md) §2.2, with the router in place of s41).

1. **On every node:** `git pull && CACHE_RAM=0 ./src/run.sh qwen`. This builds llama-server with the stage-profiling patch and restarts it with 2 slots and slot saving to `~/slots`. The same server serves every experiment (corun, split, routing).
2. **`nodes.aws.json` on the router:**
   ```json
   {
     "c7a.2xlarge": {"url": "http://127.0.0.1:18080", "ssh": "node-c7a", "peer": "node-c7a",
                     "profile": "c7a.2xlarge-p2__Qwen2.5-Omni-3B-Q8_0", "price_per_hour": 0.41},
     "c7i.2xlarge": {"url": "http://127.0.0.1:18081", "ssh": "node-c7i", "peer": "node-c7i",
                     "profile": "c7i.2xlarge__Qwen2.5-Omni-3B-Q8_0", "price_per_hour": 0.357},
     "m8g.xlarge":  {"url": "http://127.0.0.1:18082", "ssh": "node-m8g", "peer": "node-m8g",
                     "profile": "m8g.xlarge-p2__Qwen2.5-Omni-3B-Q8_0", "price_per_hour": 0.18}
   }
   ```
3. **Check the patch is live:** a request's `timings` has `encoding_ms` (empty for text).
4. **Quick check, ~5 min.** This tests the x86 → Arm KV file, the copy and the restore. Every line should have no WARNING, and `dst_encoding_ms` should be 0 (the restored image is not encoded again).
   ```bash
   python src/split.py run --nodes nodes.aws.json --label aws-check --runs 00 04 --repeats 1 \
     --pairs c7a.2xlarge:m8g.xlarge --transport direct
   ```
5. **Experiments**, one at a time on the cluster (claim it as in cluster-access.md):

   | Experiment | Where | Command | Time |
   | --- | --- | --- | --- |
   | 1. Stages | each node, Phase 1 style | `python src/benchmark_v3.py --node <type>-p2` (or from the encoding times of the runs below) | ~30 min per node |
   | 2. Network | router → each node, with `python src/movement.py sink` on the node and a tunnel to its port 8090; node → node the same way | `python src/movement.py payload --url http://127.0.0.1:<port> --label router-<node>` | ~5 min per path |
   | 3. Split | router | `python src/split.py run --nodes nodes.aws.json --label aws-p2 --runs 00 02 03 04 06 07 08 --transport direct --pairs c7a.2xlarge:m8g.xlarge c7a.2xlarge:c7i.2xlarge c7i.2xlarge:m8g.xlarge` | ~1.5 h |
   | 4. Interference | router | `python src/corun.py pairs --node <type>-p2 --url http://127.0.0.1:<port> --ssh node-<x>`, then `summarize` | ~20 min per node |

   Then `python src/split.py summarize results/split/aws-p2` writes `costs.json` for the routing policies.
