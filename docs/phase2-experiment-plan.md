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
| **1** | **Stage breakdown.** Each workload alone on each node (chat; text 512 / 1,024 / 3,072 tokens; 1 / 2 / 4 images; 10 / 40 s audio), timing encoding, prompt processing and decoding separately | Where a request's time goes, per modality and per CPU | **Is encoding the bottleneck?** For which modalities, sizes and nodes? Which node is fastest at each stage? | Encode s, prompt-processing s and decode s per request. Share of total time spent encoding. Tokens per image / audio second. Speed per stage per node (tokens/s; images or audio-seconds/s) | `stages.py` (wraps `llama-mtmd-cli`, which logs encoding and prompt processing separately; tested on s41); chat and text from the `benchmark.py` profiles |
| **2** | **Data movement.** Network round-trip and bandwidth between every pair of machines, through the router's tunnels; request and response bytes per modality; KV save, copy and restore per modality | The cost of every byte the system moves | How much does moving requests, streamed tokens and KV cost, next to compute? Does a KV file saved on x86 load on Arm? | Round-trip ms, MB/s per path. Bytes up/down per request. KV MB per request (≈36 KB per prompt token). Save / copy / restore s | `ping`, `iperf3`, `movement.py payload`; KV: `split.py` |
| **3** | **Split vs no split, one request at a time.** Each workload three ways: all on node A, all on node B, split A → B (c7a → m8g, c7a → c7i, c7i → m8g). Repeated with the link limited to 1 Gbps, 100 Mbps and 20 Mbps (`tc`) | Whether running each stage on its best node beats the KV-move cost | **When does splitting pay?** For which modalities, node pairs and link speeds? | Latency of each option, time saved by splitting, break-even bandwidth per modality, answer agreement | `split.py` (tested on s41), `tc` |
| **4** | **Interference.** Every pair of workloads together on each 2-slot node; the prompt batch size (`-b`/`-ub` 2,048 → 128) for chat paired with text, image and audio | What a request costs another request on the same node | How much does co-location slow each request? Can configuration stop chat from stalling, or must the router avoid it? Is the encoder the cause? | Slowdown matrix per node, longest pause in chat's output, pair gain, per batch size | `interference.py` (tested on s41) |
| **5** | **Routing under traffic.** Traces (steady and bursty, 60% and 90% load, default mix + held-out mixes) replayed with each policy: each node alone, `rr`, `least`, `ect`, `ect` + interference, `ect` + split | Whether the knowledge from 1–4 improves routing when requests overlap | How much does each added piece of knowledge improve tail latency and deadlines met? What does splitting cost in network traffic? | p50/p95/p99 latency per modality, % of deadlines met, throughput, $ per 1,000 requests, bytes moved per policy (requests, streams, KV), router decision time | `tracegen.py`, `main.py run` with `router.py` policies (**split execution, new policies and the extra logging: to add**), `movement.py netdev` |
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
  - First direct measurement, `stages.py` on s41 with 8 cores: **encoding is 52–53% of prompt time for 1, 2 and 4 images, and 39–43% for 10 s and 40 s audio** (28–41% of the whole request). The rest of the prompt time is the language model processing the media's tokens (~480 per image, 75 per audio second).
- **In our llama-server build, encoding can't be chunked.** An image or audio chunk is encoded and then fully prompt-processed in one call, which blocks the node's other slot. This is the likely cause of chat's long pauses next to images and audio.
- **Experiment 1 measures:**
  - the encoding share for every modality, size and node;
  - how encoding grows with images per request and audio length;
  - which CPU encodes fastest (Phase 1 suggested the c7a's 8 physical cores, by more than 2×).

## 3. Build-up: stages over two weeks

| Stage | When | What | Output | Decision at the end |
| --- | --- | --- | --- | --- |
| **0. Develop on s41** | Days 1–2 (no AWS cost) | Write the stage-breakdown script; add split execution and logging to `main.py` / `router.py`; dry-run experiments 1, 3 and 4 on s41 | Working scripts; rough numbers | – |
| **1. Bring up the cluster** | Day 3 | Start the nodes and the s41 router (§4 and the profiling plan §2); network baseline (experiment 2, first half) | Network numbers | – |
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

**Router and client:** s41, as in [phase2-profiling-plan.md](phase2-profiling-plan.md) §2. It reaches the three nodes through SSH tunnels.

**Serving nodes:** c7a.2xlarge, c7i.xlarge and m8g.xlarge (16 vCPUs), as in Phase 1. The 8 spare vCPUs could later go to a c7i.2xlarge (in place of the 8 GB c7i.xlarge) or a second m8g.

**What a router outside AWS means for the measurements:**
- **Router hop:** every request and its token stream cross the internet (UCSD → us-west-2, ~20–40 ms round trip). That's the "client → cluster" hop, and it's the same for every node, so the c7a no longer has the advantage of sharing a machine with the router.
- **KV copies must go node to node inside AWS** (`split.py --transport direct`). A copy relayed through s41 would cross the internet twice and measure the wrong link.
  - Direct copies need each source node to SSH to the destination by its private IP. Create a key on the c7a (and the c7i), and add its public key to the other nodes' `~/.ssh/authorized_keys`.
  - The security group already allows SSH between its own members.
- **Network baseline:** measure both kinds of path: s41 → each node through the tunnel (`movement.py payload`) and node → node (`iperf3`, or `movement.py payload` run on a node).

**Server settings for the split:**
- llama-server needs `--slot-save-path`; `run.sh` now passes `~/slots`.
- The c7i's hand-started server (profiling plan §2.1) needs `--slot-save-path ~/slots/` added to its command, or it can't be part of a split.

## 5. Scripts: who runs what, and what the routers read

Each experiment has its own script and writes its own results folder. The routing policies only read those results; they never re-measure.

| Experiment | Script | Runs on | Writes | What a router reads from it |
| --- | --- | --- | --- | --- |
| Node profiles | `benchmark.py` (ready) | each node | `results/phase1/<node>-p2__<model>/` | Time per workload alone (`router.load_costs`, as today) |
| 1. Stage breakdown | `stages.py` (ready, tested on s41; wraps `llama-mtmd-cli`) | each node | `results/stages/<node>__<model>/stages.csv`, `summary.csv` | Encode, prompt-processing and decode time per image/audio workload |
| 2. Network | `movement.py payload` / `netdev` (ready) | router (+ `sink` on each node) | `results/movement/<path>/` | Round-trip time and MB/s per path |
| 3. Split | **`split.py`** (ready, tested on s41) | router | `results/split/<label>/alone.csv`, `split.csv`, `costs.json` | `split.SplitCosts(costs.json)`: `.alone(node, run)`, `.split(a, b, run, mb_s=None)`, `.options(run)` |
| 4. Interference | `interference.py` (ready) | each node | `results/interference/<node>__<model>/matrix.csv` | `interference.load_matrix(folder)`: slowdown(A \| B) |
| 5. Routing | `tracegen.py` (ready), `main.py run` + `router.py` (**policies and logging to add**) | router | `results/routing/` | – (consumes the four above) |

**To carry out a split during a routing run,** a policy calls `split.split_request(src_url, dst_url, body, filename, copy, cleanup)`. It does the prefill on src, the save, copy and restore, and the answer on dst, and returns every step's time.

**Server setup:**
- `run.sh` starts llama-server with `--slot-save-path ~/slots` and takes `CACHE_RAM=` (`CACHE_RAM=0` for `split.py`, and on 8 GB nodes).
- `run.sh` now also builds `llama-mtmd-cli` (for `stages.py`).

## 6. Launch checklist (stages and split, on top of the profiling plan's setup)

Do the setup in [phase2-profiling-plan.md](phase2-profiling-plan.md) §2 first: nodes, s41 SSH config, tunnels on 18080–18082, venv, datasets. Then:

1. **On every node:** `git pull && CACHE_RAM=0 ./src/run.sh qwen`. This builds `llama-mtmd-cli` and restarts llama-server with slot saving to `~/slots`.
   - On the c7i, add `--slot-save-path ~/slots/` to its hand-started command.
   - Don't use `CACHE_RAM=0` for the profiling plan's runs if they need the prompt cache. It makes no difference there, since every input is different.
2. **Node-to-node SSH, for direct copies:**
   - On the c7a: `ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519`.
   - Append its `.pub` to `~/.ssh/authorized_keys` on the c7i and m8g.
   - Check: `ssh ubuntu@<M8G_PRIVATE_IP> hostname`.
3. **`nodes.s41.json`:** add `"peer"` (how the other nodes reach this one) to each node, e.g. `"peer": "ubuntu@<M8G_PRIVATE_IP>"`. `split.py` reads `url`, `ssh` and `peer`, and ignores the rest.
4. **Quick check, ~5 min.** This tests the x86 → Arm KV file, the copy and the restore. Every line should have no WARNING.
   ```bash
   python src/split.py run --nodes nodes.s41.json --label aws-check --runs 00 04 --repeats 1 \
     --pairs c7a.2xlarge:m8g.xlarge --transport direct
   ```
5. **Experiments:**

   | Experiment | Where | Command | Time |
   | --- | --- | --- | --- |
   | 1. Stages | on each node, llama-server idle (not during the profiling plan's lanes) | `python src/stages.py run --node <type>-p2` | ~20–40 min |
   | 2. Network | s41 → each node, with `python src/movement.py sink` running on the node and a tunnel to its port 8090 | `python src/movement.py payload --url http://127.0.0.1:<port> --label s41-<node>` | ~5 min per path |
   | 3. Split | s41 | `python src/split.py run --nodes nodes.s41.json --label aws-p2 --runs 00 02 03 04 06 07 08 --transport direct --pairs c7a.2xlarge:m8g.xlarge c7a.2xlarge:c7i.xlarge c7i.xlarge:m8g.xlarge` | ~1.5 h |

   Then `python src/split.py summarize results/split/aws-p2` writes `costs.json` for the routing policies.

   **Splits and the profiling plan's lanes both use the nodes,** so run them at different times. Otherwise each disturbs the other's measurements.
