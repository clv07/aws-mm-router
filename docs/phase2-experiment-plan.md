# Phase 2 Experiment Plan

**This is the one plan for Phase 2:** what we measure, why, who runs what, and in what order. The other Phase 2 docs are references for single scripts (see [the docs map](#docs-map)).

**Goal:** route multimodal requests (chat, text, image, audio) over a few different CPU nodes so that as many as possible meet their latency targets (SLOs), at the lowest cost. The headline result is a figure of **SLO attainment vs request rate**, one line per routing policy, one panel per modality, as in the EPD-disaggregation papers.

## 1. The logic

Phase 1 profiled each node with one request at a time and routed each request whole to one node. The router's cost model (`ect`) predicts a request's time from those solo profiles. Under real traffic it is missing five things, and each Phase 2 experiment measures one of them:

| What the router is missing | Why it matters for multimodal requests | Experiment |
| --- | --- | --- |
| **Where a request's time goes** (encode, prefill, decode) | Images and audio go through an encoder first. On s41 encoding was 52% of prompt time for images and 40% for audio; on a test node 64% for 4 images. Different CPUs are good at different stages: the c7a has the fastest prompts, the m8g decodes 2× faster | 1. Stages |
| **What moving data costs** | Requests carry base64 images and audio; a split moves the KV cache (~37 KB per prompt token, ~16 MB for one image) | 2. Network |
| **Whether a request should be split across nodes** | Encode + prefill on the node best at prompts, decode on the node best at tokens, KV cache moved in between (EP+D) | 3. Split |
| **What requests do to each other on one node** | llama-server encodes and prefills an image or audio clip in one call, which blocks the node's other slot: chat slowed 5.7× next to audio on s41, 1.8× on the c7a. `ect` assumes one fixed factor per node | 4. Interference, 5. Concurrency |
| **When to commit a request to a node** | `rr`/`least`/`ect` pick a node on arrival and send at once; extra requests wait in llama-server's queue, stuck there even if another node frees up first (a chat behind a 4-image request). A router queue sends a request only when a slot frees, so it can still choose the node, the order (deadline first) and the pairing. But the slot then sits idle while the router learns it is free, sends the request and the node receives and parses it | 6c. Where requests wait |

Experiments 1–5 measure each piece alone. Experiment 6 puts them into the router and compares policies under traffic, which produces the SLO figure; 6c separates *where requests wait* from *how a node is chosen*.

**What we split, and what we don't:** we split a request at the prefill → decode boundary (EP+D), using llama-server's slot save/restore. We don't split encoding from prefill (llama-server can't take embeddings as input) or the model's layers (only useful when a model doesn't fit). Details: [phase2-split.md](phase2-split.md).

**The SLO:** each request carries deadlines set when the trace is made (`tracegen.py`), per workload, from the fastest node running it alone: end-to-end (chat 3×, others 4×), TTFT (same factors) and TPOT (3×). A request meets its SLO when it meets both TTFT and TPOT; the end-to-end version is kept for the simulator, which models end-to-end time only.

## 2. Cluster and server setup

| Role | Instance | vCPU | SSH alias |
| --- | --- | --- | --- |
| Router and client | c7i.large | 2 | `router` |
| Node | c7a.2xlarge | 8 | `node-c7a` |
| Node | c7i.2xlarge | 8 | `node-c7i` |
| Node | m8g.xlarge | 4 | `node-m8g` |

- All four in us-west-2d; 22 of the account's 24 vCPUs, about $1.10/h. Access is set up once: [cluster-access.md](cluster-access.md).
- **One server setup serves every experiment:** on each node `CACHE_RAM=0 ./src/run.sh qwen`. That is llama-server built with `patches/llama-stage-profiling.patch` (every response reports `encoding_ms`), 2 slots, slot saving to `~/slots`, no prompt cache in RAM. Nobody restarts servers between experiments.
- The router reaches each node's llama-server through SSH tunnels: 18080 → c7a, 18081 → c7i, 18082 → m8g. Port 8080 is never opened.
- **One experiment at a time on the cluster.** Two runs at once distort each other's measurements. Claim the cluster as in cluster-access.md.

## 3. Experiments

| # | What we run | Question | What we quantify | Script | Owner | Cluster time |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | **Stages.** Every workload alone on every node, with the encoder timed inside llama-server | Is encoding the bottleneck? For which modalities, sizes and CPUs? | Encode, prefill and decode s per request; encode share of prompt and of total time | Patched llama-server; comes from experiment 3's alone phase (and from every other run) | Stage profiling | none extra |
| **2** | **Network.** Upload/download 1 KB–128 MB router → each node (through the tunnel) and node → node | What does moving requests, tokens and KV cost next to compute? | Round trip (ms) and MB/s per path; bytes per request | `movement.py payload` (+ `sink` on the far end), `netdev` | Split & network | ~20 min |
| **3** | **Split vs no split,** one request at a time: each workload alone on each node, then split A → B over c7a → m8g, c7a → c7i, c7i → m8g | When does splitting pay? For which modalities, pairs and link speeds? | Time of every option; KV MB; save/copy/restore s; break-even bandwidth | `split.py run` / `summarize` → `costs.json` | Split & network | ~1.5 h (+5 min quick check first) |
| **4** | **Interference.** Each workload alone, then every pair, one per slot, on the c7a and m8g (base sizes: chat, 1,024-token text, 1 image, 10 s audio; then chat × largest sizes) | How much does sharing a node slow each request, and which stage suffers? | Slowdown matrix (end-to-end, server-side, encode, prefill, decode), stalls, pair gain, peak RAM | `corun.py pairs` / `summarize` → `matrix.csv`, `load_matrix()` | Interference & concurrency | c7a ~45 min + ~25 min, m8g ~60 min |
| **5** | **Concurrency.** k = 1–4 streams with mixed workloads, including more streams than slots | How does cost grow with load, and does the pair matrix predict mixed load? | Cost vs k per node, queue wait, measured gain per node | `corun.py groups` | Interference & concurrency | ~20–30 min per node |
| **6a** | **Bursty replay.** A bursty trace (`onoff`, 70% load, 15 min) replayed live with `ect` and `least` | Does the simulator match reality under bursts? | p50/p95 per modality, live vs simulated | `tracegen.py`, `main.py run` / `sim` | Analysis | ~40 min |
| **6b** | **Rate sweep.** Traces at ~6 loads × 2–3 seeds, every policy: `rr`, `least`, `ect`, `slo`, then `ect` + interference and `ect` + split | Which policy sustains the highest rate at ≥ 90% SLO attainment? | SLO attainment (TTFT + TPOT, and end-to-end), goodput, $ per 1,000 met requests, latency breakdown, router CPU, bytes moved | `main.py sim` for the full curves; `main.py run` at 2–3 loads per policy; `analysis/phase2/analyze.py`, `figures.py` | Analysis + routing algorithms | ~3–4 h live |
| **6c** | **Where requests wait: dispatch on arrival vs router queue.** The same node choice (`ect`'s cost model) run two ways: commit on arrival (wait in llama-server) and late binding (wait at the router, dispatch when a slot frees); then the router queue's order: arrival (FIFO) vs least slack (deadline first), with aging. Over the 6b loads, steady and bursty traces | When does holding requests at the router help, and when does it hurt? | SLO attainment and p95 per modality; where the time went (`router_wait_s` vs node queue); slot utilization (the idle gap between requests); requests stuck behind a large one; router CPU and decision time | `router.py` queue path (`slo` has it: `order` = slack / sufferage / fifo, `wait_for_best`, aging); `main.py sim`. **To add:** an `ect`-cost queue variant and command-line options for the order, so only one thing changes at a time; live router-queue dispatch in `main.py run` | Routing algorithms + analysis | sim: none; live: 2 loads × 2 traces × 2 modes, ~2 h |
| *7 (optional)* | *Multi-turn sessions:* keep a session on the node holding its KV, move the KV, or recompute | When should a session move? | Bytes and latency per turn | to add | – | – |

**What to expect from 6c.** At low load nothing queues, so a router queue only adds its decision time. It should help under bursts, with mixed request sizes and when predictions are off, because a waiting request can still go to whichever node frees first. It costs an idle gap per request (the router-node round trip plus uploading the body; a few ms inside AWS, much more from outside or for large audio), and a 2-vCPU router has to keep up. A middle option worth testing if the gap matters: let each node hold one waiting request, so the upload overlaps the running one.

Why 70% load for the bursty replay: bursts already reach about 2× capacity, and at 90% the `rr`/`least` backlogs take too long to drain.

## 4. How the results feed the router

The routing policies only read these results; they never re-measure.

| Result | File | Read by | What changes in the router |
| --- | --- | --- | --- |
| Solo time per workload and node | `results/phase1/<node>-p2__<model>/` | `router.load_costs` | (as in Phase 1) |
| Stage times | `encoding_ms` in every request row; `alone.csv` of experiment 3 | – | Encode vs prefill per node, for choosing where a prompt runs |
| Network | `results/movement/<path>/payload.csv` | `analyze.py --link` | Dispatch delay α + β·MB added to each node's predicted time |
| Split | `results/split/<label>/costs.json` | `split.SplitCosts` | `ect` + split: the finish time of every option (each node alone, each split A → B), pick the earliest; carry it out with `split.split_request()` |
| Interference | `results/interference/<node>__<model>/matrix.csv` | `corun.load_matrix()` | `ect` + interference: slowdown(new \| running) in place of the fixed `k / slot_gain` in `cost_model` and in the simulator's `rate()`; prefer the node whose current work pairs best |
| Concurrency | `results/concurrency/<node>__<model>/` | – | Measured gain per node (replaces the m8g's borrowed 1.01) |
| Where requests wait | 6c analysis | – | Whether policies commit on arrival or queue at the router, and in what order; above which load the router queue is worth it |

## 5. Who does what

| Role | Owns | Experiments |
| --- | --- | --- |
| **Split & network** | `split.py`, `movement.py`, `run.sh`, cluster bring-up | 2, 3; quick check |
| **Interference & concurrency** | `corun.py` | 4, 5 |
| **Stage profiling** | `patches/llama-stage-profiling.patch` | 1 (reads the encoding times from 3, 4 and 5) |
| **Analysis** | `analysis/phase2/`, `tracegen.py` traces | 6a, 6b analysis and figures |
| **Routing algorithms** (two people, one algorithm each) | `router.py`, `main.py` | `ect` + interference, `ect` + split; live router-queue dispatch; 6b, 6c runs |
| **Router machine** | the c7i.large router | access setup |

## 6. Order

| # | Step | Role | After |
| --- | --- | --- | --- |
| 1 | Merge the measurement PR (split, network, patch build in `run.sh`, corun analysis, analysis add-ons) | Split & network; reviewers: the owners of each file | – |
| 2 | Cluster access setup, once | Router machine + node owners; everyone's keys | – |
| 3 | Bring-up: patched servers on all nodes, tunnels and venv on the router, check `encoding_ms` | Split & network | 1, 2 |
| 4 | Split quick check (x86 → Arm restore) | Split & network | 3 |
| 5 | Network sweep (exp. 2), then split run (exp. 3) | Split & network | 4 |
| 6 | Interference and concurrency (exp. 4, 5) | Interference & concurrency | 3 |
| 7 | Stage results (exp. 1) from 5 and 6 | Stage profiling | 5, 6 |
| 8 | Traces; bursty replay (6a); simulator calibration | Analysis | 3 |
| 9 | `ect` + interference and `ect` + split policies, tested on a short trace | Routing algorithms | 5, 6 |
| 10 | Rate sweep (6b), SLO grids | Analysis + routing algorithms | 8, 9 |
| 11 | Where requests wait (6c): simulator first, then live at 2 loads, steady and bursty | Routing algorithms + analysis | 9 (live dispatch in `main.py run`), 10 |

Steps 4–8 all use the cluster: run them one after another, not at the same time.

## 7. Commands

**Bring-up** (after [cluster-access.md](cluster-access.md)):
```bash
# each node
cd ~/aws-mm-router && git pull && CACHE_RAM=0 ./src/run.sh qwen
# router: code, venv, datasets (copied from a node inside AWS), tunnels
git clone https://github.com/clv07/aws-mm-router ~/<name>/aws-mm-router && cd ~/<name>/aws-mm-router
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
rsync -a node-c7a:aws-mm-router/dataset/ dataset/
tmux new -d -s tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 18080:127.0.0.1:8080 node-c7a; sleep 3; done'
tmux split-window -t tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 18081:127.0.0.1:8080 node-c7i; sleep 3; done'
tmux split-window -t tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 18082:127.0.0.1:8080 node-m8g; sleep 3; done'
for p in 18080 18081 18082; do curl -s http://127.0.0.1:$p/props | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("total_slots"), d.get("build_info"))'; done
```

`nodes.aws.json` on the router:
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

**Experiments**, on the router in tmux:
```bash
# 3. split: quick check (no WARNING lines; dst_prompt_n 1, dst_encoding_ms 0), then the full run
python src/split.py run --nodes nodes.aws.json --label aws-check --runs 00 04 --repeats 1 \
  --pairs c7a.2xlarge:m8g.xlarge --transport direct
python src/split.py run --nodes nodes.aws.json --label aws-p2 --runs 00 02 03 04 06 07 08 --transport direct \
  --pairs c7a.2xlarge:m8g.xlarge c7a.2xlarge:c7i.2xlarge c7i.2xlarge:m8g.xlarge
python src/split.py summarize results/split/aws-p2

# 2. network: `python src/movement.py sink` on the node, a tunnel router:18090 -> node:8090, then
python src/movement.py payload --url http://127.0.0.1:18090 --label router-c7a

# 4. interference (c7a shown; m8g with 18082 / node-m8g and --repeats 4)
python src/corun.py pairs --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a \
  --runs 00 02 04 07 --repeats 5 --pool 60
python src/corun.py pairs --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a \
  --runs 00 03 06 08 --pairs 00+03 00+06 00+08 03+06 03+08 06+08 --repeats 3 --pool 40
python src/corun.py summarize results/interference/c7a.2xlarge-p2__Qwen2.5-Omni-3B-Q8_0

# 5. concurrency
python src/corun.py groups --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a --mix default \
  --repeats 4 --allow-queue --groups 00+mix 00+mix+mix 00+mix+mix+mix mix+mix mix+mix+mix mix+mix+mix+mix

# 6a. bursty replay
python src/tracegen.py make --arrivals onoff --load 0.7 --duration 900 --seed 1 --nodes nodes.aws.json \
  --burst-factor 3 --burst-fraction 0.2 --burst-length 60
for p in ect least; do python src/main.py run --trace traces/phase2/<that trace>.csv --nodes nodes.aws.json --policy $p --cluster aws-p2; done
python src/main.py sim --trace traces/phase2/<that trace>.csv --nodes nodes.aws.json

# 6b. rate sweep: one trace per load and seed, simulated for every policy, live for 2-3 loads
for load in 0.4 0.6 0.8 1.0 1.2 1.4; do for seed in 1 2; do
  python src/tracegen.py make --load $load --seed $seed --duration 900 --nodes nodes.aws.json; done; done

# 6c. where requests wait: today `slo` (router queue, least slack) vs `ect` (commit on arrival) in the simulator,
#     on every 6b trace; the ect-cost queue variant and the order options are still to add
python src/main.py sim --trace traces/phase2/<trace>.csv --nodes nodes.aws.json --policies ect,slo

# analysis and figures (--link from experiment 2: round trip s, MB/s)
python analysis/phase2/analyze.py --nodes nodes.aws.json --out results/phase2-analysis --link <RTT_S> <MB_S>
python analysis/phase2/figures.py results/phase2-analysis      # slo-grid-*.png, breakdown, placement
```

## 8. Done when

- [ ] Split: `costs.json` for the three pairs, no warnings, x86 → Arm restore working.
- [ ] Network: round trip and MB/s for router → each node and node → node.
- [ ] Interference: c7a and m8g `matrix.csv` with ≥ 5 requests per base cell (≥ 3 for the largest sizes), no failures, solo times close to Phase 1.
- [ ] Concurrency: cost vs k = 1–4 for the c7a and m8g.
- [ ] Stages: encode share per workload and node, alone and under interference.
- [ ] Simulator calibrated: with the interference matrix, simulated p50/p95 for `ect`/`least` on the bursty trace within ~20–25% of live.
- [ ] SLO grids: attainment vs rate for every policy, TTFT + TPOT and end-to-end.
- [ ] Where requests wait: commit-on-arrival vs router queue (FIFO and least slack) on the same node choice, simulated over all loads and live at 2 loads, steady and bursty; the load above which the router queue wins.
- [ ] Results committed, instances stopped.

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| x86 → Arm KV restore fails | Split only within one CPU family (c7a ↔ c7i); report it as a finding |
| Router queue's idle gap or decision time outweighs its benefit | Measure it in 6c (slot utilization, `decision_us`); try one waiting request per node |
| Router CPU becomes the bottleneck under bursts (2 vCPUs build every request) | Check `router_cpu_max` in the analysis; move the router to a larger instance if it saturates |
| Tunnels drop | They restart themselves; failed requests are excluded; rerun only what failed |
| Quota (24 vCPUs shared by the whole account) | Nobody starts other instances while the cluster runs |
| Cost | `sudo shutdown -h +360` on every node after starting; stop all four when done |

## Docs map

| Doc | What it is | Phase |
| --- | --- | --- |
| **phase2-experiment-plan.md** | This plan | 2 |
| [cluster-access.md](cluster-access.md) | Shared cluster access, set up once | 2 |
| [phase2-split.md](phase2-split.md) | How the EP+D split works; `split.py` | 2 |
| [phase2-interference.md](phase2-interference.md) | Pairwise interference; `corun.py pairs` and `summarize` | 2 |
| [phase2-traffic-generation.md](phase2-traffic-generation.md) | Traces and deadlines; `tracegen.py` | 2 |
| [baseline-benchmark.md](baseline-benchmark.md) | Phase 1 per-node profiles (`benchmark.py`) | 1 |
| [routing-benchmark.md](routing-benchmark.md) | Phase 1 routing policies and results | 1 |
| [routing-cluster-runbook.md](routing-cluster-runbook.md) | Phase 1 cluster setup | 1 |
| [gpu-benchmark.md](gpu-benchmark.md) | GPU comparison node | 1 |
| [routing-algorithm1.pdf](routing-algorithm1.pdf) | Routing algorithm write-up | – |
