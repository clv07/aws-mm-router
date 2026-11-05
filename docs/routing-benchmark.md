# Phase 1 Routing Benchmarks: Routing Multimodal Requests over Heterogeneous CPU Nodes

**Code:** `loadgen.py`. 

Step-by-step cluster setup: `docs/routing-cluster-runbook.md`. 

The baseline benchmarks it builds on: `docs/baseline-benchmark.md`.

## 1. Motivation and question

The baseline benchmarks (`docs/baseline-benchmark.md`) ran each modality on one node at a time. Their key takeaway is that **no CPU architecture is fastest for every modality**. One request alone on 8-vCPU nodes:

| Modality              | Fastest                         | Others        |
| --------------------- | ------------------------------- | ------------- |
| Chat                  | Graviton c8g: 3.1 s             | 9.3–11 s      |
| Text, 3,072 tokens    | Intel c7i (AMX) and AMD c7a: 33–34 s | 59–66 s  |
| 4 images              | AMD c7a: 31 s                   | 42–61 s       |
| 40 s audio            | AMD c7a: 28 s                   | 35–63 s       |

This follows from where each modality spends its time. Chat is mostly decode (generating the answer), which ran fastest on Graviton. Long text is mostly prefill (processing the prompt), which ran fastest with AMX or more physical cores. Images and audio add an encoding step before prefill, which ran more than twice as fast on the AMD node's 8 physical cores as on any other node.

We route whole requests, and a request's modality is known when it arrives. So the question is:

> On a small cluster of CPU nodes with different strengths, under open-loop traffic, how much does a router gain by knowing each request's modality and what each node is good at, compared with a generic routing policy that does not look at the request?

We compare three routing policies on the same cluster and the same traffic, at a moderate and a high load.

## 2. Cluster

|                              | **c7a.2xlarge**                 | **c7i.xlarge**                     | **m8g.xlarge**                |
| ---------------------------- | ------------------------------- | ---------------------------------- | ----------------------------- |
| CPU                          | AMD EPYC (Genoa)                | Intel Xeon (Sapphire Rapids), AMX  | AWS Graviton4 (Arm)           |
| vCPUs / physical cores / RAM | 8 / 8 / 16 GB                   | 4 / 2 / 8 GB                       | 4 / 4 / 16 GB                 |
| Fastest for (baseline)       | images, audio, long text        | long text                          | chat                          |
| Slots                        | 2                               | 1                                  | 2                             |
| Prompt cache                 | default                         | off (`--cache-ram 0`)              | default                       |
| ~$/hour                      | 0.41                            | 0.18                               | 0.18                          |

Every node serves Qwen2.5-Omni-3B Q8_0 with the same llama.cpp build and 4,096 tokens per slot, so answers are the same on every node and only speed differs.

Why this cluster:
- **One node per CPU family** from the baseline, so each modality has a different best node.
- **16 vCPUs in total** (the AWS account limit), split 8 + 4 + 4; the 8 vCPUs go to the c7a, fastest on the most expensive requests.
- **Cost:** $0.77/hour for the cluster, to fit our AWS credits.
- **m8g instead of c8g:** same Graviton4 CPU, but 16 GB; Q8_0 needs more than a c8g.xlarge's 8 GB.
- **One region (us-west-2):** network delay between nodes is negligible, so only compute differs.
- **c7i.xlarge, 1 slot:** with 8 GB, two 4-image requests at once ran it out of memory.
- **c7i.xlarge, prompt cache off:** llama-server's default 8 GB prompt cache ran it out of memory during the 90% `least` run; that run and the 90% `ect` run were repeated with it off. Every input is different, so the cache never hits and turning it off does not change speed.

## 3. Traffic

**Open loop.** Requests arrive on their own schedule, whether or not earlier ones have finished, like independent users. The baseline benchmarks were closed loop (a new request is sent when one finishes), which gives a baseline per modality and node. Here we add traffic patterns and vary the load, so queues build up as they would in a real deployment and placement decisions matter.

**Trace.**
- Arrivals: a Poisson process.
- Mix: 50% chat, 20% text (512 / 1,024 / 3,072 tokens), 15% image (1 / 2 / 4 photos), 15% audio (10 / 40 s); sizes equally likely within a type.
- Inputs: a different input per request, from the baseline datasets (Dolly, CNN/DailyMail, COCO, LibriSpeech). Natural output, greedy decoding, at most 128 output tokens.
- Duration: 15 minutes of arrivals.

The trace is generated once from a seed and saved, so **every policy replays exactly the same requests at the same times**.

**Two loads**, as a share of the cluster's predicted capacity (0.148 req/s for this mix, from the node profiles in section 4). Both use seed 1: the same requests in the same order, only closer together at 90%.

| Trace                       | Target load | Rate (req/s) | Requests |
| --------------------------- | ----------- | ------------ | -------- |
| `traces/cluster-load60.csv` | 60%         | 0.089        | 85       |
| `traces/cluster-load90.csv` | 90%         | 0.134        | 122      |

## 4. Routing policies

We compare three well-known routing approaches. They differ in how much each one knows about the request when it picks a node, so comparing them in order shows what each extra piece of information is worth on a cluster of different CPUs.

Each policy picks a node **when the request arrives** (immediate dispatch). The request then goes to that node's llama-server and waits in its queue if all slots are busy. The router keeps no queue of its own.

| Policy  | Known as                                         | Picks                                | Knows node load?     | Knows modality and node speed? |
| ------- | ------------------------------------------------ | ------------------------------------ | -------------------- | ------------------------------ |
| `rr`    | Round robin                                      | nodes in turn                        | no                   | no                             |
| `least` | Least loaded / join the shortest queue           | fewest requests in flight per slot   | yes (count only)     | no                             |
| `ect`   | Minimum completion time (MCT)                    | earliest estimated completion time   | yes (predicted work) | yes                            |

**`rr` (round robin):** nodes in the order of `nodes.json`, one after another. The default in common load balancers (e.g. nginx, Envoy). It uses no information, so every node gets the same share of every modality, however fast or busy it is.

**`least` (least loaded):** picks the node with the fewest unfinished requests per slot, breaking ties at random. This is the classic join-the-shortest-queue rule (Envoy's "least request", nginx's `least_conn`). It reacts to load but treats every request as equal work and every node as equally fast.

**`ect` (earliest estimated completion time):** the minimum completion time (MCT) heuristic from scheduling on heterogeneous computers (Maheswaran et al., JPDC 1999; Braun et al., JPDC 2001): send each request to the node where it is predicted to finish first.
- *Predicted time per request:* the median time of each workload alone on each node, from profiling the cluster nodes right before the runs (2 requests per workload with `benchmark.py`; results in `results/phase1/<node>-p2__…`). For example, chat / 4 images: c7a 10.9 / 28.5 s, c7i 17.5 / 76.4 s, m8g 5.9 / 99.7 s.
- *Backlog:* the predicted work already sent to each node and not finished, draining at the node's slot gain (throughput with 2 requests at once ÷ with 1: c7a 1.39; c7i 1.00 with one slot; m8g 1.01, taken from the c8g.2xlarge to save profiling time).
- *Choice:* the node with the smallest `(backlog + predicted time) / slot gain`.

Example: a 4-image request arrives while the c7a still has about 20 s of work. `ect` predicts the c7a would finish it in about 48.5 s, the idle c7i in 76 s and the idle m8g in 100 s, so it sends it to the **busy** c7a, where `least` would pick an idle node. `ect` does not model interference between co-running requests beyond the average slot gain.

**What each comparison isolates.**
- **`rr` vs `least`: the value of knowing node load.** Both ignore what the request is; `least` adds a live count of work in flight.
- **`least` vs `ect`: the value of knowing the modality and node speed.** Both react to load; `ect` also knows that chat is cheap on the m8g and images on the c7a, and weighs queued work by how long it will take, not by how many requests it is.
- **What each one needs to run.** `rr` needs nothing; `least` needs a count of requests in flight; `ect` also needs a profile of every node and modality, redone when the hardware, model or server settings change. The comparison shows whether that extra effort pays off.

## 5. Metrics

Per request (`results/routing/requests.csv`): arrival and send time, chosen node, requests already on it, `ect`'s predicted latency (logged for every policy), end-to-end latency, time to first token (TTFT), queue wait, prompt and output tokens, decode speed, status.

Per run (`results/routing/summary.csv`), for all requests, each request type and each node: median / p95 / p99 end-to-end latency, median / p95 TTFT, median queue wait, throughput, cost per 1,000 requests, CPU and peak RAM per node.

**Primary comparison:** p95 end-to-end latency per request type, per policy, at each load, and cost per 1,000 requests.
**Secondary:** median latency, where each policy sends each request type, and how accurate `ect`'s predictions are.

<!-- Hypotheses: to be added back after the analysis.
## Hypotheses

1. **H1:** at 60% load all policies have similar median latency, but `rr` already has a worse p95 for heavy requests, because a third of them land on the m8g.
2. **H2:** at 90% load the gap widens. `least` beats `rr` but still sends heavy requests to the m8g whenever it is idle. `ect` has the lowest p95 for images, audio and long text.
3. **H3:** `ect` gives chat the lowest latency by sending most chat to the m8g, without making heavy requests on the c7a worse.
4. **H4:** `ect`'s prediction error grows with load, because interference between co-running requests is not in its model.
-->

## 6. Running it

The cluster setup (instances, llama-server, SSH tunnels, profiling) is in `docs/routing-cluster-runbook.md`. The router runs on the c7a, in the same region as the nodes, and reaches them through SSH tunnels over private IPs; llama-server listens on localhost only. Run in tmux.

**Nodes.** Copy `nodes.example.json` to `nodes.json`: one entry per node with its `url`, `profile` (its folder in `results/phase1`), `price_per_hour`, and optionally `ssh` (to sample CPU and RAM) and `slot_gain` (overrides the measured one). Entry order is the round-robin order.

**Capacity and traces.**

```bash
python loadgen.py capacity --nodes nodes.json        # predicted time per workload and node, capacity
python loadgen.py trace --rate 0.089 --duration 900 --seed 1 --out traces/cluster-load60.csv
python loadgen.py trace --rate 0.134 --duration 900 --seed 1 --out traces/cluster-load90.csv
```

`--mix` changes the mix (default `chat=50,text=20,image=15,audio=15`); a key is a modality or a workload id, e.g. `03=10` for 3,072-token text.

**Runs.**

```bash
python loadgen.py run --trace traces/cluster-load60.csv --nodes nodes.json --policy ect --cluster c7a-c7i-m8g --requests 10   # quick test
for load in 60 90; do
  for p in rr least ect; do
    python loadgen.py run --trace traces/cluster-load$load.csv --nodes nodes.json --policy $p --cluster c7a-c7i-m8g
  done
done
```

Each run warms up every node (one request per modality, not recorded), replays the trace, waits for the last request and prints a summary; it takes about 17 minutes. Results are appended to `results/routing/requests.csv` and `summary.csv`, with a `run_id` per run. `--output fixed` makes every request generate 128 tokens; `--seed` sets tie-breaking.

**Time and cost:** about 2 hours for the 6 runs plus about 1 hour of profiling, at $0.77/hour for the cluster.
