# Phase 1 Routing Benchmarks: Routing Multimodal Requests over Heterogeneous CPU Nodes

**Code:** `loadgen.py`. 

Step-by-step cluster setup: `docs/routing-cluster-runbook.md`. 

The baseline benchmarks it builds on: `docs/benchmark.md`.

## 1. Question

We initially benchmarked each modality on one node at a time to get baseline metrics. We now extend this to benchmark routing policies: a cluster of different CPU nodes serving a mix of requests that arrive like real traffic, at different loads. The question is:

> On a small cluster of CPU nodes with different strengths, under open-loop traffic, how much does a router gain by knowing what each request needs and what each node is good at, compared with a generic routing policy that does not look at the request?

We compare three routing policies on the same cluster and the same traffic, at a moderate and a high load.

## 2. Cluster

### 2.1 Configuration

|                              | **c7a.2xlarge**                 | **c7i.xlarge**                     | **m8g.xlarge**                |
| ---------------------------- | ------------------------------- | ---------------------------------- | ----------------------------- |
| CPU                          | AMD EPYC (Genoa)                | Intel Xeon (Sapphire Rapids), AMX  | AWS Graviton4 (Arm)           |
| vCPUs / physical cores / RAM | 8 / 8 / 16 GB                   | 4 / 2 / 8 GB                       | 4 / 4 / 16 GB                 |
| Slots                        | 2                               | 1                                  | 2                             |
| Prompt cache                 | default                         | off (`--cache-ram 0`)              | default                       |
| ~$/hour                      | 0.41                            | 0.18                               | 0.18                          |

Every node serves Qwen2.5-Omni-3B Q8_0 with the same llama.cpp build and 4,096 tokens per slot, so the answers are the same on every node and only speed differs. The cluster costs $0.77/hour.

### 2.2 Why these choices

**Node types.** We want one node from each CPU family that the baseline benchmarks measured as different (AMD, Intel with AMX, Graviton), so that the request type can matter for where a request should go.

**Sizes.** The AWS account allows 16 vCPUs running at once, so the cluster is 8 + 4 + 4 vCPUs. The 8-vCPU node is the c7a, because it was fastest on the most expensive requests (images, audio, long text).

**m8g instead of c8g.** Q8_0 needs more than the 8 GB of a c8g.xlarge (the c8g.2xlarge peaked at 10.4 GB). The m8g.xlarge has the same Graviton4 CPU with 16 GB.

**c7i.xlarge memory.** With 8 GB it needs two changes to stay up:
- one slot, because two 4-image requests at once ran it out of memory;
- the prompt cache off. llama-server keeps past prompts in RAM, up to 8 GB by default; on this node the cache grew until the server was killed during the 90% `least` run. That run and the 90% `ect` run were repeated with the cache off. Every request in a trace has a different input, so the cache never gets a hit and turning it off does not change speed.

### 2.3 Profiles

The router predicts request times from a profile of each node, measured on the same machines right before the routing runs with the baseline benchmark (`benchmark.py`): 2 requests per workload, alone (natural and fixed output) and two at once (fixed output). Results are in `results/phase1/<node>-p2__…`. The m8g.xlarge was not profiled two at once, to save time; its slot gain is taken from the c8g.2xlarge (same CPU).

Time per request alone (s), slot gain (throughput with 2 requests at once ÷ with 1), and capacity for the trace mix:

|                 | chat | text 3,072 tok | 4 images | 40 s audio | slot gain | capacity (req/s) |
| --------------- | ---- | -------------- | -------- | ---------- | --------- | ---------------- |
| c7a.2xlarge     | 10.9 | 31.9           | 28.5     | 27.9       | 1.39      | 0.088            |
| c7i.xlarge      | 17.5 | 64.0           | 76.4     | 65.9       | 1.00      | 0.031            |
| m8g.xlarge      | 5.9  | 126.7          | 99.7     | 110.8      | 1.01      | 0.029            |
| **cluster**     |      |                |          |            |           | **0.148**        |

## 3. Traffic

**Open loop.** Requests arrive on their own schedule, whether or not earlier ones have finished, like independent users. The baseline benchmarks were closed loop (a new request is sent when one finishes), which gives a baseline per modality and node. Here we add traffic patterns and vary the load, so queues build up as they would in a real deployment and placement decisions matter.

**Trace.**
- Arrivals: a Poisson process.
- Mix: 50% chat, 20% text (512 / 1,024 / 3,072 tokens), 15% image (1 / 2 / 4 photos), 15% audio (10 / 40 s); sizes equally likely within a type.
- Inputs: a different input per request, from the baseline datasets (Dolly, CNN/DailyMail, COCO, LibriSpeech). Natural output, greedy decoding, at most 128 output tokens.
- Duration: 15 minutes of arrivals.

The trace is generated once from a seed and saved, so **every policy replays exactly the same requests at the same times**.

**Two loads**, both seed 1: the same requests in the same order, only closer together at 90%.

| Trace                       | Target load | Rate (req/s) | Requests |
| --------------------------- | ----------- | ------------ | -------- |
| `traces/cluster-load60.csv` | 60%         | 0.089        | 85       |
| `traces/cluster-load90.csv` | 90%         | 0.134        | 122      |

Capacity is Σ over nodes of slot gain ÷ (mix-weighted mean time per request), from the profiles above.

## 4. Routing policies

Each policy picks a node **when the request arrives**. The request then goes to that node's llama-server and waits in its queue if all slots are busy. The router keeps no queue of its own.

| Policy  | Picks                                        | Knows request type? | Knows node speed? | Knows node load?     |
| ------- | -------------------------------------------- | ------------------- | ----------------- | -------------------- |
| `rr`    | nodes in turn                                | no                  | no                | no                   |
| `least` | fewest requests in flight per slot           | no                  | no                | yes (count only)     |
| `ect`   | earliest estimated completion time           | yes                 | yes               | yes (predicted work) |

**`rr` (round robin):** nodes in the order of `nodes.json`, one after another.

**`least` (least loaded):** for each node, the router counts the requests it has sent there that have not finished, and picks the lowest `in_flight / slots`, breaking ties at random. A chat request and a 4-image request count the same.

**`ect` (earliest estimated completion time):**
1. *Cost:* `cost[node][workload]` is the median time of that workload alone on that node, from the profiles. A workload missing on a node is estimated from other baseline results, scaled by how much faster or slower the node is on the workloads both have.
2. *Backlog:* for each node, the predicted seconds of work sent there and not yet finished. A request's cost is added when it is sent. While the node is busy, the backlog drops by `slot_gain[node]` seconds per second, and it is reset to 0 when the node has nothing in flight. `slot_gain` is the node's throughput with 2 requests at once ÷ with 1.
3. *Choice:* the node with the smallest `(backlog[node] + cost[node][request]) / slot_gain[node]`.

Example: a 4-image request arrives while the c7a still has about 20 s of work. `ect` predicts the c7a would finish it in about 20 + 28.5 = 48.5 s, the idle c7i in 76 s and the idle m8g in 100 s, so it sends it to the **busy** c7a. `least` would send it to an idle node. This is the decision the experiment tests.

`ect` uses only what is known when the request arrives and only measurements of requests running alone. It does not model interference between co-running requests beyond the average slot gain, or each request's output length beyond the median.

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
