# Phase 2: Routing Multimodal Requests over Heterogeneous CPU Nodes

**Status:** design, for team review before the runs. Results will be added to this document.
**Code:** `loadgen.py` (PR #6), setup in `docs/benchmark.md` section 8, step-by-step procedure in `docs/phase2-cluster-runbook.md`.

## 1. Question

Phase 1 showed that CPU instance types differ in *which kind of request* they are fast at, not just in overall speed. Phase 2 asks:

> On a small cluster of CPU nodes with different strengths, under realistic open-loop traffic, how much does a router gain by knowing what each request needs and what each node is good at, compared with routing that ignores both?

We compare four policies on the same cluster and the same traffic, at a moderate and a high load.

## 2. What Phase 1 tells us

All numbers are Qwen2.5-Omni-3B Q8_0 served by llama.cpp `llama-server` with 2 slots, client on the node, one request at a time unless noted.

| Finding | Evidence | Why it matters for routing |
|---|---|---|
| **Image/audio encoding scales with physical cores** and is not helped by Intel AMX | ~2.4 s per image on c7a.2xlarge (8 physical cores) vs 5–6 s on c7i.2xlarge (4 physical cores), the same with AMX off | multimodal requests have a node they are clearly fastest on |
| **Text prefill benefits from AMX** | 7.4–9.4 ms/token on c7i.2xlarge, 15.5 ms/token with AMX off, 18 ms/token on c6i (no AMX) | long-text requests rank nodes differently from image requests |
| **Decode (generating the answer) is fastest on Graviton** | 43 tok/s on c8g.2xlarge vs 12–14 tok/s on the x86 nodes (fixed 128-token output) | short chat, which is mostly decode, has yet another best node |
| **Once the slots are full, throughput is flat** | concurrency 2 → 10 on 2 slots: requests/s unchanged; latency = in-flight / throughput (Little's law) within ~3% | a node's capacity is a fixed rate, so queued work can be predicted |
| **A second slot helps little and unevenly** | throughput at 2 vs 1 request: 1.36× on c7a, 1.20× on c7i (Q4_K_M), 1.01× on c8g; chat alone gains most (1.9× on c7a) | sending a request to a "less busy" node is not free; nodes differ in how much parallelism helps |
| **Co-running requests interfere** | while another request prefills or encodes, a request's decode drops from ~11–14 to ~2–4 tok/s | placement affects the requests already running, not only the new one |
| **Within an instance family, halving vCPUs costs 1.7–1.9×**, consistently across workloads | c7i.xlarge vs c7i.2xlarge vs c7i.4xlarge | small nodes are predictable, slower versions of large ones |

## 3. Cluster

The AWS account allows **16 vCPUs running at once**. That rules out three 8-vCPU nodes, so we split it **8 + 4 + 4**, and give the 8-vCPU slot to the node that is best at the most expensive requests.

| | **c7a.2xlarge** | **c7i.xlarge** | **m8g.xlarge** |
|---|---|---|---|
| CPU | AMD EPYC (Genoa) | Intel Xeon (Sapphire Rapids), AMX | AWS Graviton4 (Arm) |
| vCPUs / physical cores / RAM | 8 / 8 / 16 GB | 4 / 2 / 8 GB | 4 / 4 / 16 GB |
| Expected role | images, audio, long text | overflow for heavy requests | chat |
| ~$/hour | 0.41 | 0.18 | 0.18 |

Expected time per request when it runs alone (from Phase 1; m8g.xlarge estimated as c8g.2xlarge × 1.8, the within-family factor above):

| | c7a.2xlarge | c7i.xlarge | m8g.xlarge (est.) |
|---|---|---|---|
| chat | 10.9 s | 17.9 s | **~5.6 s** |
| 3,072-token text | **34 s** | 65 s | ~120 s |
| 4 images | **27 s** | 77 s | ~100 s |
| 40 s audio | **28 s** | ~70 s (failed in Phase 1) | ~100 s |

Each request type ranks the three nodes differently. Chat is fastest on the m8g and slowest on the c7i. Heavy requests are fastest on the c7a; the c7i is the better second choice for them (AMX), and the m8g the worst. A router that ignores request type cannot exploit this.

**Why not other combinations?**
- A 4-vCPU AMD node has only 4 physical cores, the same as an 8-vCPU Intel node, so their encoding speeds overlap and the cluster loses its clearest specialization.
- Giving the 8 vCPUs to Graviton speeds up chat, but leaves every heavy request on a small x86 node, and heavy requests are most of the work.
- c8g.xlarge has 8 GB, too little for Q8_0 (c8g.2xlarge peaked at 10.4 GB), so we use m8g.xlarge: the same Graviton4 CPU with 16 GB.

**Profiled on the same machines.** Machines of the same type differed by up to ±12% in Phase 1, and the m8g.xlarge is new. Right before the routing runs, every cluster node is profiled with the Phase 1 benchmark: 2 requests per workload, at concurrency 1 with natural output, and at concurrency 1 and 2 with fixed output. The router's cost model uses only these profiles.

All nodes run the same model, quantization (Q8_0) and server settings (2 slots, 4,096 tokens per slot), so answer quality is identical across nodes and only speed differs.

## 4. Traffic

**Open loop.** Requests arrive on their own schedule, whether or not earlier ones have finished, like independent users. (Phase 1 and `roundrobin.py` were closed loop: a new request is sent only when one finishes. That never lets load exceed capacity, so queues never build the way they do in practice.)

**Trace.**
- Arrivals: a Poisson process (random, independent arrival times).
- Request types: 50% chat, 20% text (512 / 1,024 / 3,072 tokens), 15% image (1 / 2 / 4 photos), 15% audio (10 / 40 s). Sizes are equally likely within each type.
- Inputs: each request has a different input from the Phase 1 datasets (Dolly, CNN/DailyMail, COCO, LibriSpeech). All requests use natural output, greedy decoding, and at most 128 output tokens.
- Duration: 15 minutes of arrivals.

The trace is generated once from a seed and written to a file, so **every policy replays exactly the same requests at exactly the same times**. With greedy decoding the same input gives the same answer, so output lengths repeat too.

**Two load levels:** 60% and 90% of the cluster's predicted capacity for this mix. Capacity is computed from the profiles as Σ over nodes of slot gain ÷ (mix-weighted mean time per request). Both traces use the same seed: the same requests in the same order, only closer together at 90%.

## 5. Routing policies

Each policy chooses a node **when the request arrives**. The request then goes to that node's `llama-server` and waits in its queue if both slots are busy. The router keeps no queue of its own.

| Policy | Picks | Knows request type? | Knows node speed? | Knows node load? |
|---|---|---|---|---|
| `random` | a node at random (seeded) | no | no | no |
| `rr` | nodes in turn | no | no | no |
| `least` | fewest requests in flight per slot | no | no | yes (count only) |
| `ect` | earliest estimated completion time | yes | yes | yes (predicted work) |

**`least`:** the router counts, for each node, the requests it has sent there that haven't finished, and picks the lowest `in_flight / slots`, breaking ties at random. It treats a chat and a 4-image request the same.

**`ect` (earliest estimated completion time):**

1. *Cost model:* `cost[node][workload]` is the median time of that workload alone on that node, from the profile in section 3.
2. *Backlog:* for each node, the predicted seconds of work it has been sent and not yet finished. When a request is sent, its cost is added. While the node is busy, the backlog shrinks by `slot_gain[node]` seconds per second, and it is reset to 0 when the node has nothing in flight. `slot_gain` is the node's measured throughput at concurrency 2 ÷ concurrency 1.
3. *Choice:* the request goes to the node with the smallest `now + (backlog[node] + cost[node][request]) / slot_gain[node]`.

Example (Phase 1 numbers, ignoring slot gain): a 4-image request arrives while the c7a still has ~20 s of a long text to finish. The c7a is predicted to finish it at 20 + 27 = 47 s, the idle c7i at 77 s, and the idle m8g at ~100 s, so `ect` sends it to the **busy** c7a. `least` would send it to an idle node. This is the decision the experiment tests.

`ect` is deliberately simple: it uses only information available when the request arrives, and only Phase 1 measurements. It does not model interference between co-running requests beyond the average slot gain, or output length beyond the median.

## 6. Metrics

Per request, logged as it completes (`results/phase2/requests.csv`):
- arrival time and send time;
- the chosen node and the number of requests already on it;
- `ect`'s predicted time, logged for every policy;
- end-to-end latency, time to first token (TTFT), time waiting in the queue, prompt and output tokens, decode speed.

Per run (`results/phase2/summary.csv`), for all requests, each request type, and each node:
- median, p95 and p99 end-to-end latency;
- median and p95 TTFT;
- median queue wait;
- throughput;
- cost per 1,000 requests;
- CPU and peak RAM per node.

**Primary comparison:** p95 end-to-end latency per request type, per policy, at each load.
**Secondary:** median latency, TTFT, cost per request, where each policy sends each request type, and how accurate `ect`'s predictions are.

## 7. Hypotheses

1. **H1:** at 60% load, all policies have similar median latency, but `random` and `rr` already have a worse p95 for heavy requests, because a third of them land on the m8g.
2. **H2:** at 90% load, the gap widens. `least` beats `random` and `rr` but still sends heavy requests to the m8g whenever it is idle. `ect` has the lowest p95 for images, audio and long text.
3. **H3:** `ect` gives chat the lowest latency, by sending most chat to the m8g, without making the heavy requests on the c7a worse.
4. **H4:** `ect`'s prediction error grows with load, because interference between co-running requests is not in its model. If it is large, a model of that interference is the natural next step.

Throughput should be about the same for every policy (each sees the same arrivals, and all are below capacity). The differences should show up in latency and its tail.

## 8. Procedure

The full commands are in `docs/phase2-cluster-runbook.md`.

1. Start the two existing nodes, launch the m8g.xlarge, update code, and start `llama-server` (Q8_0, 2 slots) on each.
2. Test the c7i.xlarge with two 40 s audio requests at once (its audio failed in Phase 1 with 8 GB of RAM).
3. Profile all three nodes in parallel (~1–1.5 h).
4. The router runs on the c7a and reaches the other nodes through SSH tunnels over private IPs (same region; no inference port is exposed).
5. Compute the capacity from the profiles and generate the 60% and 90% traces.
6. Run 4 policies × 2 loads = 8 runs, ~20 minutes each, one after another, with every node idle at the start of each run.

**Time and cost:** about 5 hours in total, ~$0.77/hour for the cluster, ~$3.50 in total.

## 9. Limitations

- **One trace seed and one run per configuration.** Differences of a few percent are within noise; we will only interpret large differences. A second seed doubles the cost and is the first thing to add.
- **The router shares the c7a with a llama-server.** Its CPU use is small (it mostly waits on network sockets, and request bodies are built before the run), but it is not zero.
- **No real network.** Nodes are in one AWS region, so transfer times are milliseconds. Edge deployments with slow links would favor keeping requests local, which this experiment does not capture.
- **Immediate dispatch only.** Each request is assigned on arrival. Holding requests at the router until a node frees up (late binding) is a different design and not tested here.
- **`ect`'s costs come from requests running alone.** Interference is only captured by the average slot gain.
- **Mix and sizes are fixed.** The results apply to this mix; a mix with more heavy requests would shift load to the c7a.
- **Single model and quantization.** Quality-aware routing (e.g. a Q4_K_M node for chat and transcription, where Phase 1 found little quality loss) is future work.

## 10. Next steps after this experiment

- **Interference matrix:** measure how much each request type slows each other type on the same node, and add it to `ect`.
- **Prediction analysis:** how much of each request's latency can be predicted from what is known at arrival (type, size, prompt tokens), using the per-request logs from these runs.
- **More seeds and load levels**, including above 100%, to see when each policy breaks down.
- **Quality-aware routing** with a mixed-quantization cluster.
