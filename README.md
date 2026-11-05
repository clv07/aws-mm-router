# Characterizing and Routing Multimodal Inference on Heterogeneous Edge Machines

CSE 291A course project (Option 3). Team: Sushila Seshasayee, Livia Chandra, Liangyuan Wang, Shofiya Bootwala, Abhinav Sanisetty.

Emerging edge applications such as private on-premises AI, industrial automation, and robotics increasingly process varied inputs such as short chat questions, long documents, images, and audio. Unlike large datacenters, these deployments may rely on a handful of machines with different compute and memory capabilities. We study how modality, request size, and system load affect inference performance across such heterogeneous edge machines, and when workload-aware routing can improve on simpler routing policies.

In Phase 1, we profile AWS instances differing in CPU architecture (Intel with and without AMX, AMD, and ARM), vCPU count, and memory, changing one factor at a time. Each node serves the same small open multimodal model (Qwen2.5-Omni-3B) through llama.cpp on CPU only, with matched model configurations, and we also compare quantization levels. We measure each node first with one request at a time and then under concurrent load. In our measurements, no machine is best for every request: ARM Graviton4 decodes fastest, so it answers short chat questions fastest, but has the slowest prefill on long text; long text runs fastest on AMD Genoa or on Intel with AMX, and images and audio on AMD Genoa. This is what makes workload-aware routing worthwhile. We also compare round robin (`rr`), least loaded (`least`), and minimum completion time (`ect`) routing on a 3-node cluster with different CPUs.

## Key Findings

**Baseline benchmarks** (one node at a time, Q8_0 unless noted)

- **No architecture wins everything.** ARM Graviton4 decodes about 3× faster than the x86 nodes (43 vs. 12–13 tokens/s), so it answers short chat questions fastest, but it has the slowest prefill on long text. Long text runs fastest on AMD Genoa or on Intel with AMX (tied), and images and audio on AMD Genoa. The best machine depends on the request, which is why routing helps.
- **Input processing grows with input size and depends on modality.** Prefill time grows roughly linearly with prompt tokens, and image and audio tokens each cost about 2–3× more than text tokens, because they first pass through the encoder.
- **vCPUs speed up both prefill and decode.** Going from 4 to 8 to 16 vCPUs roughly halves both at each step.
- **Memory size makes no difference.** The 32 GB m7i and 64 GB r7i matched the 16 GB c7i, whose peak use stayed below its memory. Use the cheapest size that fits the model.
- **Q8_0 is the best quantization trade-off.** Q4_K_M is only modestly faster but changes answers more, and f16 is much slower.
- **Concurrency adds waiting, not throughput.** Throughput stops rising once the number of requests in flight reaches the server's slot count (2); beyond that, requests only wait in the queue. Spreading requests across nodes beats packing them onto one.

**Routing benchmarks** (3-node cluster, same traffic for every policy)

- **Knowing the request matters most under load.** At 90% load, p95 latency over all requests was 240 s for round robin, 136 s for least loaded, and 67 s for `ect`: knowing node load roughly halves it, and also knowing the request halves it again. At 60% load, `ect` still cut p95 from 78 s to 45 s compared with least loaded.
- **Only `ect` keeps up at 90% load.** It served 0.131 of the 0.136 requests/s offered; round robin (0.106) and least loaded (0.119) fell behind, so their queues grew for the whole run.
- **Each modality goes to its fastest node.** `ect` sent 78–88% of chat to the Graviton m8g and 84–100% of images and audio to the AMD c7a, cutting chat p95 at 90% load from 228 s (round robin) and 72 s (least loaded) to 27 s. Text is the exception: at 60% load, least loaded had a lower text p95 (46 vs. 52 s), because `ect` also sent some text to the m8g, whose prefill is slowest.
- **Cost follows latency.** A policy that falls behind pays to drain its queue: at 90% load, `ect` cost $1.64 per 1,000 requests vs. $2.02 for round robin (19% less).
- **A simple cost model is enough.** `ect` predicts each request's time from profiles of requests running alone, and its median prediction error grew only from 29% to 34% between the two loads.
- **Routers need health checks.** When the 8 GB c7i.xlarge ran out of memory, least loaded kept choosing it, since a node that fails instantly looks idle.

Plots for each baseline experiment (architecture, memory, vCPU, quantization, and concurrency, plus a summary figure) are in [`analysis/plot.ipynb`](analysis/plot.ipynb), with a closer look at quantization (speed, memory, and answer quality) in [`analysis/quantization/README.md`](analysis/quantization/README.md). The routing results, with figures, are written up in [`analysis/routing/README.md`](analysis/routing/README.md).

## Benchmarks

The project has two sets of benchmarks. Each document covers the setup, experiment design, where the results are, and how to reproduce them.

- **[Baseline benchmarks](docs/baseline-benchmark.md)**: Qwen2.5-Omni-3B on llama.cpp, one CPU instance at a time. Covers the model and server setup, the workloads (chat, text, image, audio) and datasets, the experiment design (one variable changed at a time: CPU architecture, AMX, vCPUs, RAM, quantization, concurrency), the metrics, the results folder for each configuration (`results/phase1/`), and the commands to reproduce each one.
- **[Routing benchmarks](docs/routing-benchmark.md)**: a 3-node cluster with different CPUs (AMD c7a, Intel c7i, Graviton m8g) serving open-loop traffic. Covers the motivation and goals, the cluster configuration, the traffic traces, the three routing policies (round robin, least loaded, minimum completion time), the metrics, the results (`results/routing/`) and how to run them. Step-by-step cluster setup: [routing cluster runbook](docs/routing-cluster-runbook.md).

## Project Goals and Phases

**Research Question:** How do modality, request characteristics, and system load affect inference performance across heterogeneous edge machines, and when can workload-aware routing improve end-to-end latency and service level objective (SLO) attainment over simpler routing policies?

### Phase 1: Baseline benchmarks

We profile CPU-only Qwen2.5-Omni-3B on one AWS instance at a time to identify how hardware, quantization, and concurrent workloads affect latency, throughput, resource use, and feasibility. The measurements guide the routing decisions. Each experiment answers a specific question:

- **One variable at a time.** Starting from a c7i.2xlarge serving Q8_0, each configuration changes exactly one thing (CPU architecture, AMX, vCPUs, RAM, quantization, or concurrency), so any difference can be attributed to that variable.
- **Repeated baseline.** The baseline is measured on three separate c7i.2xlarge instances, to measure how much two "identical" instances differ. Comparisons use runs from the same machine where possible.
- **Concurrency.** A router sends requests to machines that are already busy, so single-request numbers are not enough. On a subset of nodes we keep 1, 2, or 4 requests in flight (with a 2- or 4-slot server) to see how much processing requests in parallel helps.
- **Queueing.** At concurrency 5 and 10 on a 2-slot server, there are more requests than slots, so a queue builds up in the server. This shows how waiting time and latency grow once a node is overloaded, which a load-aware router has to predict.
- **Output modes.** Natural answers differ in length between requests, which adds noise to end-to-end latency. Fixed output (exactly 128 tokens) makes end-to-end times comparable across nodes, while natural output gives realistic latency.
- **Answer quality.** Quantization trades answer quality for speed and memory, so each quantization's answers are also scored once on one node, since quality depends on the weights and not the machine.

Details: [`docs/baseline-benchmark.md`](docs/baseline-benchmark.md).

### Phase 1: Routing benchmarks

Using the node profiles from the baseline benchmarks, we compare routing policies on a small cluster of different CPUs: an AMD c7a.2xlarge, an Intel c7i.xlarge, and an ARM Graviton4 m8g.xlarge. The question is how much a router gains by knowing each request's modality and what each node is good at, compared with a generic policy that does not look at the request.

All policies replay the same open-loop traffic trace (Poisson arrivals; 50% chat, 20% text, 15% image, 15% audio) at 60% and 90% of the cluster's predicted capacity. We compare three policies, each with more information than the last:

- **Round robin (`rr`):** sends requests to nodes in turn, using no information.
- **Least loaded (`least`):** picks the node with the fewest requests in flight per slot. It knows node load, but not the request type or node speed.
- **Minimum completion time (`ect`):** picks the node where the request is predicted to finish first, using the measured time of each workload on each node plus the work already queued there.

Comparing `rr` with `least` shows the value of knowing node load; comparing `least` with `ect` shows the value of knowing the modality and node speed.

Details: [`docs/routing-benchmark.md`](docs/routing-benchmark.md); cluster setup: [`docs/routing-cluster-runbook.md`](docs/routing-cluster-runbook.md).

**Next (Phase 2):** use these profiles and routing results to design and evaluate a workload-aware router.

### Changes from the proposal

- **Modalities:** text, images, and audio; video is not included.
- **Models:** one model (Qwen2.5-Omni-3B) with three quantizations, instead of several models.
- **Hardware:** CPU-only instances; no GPUs.
- **Concurrency and queueing** are part of Phase 1, since a router must account for nodes that are already busy.
- **Routing:** Phase 1 includes a first routing experiment, using the baseline profiles.

## Experimental Setup

**Fixed for all runs.** Every node serves Qwen2.5-Omni-3B on CPU in llama.cpp's `llama-server` (commit `8642541b2`), with Q8_0 weights unless a run varies the quantization, and the image/audio encoder always in Q8_0. Each server slot has 4,096 tokens of context. Every request uses temperature 0, at most 128 output tokens, and no prompt cache, so the same input gives the same answer on every node. All instances run in AWS us-west-2 on Ubuntu 24.04 or later, and the benchmark client runs on the same instance as the server, so network delay is not part of the measurements.

**Workloads.** Four request types in nine sizes, each from a public dataset: chat with a short question (Dolly-15k), text summarization of 512, 1,024, or 3,072 tokens (CNN/DailyMail), image description of 1, 2, or 4 images (COCO 2017 validation), and transcription of 10 or 40 seconds of audio (Mini LibriSpeech). Chat time is mostly spent generating the answer (decode), while long text, images, and audio are mostly spent processing the input (prefill, plus encoding for images and audio). The full workload table and prompts are in [`docs/baseline-benchmark.md`](docs/baseline-benchmark.md) (section 1).

**Baseline benchmarks.** The baseline is a c7i.2xlarge (Intel Sapphire Rapids with AMX, 8 vCPUs, 16 GB) serving Q8_0. Every other configuration changes exactly one thing from it:

| Variable     | Configurations compared with the baseline                                   |
| ------------ | ---------------------------------------------------------------------------- |
| Architecture | c7a.2xlarge (AMD Genoa), c8g.2xlarge (ARM Graviton4), c7i.2xlarge with AMX off, c6i.2xlarge (Intel Ice Lake, no AMX) |
| vCPUs        | c7i.xlarge (4 vCPUs, 8 GB), c7i.4xlarge (16 vCPUs, 32 GB)                    |
| RAM          | m7i.2xlarge (32 GB), r7i.2xlarge (64 GB)                                     |
| Quantization | Q4_K_M (4-bit), f16 (16-bit), both on c7i.2xlarge                            |

The vCPU rows change RAM along with vCPUs, because EC2 ties them within a family. A subset of these nodes also runs the concurrency and queueing sweep. The full matrix, with physical cores, prices, and the results folder for each configuration, is in [`docs/baseline-benchmark.md`](docs/baseline-benchmark.md) (section 2).

**Routing benchmarks.** One node per CPU family from the baseline benchmarks, all serving Q8_0 with the same llama.cpp build:

| Node        | CPU                               | vCPUs / RAM | Slots |
| ----------- | --------------------------------- | ----------- | ----- |
| c7a.2xlarge | AMD Genoa                         | 8 / 16 GB   | 2     |
| c7i.xlarge  | Intel Sapphire Rapids, AMX        | 4 / 8 GB    | 1     |
| m8g.xlarge  | ARM Graviton4                     | 4 / 16 GB   | 2     |

The m8g.xlarge has the same Graviton4 CPU as the c8g but 16 GB of memory, and the c7i.xlarge uses one slot because two large image requests at once ran it out of memory. Reasons for each choice, the traffic traces, and prices are in [`docs/routing-benchmark.md`](docs/routing-benchmark.md) (sections 2 and 3).

## Running the Benchmarks

### Requirements

- An AWS account in us-west-2 with at least 16 vCPUs of on-demand quota (the routing cluster uses all 16; stopped instances do not count).
- Ubuntu Server 24.04 LTS (64-bit x86 for Intel and AMD, ARM for Graviton), a 30 GiB gp3 disk, an SSH key pair, and inbound SSH from your IP only.
- On each instance: Python 3 with a virtual environment (`requirements.txt`) and about 3 GB of datasets. `run.sh` installs the build tools and builds llama.cpp itself.
- Time and cost: a full baseline sweep takes about 5–6 hours per node; the routing benchmarks take about 40 minutes of setup, 1–1.5 hours of profiling, and about 1.7 hours of runs, at about $0.77/hour for the cluster.

### Baseline benchmarks (one node at a time)

On the instance, inside tmux so the run continues if SSH disconnects:

```bash
git clone https://github.com/clv07/aws-mm-router && cd aws-mm-router
./run.sh qwen                                   # builds llama.cpp, downloads Q8_0 + encoder, starts llama-server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python dolly.py --download && python cnn_dailymail.py --download && python coco.py --download && python librispeech.py --download
tmux new -s bench
NODE=<label> ./experiment.sh qwen               # full sweep, ~5–6 h; results in results/phase1/<label>__<model>/
```

Copy the results folder off the instance (`scp -r`) before terminating it; stopping it keeps the disk. Variants (quantization rows, AMX off, the 8 GB c7i.xlarge, answer quality) are in [`docs/baseline-benchmark.md`](docs/baseline-benchmark.md) (section 5).

### Routing benchmarks (3-node cluster)

These steps match the runs in `results/routing/`; [`docs/routing-cluster-runbook.md`](docs/routing-cluster-runbook.md) has the full commands for the cluster setup:

1. **Network and security groups.** All three nodes must be in the same VPC and availability zone. Allow SSH from your IP to each node, and SSH from the c7a's security group to the other two, so the router can reach them; port 8080 stays closed.

2. **Start the nodes.** Start or launch the c7a.2xlarge, c7i.xlarge, and m8g.xlarge (ARM image, 30 GiB) in the c7a's subnet, with shutdown behavior set to **Stop**. Note each node's public and private IP; public IPs change on every start.

3. **Set up each node.** Clone the repo, start the server with `./run.sh qwen` (2 slots), create the venv, and download the datasets, as in the baseline benchmarks.

4. **Profile each node** in tmux, with nothing else running on it (~1–1.5 h, all three in parallel):
   ```bash
   L=<node>-p2      # c7a.2xlarge-p2, c7i.xlarge-p2, or m8g.xlarge-p2
   python benchmark.py --node $L --repeats 2 --concurrency 1 --output natural && \
   python benchmark.py --node $L --repeats 2 --concurrency 1 2 --output fixed
   ```
   The first command gives `ect` its time per workload; the second gives each node's slot gain.

5. **Configure the c7i.xlarge for its 8 GB of memory.** Restart its server with one slot and the prompt cache off (`llama-server ... -np 1 --cache-ram 0`; the full command is in step 5 of the runbook). With two slots, two large image requests at once ran it out of memory, and llama-server's default 8 GB prompt cache did too under 90% load.

6. **Set up the router on the c7a.** Create a router SSH key and add it to the other nodes' `authorized_keys`, add `node-c7a`, `node-c7i`, and `node-m8g` entries to `~/.ssh/config` (private IPs), open SSH tunnels to the other two servers in tmux (ports 8081 and 8082), and check all three `/health` endpoints. Then write `nodes.json` as in [`results/routing/nodes.json`](results/routing/nodes.json): the entry order is the round-robin order, and `slot_gain` is set to 1.0 for the c7i.xlarge (one slot) and 1.01 for the m8g.xlarge.

7. **Pick the loads.** Copy the other two profiles to the c7a, then:
   ```bash
   python loadgen.py capacity --nodes nodes.json
   python loadgen.py trace --rate <R60> --duration 900 --seed 1 --out traces/cluster-load60.csv
   python loadgen.py trace --rate <R90> --duration 900 --seed 1 --out traces/cluster-load90.csv
   ```

8. **Quick test.** `python loadgen.py run --trace traces/cluster-load60.csv --nodes nodes.json --policy rr --cluster test --requests 6` should return 6 `ok` requests, two per node; then delete the test output (`rm -rf results/routing`).

9. **Run the three policies at both loads** on the c7a, in tmux (~1.7 h, about 17 minutes per run), then stop all three nodes:
   ```bash
   for load in 60 90; do
     for p in rr least ect; do
       python loadgen.py run --trace traces/cluster-load$load.csv --nodes nodes.json --policy $p --cluster c7a-c7i-m8g \
         2>&1 | tee -a results/routing-console.log
     done
   done; \
   ssh node-c7i sudo shutdown -h now; ssh node-m8g sudo shutdown -h now; sudo shutdown -h now
   ```

10. **Collect the results.** Start only the c7a and copy `results/routing/`, the `results/phase1/*-p2__*` profiles, `traces/`, and the console log back with `scp`. Then stop it again, and terminate the nodes once nothing more needs to run.

Policies, metrics, and result files: [`docs/routing-benchmark.md`](docs/routing-benchmark.md) (sections 4–6).

## Repository Layout

**Server and baseline benchmarks**

- [`run.sh`](run.sh): builds llama.cpp (commit `8642541b2`, CPU only), downloads Qwen2.5-Omni-3B (`Q8_0` by default, or `Q4_K_M` / `f16`) with its Q8_0 image/audio encoder, and starts `llama-server` in tmux. `SLOTS=4` changes the number of requests it processes at once (default 2).
- [`benchmark.py`](benchmark.py): the single-node benchmark. Runs on the instance next to the server, sends workloads 00–08, samples `llama-server`'s CPU and memory, and writes one row per request and one summary row per workload.
- [`experiment.sh`](experiment.sh): the full concurrency sweep in one command (2 slots at concurrency 1 and 2, the queue at 5 and 10, 4 slots at concurrency 4), with a log and a final request count check.
- [`requirements.txt`](requirements.txt): Python packages for the client scripts.

**Datasets** (each script downloads its data into `dataset/`, which git ignores)

- [`dolly.py`](dolly.py): Dolly-15k short questions (chat, run 00).
- [`cnn_dailymail.py`](cnn_dailymail.py): CNN/DailyMail news articles cut to 512, 1,024, or 3,072 tokens (text, runs 01–03).
- [`coco.py`](coco.py): COCO 2017 validation images (image, runs 04–06).
- [`librispeech.py`](librispeech.py): Mini LibriSpeech recordings as 10 s and 40 s clips (audio, runs 07–08).

**Answer quality**

- [`quality.py`](quality.py): collects answers from the served quantization and scores them against f16 (ROUGE-L) and the human transcripts (word error rate).
- [`perplexity.sh`](perplexity.sh): perplexity of each quantization on wikitext-2.

**Routing**

- [`loadgen.py`](loadgen.py): generates traffic traces, estimates cluster capacity, and runs the routing policies (`rr`, `least`, `ect`).
- [`nodes.example.json`](nodes.example.json): template for the cluster's node list (`nodes.json`).
- [`traces/`](traces/): the two traffic traces replayed by every policy, [`cluster-load60.csv`](traces/cluster-load60.csv) and [`cluster-load90.csv`](traces/cluster-load90.csv).

**Analysis and docs**

- [`analysis/plot.ipynb`](analysis/plot.ipynb): plots from `results/phase1/` for architecture, memory, vCPU, quantization, and concurrency, plus a summary figure. The figures are saved in the notebook's outputs.
- [`analysis/quantization/`](analysis/quantization/): f16 vs. Q8_0 vs. Q4_K_M on the c7i.2xlarge: speed, memory, and answer quality ([`README.md`](analysis/quantization/README.md)), its figures, and [`make_plots.py`](analysis/quantization/make_plots.py), which regenerates them.
- [`analysis/routing/`](analysis/routing/): the routing results write-up ([`README.md`](analysis/routing/README.md)), its figures, and [`analyze_routing.py`](analysis/routing/analyze_routing.py), which regenerates them from `results/routing/`.
- [`docs/baseline-benchmark.md`](docs/baseline-benchmark.md): baseline benchmark setup, methodology, metrics, and reproduction steps.
- [`docs/routing-benchmark.md`](docs/routing-benchmark.md): routing benchmark motivation, cluster, traffic, policies, and metrics.
- [`docs/routing-cluster-runbook.md`](docs/routing-cluster-runbook.md): step-by-step setup and runs for the routing cluster.

**Results**

- [`results/phase1/`](results/phase1/): one folder per node label and quantization, named `<node>__<model>`. `summary.csv` / `requests.csv` (v1) hold one-request-at-a-time runs from the first version of `benchmark.py`; `summary_v2.csv` / `requests_v2.csv` (v2) hold runs from the current version (concurrency sweeps and routing profiles); `logs/` holds `experiment.sh` and `llama-server` output.

  | Folder (`…__Qwen2.5-Omni-3B-<quant>`) | Configuration | Files |
  | ------------------------------------- | ------------- | ----- |
  | `c7i.2xlarge__…-Q8_0` | Baseline; Q8_0 concurrency sweep | v1, v2, log |
  | `c7i.2xlarge-abhinav__…-Q8_0` | Baseline on a second instance (reference for the RAM rows) | v1 |
  | `c7i.2xlarge-amx__…-Q8_0`, `c7i.2xlarge-noamx__…-Q8_0` | AMX on and off, same instance | v1 |
  | `c6i.2xlarge__…-Q8_0` | Intel Ice Lake, no AMX | v1, v2, log |
  | `c7a.2xlarge__…-Q8_0` | AMD Genoa | v1, v2 (partial sweep), log |
  | `c8g.2xlarge__…-Q8_0` | ARM Graviton4 | v2 (partial sweep) |
  | `c7i.1xlarge-q8__…-Q8_0`, `c7i.4xlarge-q8__…-Q8_0` | 4 and 16 vCPUs (`c7i.1xlarge` is the c7i.xlarge) | v1 |
  | `m7i.2xlarge__…-Q8_0`, `r7i.2xlarge__…-Q8_0` | 32 and 64 GB RAM | v1 |
  | `c7i.2xlarge__…-Q4_K_M` | 4-bit quantization; Q4_K_M concurrency sweep | v1, v2, log |
  | `c7i.2xlarge__…-f16` | 16-bit weights | v1, log |
  | `c7a.2xlarge-p2__…`, `c7i.xlarge-p2__…`, `m8g.xlarge-p2__…` | Routing cluster node profiles | v2 |

- [`results/routing/`](results/routing/): [`summary.csv`](results/routing/summary.csv) (one row per run and request group), [`requests.csv`](results/routing/requests.csv) (one row per request), [`nodes.json`](results/routing/nodes.json) (the cluster configuration used), [`c7i-memory-rerun.csv`](results/routing/c7i-memory-rerun.csv) (c7i.xlarge memory samples during the rerun), and [`routing-console.log`](results/routing/routing-console.log).
- [`results/quality/`](results/quality/): [`answers.csv`](results/quality/answers.csv), [`scores.csv`](results/quality/scores.csv), [`perplexity.csv`](results/quality/perplexity.csv), and the raw perplexity logs (`perplexity-<quant>.log`).
- Earlier runs not used in the report, in [`archive/`](archive/): [`roundrobin.py`](archive/roundrobin.py) with [`roundrobin.csv`](archive/results/roundrobin.csv) and [`roundrobin_requests.csv`](archive/results/roundrobin_requests.csv) (a closed-loop round-robin test), and [`benchmark.csv`](archive/results/benchmark.csv) (a first c8g.4xlarge run).

## Results Format

Each row in the v2 files records its configuration (node, model, slots, concurrency, output mode, and `run_id`), so rows can be filtered and compared across folders; the v1 files record node, model, and concurrency. In the baseline results, `summary*.csv` reports medians per workload (end-to-end latency and p95, TTFT, prefill and decode speed, queue wait, throughput, peak memory, and CPU use), and `requests*.csv` has the same measurements for every request. Compare end-to-end latency across nodes on `fixed` output rows, and match individual requests across nodes by `run`, `input`, `concurrency`, and `output_mode`. The routing results report the same latencies per run, for all requests, each request type, and each node, plus cost per 1,000 requests. Full metric definitions: [`docs/baseline-benchmark.md`](docs/baseline-benchmark.md) (section 4) and [`docs/routing-benchmark.md`](docs/routing-benchmark.md) (section 5).

## References

**Related work**

- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving. [arXiv:2502.00937](https://arxiv.org/abs/2502.00937)
- Rocks, Pebbles, and Sand (RPS Serve). [arXiv:2603.26498](https://arxiv.org/abs/2603.26498)
- Salesforce AI Research, Enterprise Realtime Voice Agent. [GitHub](https://github.com/SalesforceAIResearch/enterprise-realtime-voice-agent)

**Model and runtime**

- Qwen2.5-Omni-3B GGUF weights: [ggml-org/Qwen2.5-Omni-3B-GGUF](https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF)
- llama.cpp: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

**Datasets**

- Dolly-15k: [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- CNN/DailyMail 3.0.0: [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
- COCO 2017 validation images: [cocodataset.org](https://cocodataset.org)
- Mini LibriSpeech: [OpenSLR 31](https://www.openslr.org/31/)
