# Characterizing and Routing Multimodal Inference on Heterogeneous Edge Machines

CSE 291A course project (Option 3). Team: Sushila Seshasayee, Livia Chandra, Liangyuan Wang, Shofiya Bootwala, Abhinav Sanisetty.

## Goals

**Research Question:** How do modality, request characteristics, and system load affect inference performance across heterogeneous edge machines, and when can workload-aware routing improve end-to-end latency and service level objective (SLO) attainment over simpler routing policies?

In the baseline benchmarks, we serve one open multimodal model, Qwen2.5-Omni-3B, with llama.cpp on AWS CPU instances, changing one factor at a time from a common baseline: CPU architecture, AMX, vCPU count, memory, and quantization. We measure latency, throughput, memory, and answer quality for each request type, first one request at a time and then under concurrent load. In the routing benchmarks, we use these profiles to compare three routing policies, round robin (`rr`), least loaded (`least`), and minimum completion time (`ect`), on a 3-node cluster of mixed CPUs.

**Next (Phase 2):** use these profiles and routing results to design and evaluate a workload-aware router.

## Key Findings

**Baseline benchmarks** (one node at a time, Q8_0 unless noted)

- **No architecture wins everything:** ARM Graviton4 decodes about 3× faster than the x86 nodes and is fastest for chat and short text, while long text is fastest on AMD Genoa or Intel with AMX (tied), and images and audio on AMD Genoa.
- **Images and audio cost more to process:** prefill time grows linearly with prompt tokens, and each image or audio token costs about 1.5–3× more than a text token, depending on the CPU, because the encoder runs first.
- **vCPUs set the speed within an architecture:** going from 4 to 8 to 16 vCPUs roughly halves both prefill and decode time at each step.
- **Memory is a capacity limit, not a speed factor:** 16, 32, and 64 GB give the same latency because the server uses at most about 11 GB, but the 8 GB c7i.xlarge crashed on 40 s audio requests.
- **Q8_0 is the best quantization trade-off:** it decodes 1.8× faster than f16 with the same perplexity (10.71 vs. 10.67), while Q4_K_M is only 1.2× faster than Q8_0 and raises perplexity by 27%.
- **Concurrency adds waiting, not throughput:** throughput stops rising once the requests in flight reach the server's 2 slots, so extra requests only wait in the queue and are better spread across nodes.

**Routing benchmarks** (3-node cluster, same traffic for every policy)

- **Knowing the request halves tail latency under load:** at 90% load, p95 latency was 240 s for `rr`, 136 s for `least`, and 67 s for `ect`, and at 60% load `ect` still cut it from 78 s to 45 s compared with `least`.
- **Only `ect` keeps up at 90% load:** it served 0.131 of the 0.136 requests/s offered, while `rr` (0.106) and `least` (0.119) fell behind and their queues kept growing.
- **Each modality goes to its fastest node:** `ect` sent 78–88% of chat to the Graviton m8g and 84–100% of images and audio to the AMD c7a, cutting chat p95 at 90% load to 27 s, compared with 228 s for `rr` and 72 s for `least`.
- **Text is the exception:** at 60% load, `least` had a lower text p95 than `ect` (46 vs. 52 s), because `ect` sent some text to the m8g, whose prefill is slowest.
- **Cost follows latency:** at 90% load, `ect` cost $1.64 per 1,000 requests compared with $2.02 for `rr` (19% less), because a policy that falls behind pays to drain its queue.
- **A simple cost model is enough:** `ect` predicts each request's time from profiles of single requests, and its median prediction error grew only from 29% to 34% between the two loads.
- **Routers need health checks:** when the 8 GB c7i.xlarge ran out of memory, `least` kept choosing it, since a node that fails instantly looks idle.

Plots for each baseline experiment (architecture, memory, vCPU, quantization, and concurrency, plus a summary figure) are in [`analysis/plot.ipynb`](analysis/plot.ipynb), with a closer look at quantization (speed, memory, and answer quality) in [`analysis/quantization/README.md`](analysis/quantization/README.md). The routing results, with figures, are written up in [`analysis/routing/README.md`](analysis/routing/README.md).

## Benchmarks

The project has two sets of benchmarks. Each document covers the setup, experiment design, where the results are, and how to reproduce them.

- **[Baseline benchmarks](docs/baseline-benchmark.md)**: Qwen2.5-Omni-3B on llama.cpp, one CPU instance at a time. Covers the model and server setup, the workloads (chat, text, image, audio) and datasets, the experiment design (one variable changed at a time: CPU architecture, AMX, vCPUs, RAM, quantization, concurrency), the metrics, the results folder for each configuration (`results/phase1/`), and the commands to reproduce each one.
- **[Routing benchmarks](docs/routing-benchmark.md)**: a 3-node cluster with different CPUs (AMD c7a, Intel c7i, Graviton m8g) serving open-loop traffic. Covers the motivation and goals, the cluster configuration, the traffic traces, the three routing policies (round robin, least loaded, minimum completion time), the metrics, the results (`results/routing/`) and how to run them. Step-by-step cluster setup: [routing cluster runbook](docs/routing-cluster-runbook.md).

## Changes from the proposal

- **Modalities:** text, images, and audio; video is not included.
- **Models:** one model (Qwen2.5-Omni-3B) with three quantizations, instead of several models.
- **Hardware:** CPU-only instances; no GPUs.
- **Concurrency and queueing** are part of Phase 1, since a router must account for nodes that are already busy.
- **Routing:** Phase 1 includes a first routing experiment, using the baseline profiles.

## Experimental Setup

- **Model, server, and workloads:** Qwen2.5-Omni-3B on CPU in llama.cpp (Q8_0 unless the quantization is varied, temperature 0, at most 128 output tokens, no prompt cache), serving chat, text, image, and audio requests from public datasets ([baseline doc, section 1](docs/baseline-benchmark.md#1-setup)).
- **Baseline nodes:** a c7i.2xlarge (Intel with AMX, 8 vCPUs, 16 GB) and configurations that each change one thing from it ([baseline doc, section 2](docs/baseline-benchmark.md#2-methodology)).
- **Routing cluster:** c7a.2xlarge (AMD, 2 slots), c7i.xlarge (Intel with AMX, 1 slot), and m8g.xlarge (ARM Graviton4, 2 slots) ([routing doc, section 2](docs/routing-benchmark.md#2-cluster)).

## Reproducing

- **Baseline benchmarks:** follow section 5 of [`docs/baseline-benchmark.md`](docs/baseline-benchmark.md); a full sweep takes about 5–6 hours per node.
- **Routing benchmarks:** follow [`docs/routing-benchmark.md`](docs/routing-benchmark.md), with step-by-step cluster setup and runs in [`docs/routing-cluster-runbook.md`](docs/routing-cluster-runbook.md).

## Repository Layout

- Repository root: the server, benchmark, dataset, answer-quality, and routing scripts (`run.sh`, `benchmark.py`, `experiment.sh`, `loadgen.py`, and others).
- [`docs/`](docs/): setup, methodology, and reproduction steps for both benchmarks.
- [`analysis/`](analysis/): plots and write-ups of the baseline, quantization, and routing results.
- [`results/phase1/`](results/phase1/): baseline benchmark results, one folder per node and quantization.
- [`results/routing/`](results/routing/): routing benchmark results and the cluster configuration used.
- [`results/quality/`](results/quality/): answer-quality scores and perplexity for each quantization.
- [`traces/`](traces/): the traffic traces replayed by every routing policy.
- [`archive/`](archive/): earlier runs not used in the report.

## References

**Model and runtime**

- Qwen2.5-Omni-3B GGUF weights: [ggml-org/Qwen2.5-Omni-3B-GGUF](https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF)
- llama.cpp: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

**Datasets**

- Dolly-15k: [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- CNN/DailyMail 3.0.0: [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
- COCO 2017 validation images: [cocodataset.org](https://cocodataset.org)
- Mini LibriSpeech: [OpenSLR 31](https://www.openslr.org/31/)
