# GPU Benchmarks: Qwen2.5-Omni-3B on a g4dn.xlarge (NVIDIA T4)

This adds one GPU node to the Phase 1 profiles. It uses the same model file (Q8_0), the same llama.cpp commit (`8642541b2`), and the same 9 workloads, so a GPU number can be put next to a CPU number with only the hardware changed. The new part is the **batch-size sweep**: on a GPU, many requests can share one forward pass almost for free, so we measure how throughput, latency and energy change as 1 to 32 requests are batched together.

## 1. The instance: g4dn.xlarge

| | g4dn.xlarge |
|---|---|
| GPU | 1 × NVIDIA Tesla T4 (Turing, compute capability 7.5) |
| GPU memory | 16 GB GDDR6, 320 GB/s |
| GPU compute | 2,560 CUDA cores, 320 Tensor Cores; 65 TFLOPS FP16 (Tensor), 130 TOPS INT8, 8.1 TFLOPS FP32 |
| GPU power | 70 W, passively cooled (clocks drop when it gets hot; we log them) |
| CPU | 4 vCPUs = 2 physical cores, Intel Xeon Platinum 8259CL (Cascade Lake: AVX-512 and VNNI, no AMX) |
| RAM | 16 GiB |
| Local disk | 125 GB NVMe instance store (lost on stop); use an EBS gp3 root for everything |
| Network / EBS | up to 25 Gbps / up to 3.5 Gbps |
| Price | about $0.526/hour on demand in us-west-2 (check the current price). For comparison, the 3-node CPU routing cluster costs $0.77/hour |

**What fits in 16 GB.** The Q8_0 language model plus the Q8_0 image/audio encoder (mmproj) take about 5 GB. Each slot reserves its own KV cache when the server starts: 36 layers × 2 KV heads × 128 dims × 2 (K and V) × 2 bytes = 36 KB per token, so a 4,096-token slot needs 144 MiB. That is 4.5 GiB for 32 slots, so the whole sweep fits with room for compute buffers. 64 slots (9 GiB of KV cache) is roughly the limit.

**What to expect.** Decode on a 3B model is limited by memory bandwidth: every step reads all of the weights. At 320 GB/s the ceiling for one request is about 90 tokens/s, against 12–13 on the x86 CPUs and 43 on Graviton4. Prefill does real matrix math, so the T4 should be one to two orders of magnitude faster than the CPUs on long text, images and audio.

**Weak spots.** Only 4 vCPUs are shared by llama-server, the image/audio preprocessing, and the benchmark client with up to 32 Python threads. Watch `cpu_pct`: if it is close to 100%, the CPU is the bottleneck, not the GPU. The T4 has no bfloat16 support, which matters for vLLM (fp16 only) but not for llama.cpp's GGUF.

## 2. How llama.cpp batches: slots

`llama-server -np N` creates **N slots**. A slot is one request in progress. It has its own sequence id and its own region of the KV cache (`-c` / N tokens, which is why `run.sh` sets `-c` to 4096 × N).

Each server step, llama-server builds **one batch from all busy slots** and runs **one forward pass** (`llama_decode`) over it:

- every slot that is generating adds its next token (1 token each), and
- every slot that is still reading its prompt adds a chunk of prompt tokens.

This is continuous batching (`-cb`, on by default). When a request finishes, its slot is free on the next step, and a waiting request can start without waiting for the rest of the batch. Requests beyond N wait in the server's queue and are not in any batch. **So in llama.cpp, the number of slots is the maximum batch size**, and to batch 16 requests on the GPU you start the server with `SLOTS=16` and keep 16 requests in flight.

This is the same mechanism we already used on CPU. The Phase 1 runs with 2 and 4 slots were batched, in batches of at most 4. The difference is what batching buys:

- **CPU:** a decode step for 2 requests cost nearly twice as much as for 1. The cores were already busy, so throughput stopped rising at 2 slots and extra requests mostly waited.
- **GPU:** a decode step for 1 request leaves most of the GPU idle while it reads 3+ GB of weights. Adding requests to the batch reuses those same weights, so a step for 16 requests costs little more than a step for 1. Throughput should keep rising with slots until the GPU's compute or memory runs out. That is the curve this experiment measures.

Two more flags control the batch size in tokens:

- `-b` (logical batch, default 2048) and `-ub` (physical micro-batch, default 512) cap the tokens in one forward pass. A 3,072-token prompt is processed over several steps. During those steps, every other slot still gets one decode token per step, so a long prefill slows down everyone's decode. Expect `tpot_ms` to rise when long prompts share the batch.
- The **image/audio encoder is not batched across slots.** Each request's images or audio are encoded on their own, before its tokens join the batch. So image and audio workloads should scale worse with slots than text does.

How this compares with vLLM: vLLM has no fixed slots. Its scheduler puts up to `max_num_seqs` requests (default 256) in a batch, limited by free KV-cache pages (PagedAttention), and it splits long prompts into chunks (chunked prefill). It can batch more requests in the same memory, but it uses different weights (fp16 or AWQ, not GGUF Q8_0), so a comparison with vLLM is a separate experiment.

## 3. Experiment design

`src/gpu_experiment.sh` runs these steps in one command:

| Step | Slots | Concurrency | Output | Repeats | Why |
|---|---|---|---|---|---|
| 1 | 2 | 1, then 2 | natural, fixed | `REPEATS` (5) | Same as Phase 1: the concurrency-1 profile and slot gain that `loadgen.py`'s `ect` uses |
| 2 | 4 | 4 | natural, fixed | `REPEATS` (5) | Same as Phase 1 |
| 3 | 8, 16, 32 (`SWEEP`) | = slots | fixed | `BATCH_REPEATS` (3) | The batch-size sweep |

Concurrency always equals the slot count, so every request in flight is in the batch and none queues. The sweep uses fixed output (exactly 128 tokens, `ignore_eos`), so every request stays in the batch for the same number of decode steps. Requests per workload are repeats × concurrency (96 at 32 slots). Expect about 1–2 hours on a T4, plus 20–40 minutes for the first CUDA build.

**Measurements.** `benchmark.py` writes its usual files (`summary_v2.csv`, `requests_v2.csv`: e2e, TTFT, prefill and decode speed, `req_s`, `out_tok_s`, host CPU and RAM). In addition:

- `logs/gpu-<run_id>.csv`: nvidia-smi every 0.5 s (GPU and memory-bandwidth utilization, VRAM, power, temperature, SM clock).
- `gpu_summary.csv` (from `src/gpu_summary.py`): those samples per workload pass, with **energy per request** (mean power × pass duration ÷ requests), an edge-relevant metric the CPU runs don't have.
- `logs/gpu-offload-<run_id>-np<slots>.txt`: llama-server's own lines on what it put on the GPU. Check it says `offloaded 37/37 layers to GPU`.

**Questions to answer from the results:**

1. How much faster is the T4 than each CPU node, per workload, at concurrency 1? Prefill vs decode.
2. How do `req_s` and `out_tok_s` grow with the batch size (2 → 32), and where do they stop growing? How much worse does each request's latency (`median_e2e_s`, `tpot_ms`) get as the batch grows?
3. Do image and audio workloads scale worse than text and chat, because their encoding is not batched?
4. Cost and energy: $ per 1,000 requests and J per request on the T4 at its best batch size, against the CPU nodes and the 3-node cluster.

Results go to `results/phase1/g4dn.xlarge-gpu__Qwen2.5-Omni-3B-Q8_0/`, next to the CPU profiles, so the node can be added to `nodes.json` as a routing target.

## 4. Running it

### Launch the instance

1. **Quota first.** Service Quotas → EC2 → "Running On-Demand G and VT instances" must be at least 4 vCPUs in us-west-2. It is often 0 by default, and an increase can take a day or more.
2. **AMI:** "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)" (search the AMI catalog). It has the NVIDIA driver and the CUDA toolkit (`nvcc` in `/usr/local/cuda/bin`), which `run.sh` needs. Plain Ubuntu works only after you install both yourself.
3. **Disk:** gp3 root, at least as large as the AMI asks for (about 75 GiB), so the AMI, llama.cpp build, model and datasets fit.
4. Same key pair and security group as the CPU nodes: inbound SSH from your IP only.

### On the instance

```bash
nvidia-smi                       # should list a Tesla T4
git clone https://github.com/clv07/aws-mm-router && cd aws-mm-router
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python src/loaders/dolly.py --download && python src/loaders/cnn_dailymail.py --download \
  && python src/loaders/coco.py --download && python src/loaders/librispeech.py --download

tmux new -s exp
./src/gpu_experiment.sh qwen                  # everything: build, Phase 1 steps, sweep 8 16 32
```

Variations:

```bash
SWEEP="8 16 32 64" ./src/gpu_experiment.sh qwen      # push the batch size further
REPEATS=2 BATCH_REPEATS=1 SWEEP="8" ./src/gpu_experiment.sh qwen   # quick smoke test first
./src/gpu_experiment.sh qwen f16                     # f16 weights (fits easily in 16 GB)
NODE=g4dn.xlarge-cpu ./src/experiment.sh qwen        # control: same machine, CPU only (GPU=0)
GPU=1 SLOTS=16 ./src/run.sh qwen                     # just start a 16-slot GPU server
```

`run.sh` without `GPU=1` behaves exactly as before. With `GPU=1` it builds llama.cpp with CUDA into `~/llama.cpp/build-cuda`, compiled only for `CUDA_ARCH` (default 75, the T4; use 86 for an A10G or 89 for an L4), and adds `-ngl 99`. `EXTRA_ARGS` passes any other flag, for example `EXTRA_ARGS="-ub 1024"` to try a larger micro-batch.

Copy the results folder off the instance before terminating it.
