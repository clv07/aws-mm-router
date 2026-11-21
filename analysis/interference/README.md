# Interference Between Modalities: Test Run on the Lab Server

**A test run on a lab server (s41), not on AWS.** It checks that `src/interference.py` works and shows the first trends. (`interference.py` has since been replaced by `corun.py pairs`, which takes the same measurement from the router; `summarize` and `load_matrix()` are now in `corun.py`. The commands at the end are the ones used for this run.) The numbers are not from our EC2 instances, so they shouldn't go in the report as cluster results. Method and options: [`docs/phase2-interference.md`](../../docs/phase2-interference.md).

Raw files (per-request CSVs, logs, the server and setup scripts, `lscpu` output) are kept on s41 in `/mnt/intC-ssdr1-480gb/sushila/omni/results/interference-s41-2026-11-16/`.

## Setup

| | s41 test run | c7a.2xlarge | c7i.xlarge | m8g.xlarge |
| --- | --- | --- | --- | --- |
| CPU | Intel Xeon Gold 6326 (Ice Lake), 2.9 GHz, up to 3.5 GHz | AMD EPYC Genoa | Intel Sapphire Rapids | Arm Graviton4 |
| Cores used | **8 physical** (`taskset`, one per core, no hyperthreads), one NUMA node | 8 physical (8 vCPUs) | 2 physical (4 vCPUs) | 4 physical (4 vCPUs) |
| Vector / matrix units | AVX-512, AVX-512 VNNI; **no AMX, no AVX-512 BF16** | AVX-512 incl. BF16 | AVX-512 + AMX | SVE2 |
| Memory | 503 GB shared machine; llama-server peaked at 7.2 GB | 16 GB | 8 GB | 16 GB |
| Closest Phase 1 node | c6i.2xlarge (also Ice Lake, no AMX, but 4 physical cores) | | | |

The rest matches the routing cluster:
- **Software:** llama.cpp build `8642541b2` (the commit in `src/run.sh`), CPU only; Qwen2.5-Omni-3B Q8_0 with the Q8_0 mmproj.
- **Server:** 2 slots, 4,096 tokens per slot, 8 threads, other settings as in `run.sh`.
- **Client:** on the same machine, pinned to 4 other cores.
- **Workloads:** chat, 1,024-token text, 1 image and 10 s audio. Fixed output: every request generates 128 tokens.
- **Size of the run:** 5 requests per stream, about 27 minutes in total.

**s41 is a shared lab machine.** Other users' jobs ran on its CPUs during the test (load ≈ 4 of 64 threads). The cores were pinned, but memory bandwidth and caches are shared.

### Speed alone, compared with the routing cluster

Median latency per request alone (s), fixed output of 128 tokens. Rates are tokens per second. The EC2 numbers are from the routing cluster's Phase 1 profiles (`results/phase1/*-p2__*`).

| | chat | text 1,024 | 1 image | 10 s audio | decode tok/s | text prefill tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| **s41, 8 cores** | 6.6 | 16.2 | 13.7 | 18.3 | 18–20 | 113 |
| c7a.2xlarge | 10.9 | 18.0 | 17.1 | 19.3 | 11–12 | 154 |
| c7i.xlarge | 17.4 | 31.3 | 36.6 | 40.8 | 7 | 79 |
| m8g.xlarge | 5.7 | 26.6 | 31.9 | 53.1 | 20–23 | 51 |

With 8 cores, s41 decodes about as fast as the m8g and processes prompts somewhat slower than the c7a. Overall it is closest to a c7a with a faster decode. Interference on the c7a and m8g may differ: their balance between prompt processing and decoding differs, and so does what blocks what.

## Results

### Slowdown

**Table 1: Slowdown.** Median latency while the column's workload runs in the other slot ÷ median latency alone.

| Request measured ↓ / running alongside → | chat | text 1,024 | 1 image | 10 s audio |
| --- | --- | --- | --- | --- |
| chat | 1.20 | **2.67** | **2.42** | **5.66** |
| text 1,024 | 1.09 | 1.71 | 1.60 | 1.87 |
| 1 image | 1.17 | 1.96 | 1.73 | 2.02 |
| 10 s audio | 2.09 | 1.65 | 1.52 | 1.77 |

### Pair gain

**Table 2: Pair gain.** 1/slowdown(A | B) + 1/slowdown(B | A): the two requests' combined throughput relative to running them one after the other. 1 means sharing the node gains nothing; below 1, they would finish sooner one at a time.

| | chat | text 1,024 | 1 image | 10 s audio |
| --- | --- | --- | --- | --- |
| chat | 1.67 | 1.29 | 1.27 | **0.66** |
| text 1,024 | | 1.17 | 1.14 | 1.14 |
| 1 image | | | 1.15 | 1.15 |
| 10 s audio | | | | 1.13 |

### Stalls

**Table 3: Where chat loses its time.** Chat request measured, with the column's workload in the other slot. "Stall" is the mean time per request spent in pauses between tokens longer than 5× the usual gap.

| | alone | with chat | with text 1,024 | with 1 image | with 10 s audio |
| --- | --- | --- | --- | --- | --- |
| Latency (s) | 6.6 | 7.9 | 17.6 | 16.0 | 37.4 |
| Longest pause between tokens (s) | 0.05 | 0.42 | 9.4 | 9.7 | 26.6 |
| Stall per request (s) | 0 | 0.1 | 9.3 | 7.2 | 17.2 |
| Decode slowdown | 1 | 1.17 | 2.74 | 2.47 | 5.05 |

Requests used: 5 per cell, 10 for a workload with itself (both streams). Each paired request had the other workload running for at least 80% of its time.

## Findings

1. **Chat suffers most from sharing.** Next to text, an image or audio, chat took 2.4–5.7× longer, almost all of it while decoding. Its answer stopped for about 9 s at a time, against 0.05 s alone.
2. **The cause is the other request's prompt processing.** Next to text, chat's stall (9.3 s) matches the text request's time to first token alone (9.3 s). The likely reason: a 1,024-token prompt fits in one of llama.cpp's processing batches (2,048 tokens by default), and the other slot's next token is processed in the same batch, so it waits until the whole prompt is done. Text causes this as well as images and audio, so it isn't only image or audio encoding.
3. **The large requests barely notice chat.** Text was 1.09× slower and the image 1.17× slower next to chat; audio was the exception at 2.09×.
4. **Chat + audio is worse than running them one at a time** (pair gain 0.66). Both slow down: chat 5.7×, audio 2.1×, audio mostly in its prompt processing (2.2× slower). We don't have an explanation yet; check this on AWS before relying on it.
5. **Two large requests together gain little:** pair gain 1.13–1.29. On AWS the second slot gave 1.39× on the c7a and 1.01× on the m8g (Phase 1 `slot_gain`), so a single factor per node hides which pairs share well.
6. **Memory:** llama-server peaked at 7.2 GB with two audio requests (5.9 GB with one chat request alone). That is tight for the 8 GB c7i.xlarge with 2 slots.

## What this suggests for routing

- **What's in the other slot matters, mostly for chat.** Avoid sending chat to a node whose other slot is processing a long prompt. Large requests can go next to chat at little cost. The slowdown matrix (`load_matrix()` in `corun.py`) gives an interference-aware `ect` its numbers.
- **llama.cpp's batch size may reduce the stalls.** A smaller prompt batch (`-b`/`-ub`) should let the other slot's tokens through between chunks, at some cost in prompt speed. This is a quick second test on s41.

## Next steps

- Repeat on the c7a and m8g, where the router will use the results. Add the largest sizes (3,072-token text, 4 images, 40 s audio).
- More requests per cell. Five gives a rough median, not a tail.
- Run the smaller-batch test above.
- Investigate chat + audio.

## Reproducing

On s41 (setup, server and client commands: `RUNBOOK.md` §11 in the project folder; scripts in the raw-files folder above):

```bash
taskset -c 1,3,5,7 python src/interference.py run --node s41-8c --url http://127.0.0.1:18030
python src/interference.py summarize results/interference/s41-8c__Qwen2.5-Omni-3B-Q8_0
```

The server: `taskset -c 17,19,21,23,25,27,29,31 llama-server -m Qwen2.5-Omni-3B-Q8_0.gguf --mmproj mmproj-Qwen2.5-Omni-3B-Q8_0.gguf -c 8192 -np 2 -t 8` (llama.cpp `8642541b2`, CPU-only build).
