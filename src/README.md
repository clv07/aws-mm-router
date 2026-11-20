# Code

All code is in `src/`, with the dataset loaders in `src/loaders/`. Run everything from the repository root: results, traces and downloaded data are read and written relative to it.

```
src/
  benchmark.py      Phase 1 single-node benchmark: sends the 9 workloads to one llama-server and records
                    latency, prefill/decode speed, CPU and memory; its request_once() is shared by
                    every tool and also returns encoding_ms (patched llama-server) and response_bytes
  benchmark_v3.py   benchmark.py with the encode/prefill/decode columns (requests_stages.csv)
  quality.py        answer quality per quantization (ROUGE-L vs f16, word error rate)
  router.py         the routing algorithm: Router with the policies (random, rr, least, ect, slo),
                    its cost model, and the node models it is built from (Phase 1 profiles +
                    nodes.json); a library, no command line
  main.py           the routing runner: `capacity` (predicted cluster capacity), `run` (replays a
                    trace over the live cluster with one policy), `sim` (simulates every policy);
                    summary metrics are in its evaluation stage (evaluate_run / evaluate_sim)
  tracegen.py       all trace generation: steady, bursty, step, ramp and replayed arrivals, mixes,
                    deadlines (`make`), and the routing benchmarks' traces (`phase1`)
                    (docs/phase2-traffic-generation.md)
  corun.py          co-running streams on one node, run from the router: `pairs` (each workload alone,
                    then every pair, one per slot), `groups` (k streams, fixed or drawn from a mix);
                    `summarize` writes the pairs' slowdown matrix, load_matrix() reads it for a router
                    (docs/phase2-interference.md)
  phase2_measurement.py  main.py's side files: request/response bytes, decision time, encoder time,
                    CPU/RAM samples of the nodes and of the router machine
  movement.py       network: time vs size through the request path (`payload`, with `sink` on the
                    other end) and network byte counters per node (`netdev`)
  split.py          prompt on one node, answer on another: every workload alone on every node and
                    split over every pair, KV saved/copied/restored; `summarize` writes costs.json
                    for the routers (docs/phase2-experiment-plan.md)
  run.sh            builds llama.cpp with patches/llama-stage-profiling.patch (llama-server reports
                    each request's encoding time), downloads the model, starts llama-server with
                    slot saving to ~/slots (runs on each instance)
  experiment.sh     the full single-node concurrency sweep in one command
  perplexity.sh     perplexity of each quantization
  loaders/          dataset loaders, one per modality; each downloads its data with --download
    dolly.py          chat questions (Dolly-15k)
    cnn_dailymail.py  news articles cut to exact token counts (CNN/DailyMail)
    coco.py           images (COCO 2017 validation)
    librispeech.py    speech clips (Mini LibriSpeech)
```

Setup on each machine, from the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/loaders/dolly.py --download && python src/loaders/cnn_dailymail.py --download \
  && python src/loaders/coco.py --download && python src/loaders/librispeech.py --download   # into dataset/
```

Then, for example:

```bash
./src/run.sh qwen                                           # start llama-server
python src/benchmark.py --node c7i.2xlarge                  # profile one node
python src/tracegen.py make --arrivals onoff --load 0.9 --nodes nodes.json
python src/main.py run --trace traces/phase2/... --nodes nodes.json --policy ect --cluster C
python src/corun.py pairs --node c7a.2xlarge-p2 --url http://127.0.0.1:18080 --ssh node-c7a --runs 02 04 07 --pairs 02+04 04+07
python src/corun.py summarize results/interference/c7a.2xlarge-p2__Qwen2.5-Omni-3B-Q8_0
python analysis/phase2/analyze.py --out results/phase2-analysis && python analysis/phase2/figures.py results/phase2-analysis
```

Scripts import each other by file name (`from benchmark import ...`, `from loaders.coco import ...`), which works because Python puts the running script's folder (`src/`) on the import path. New code goes directly in `src/`.

Don't name a module after a library or standard-library module (for example `datasets.py`, `io.py`, `queue.py`, `random.py`): a file in `src/` would hide the real one for every script.
