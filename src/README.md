# Code

All code is in `src/`, with the dataset loaders in `src/loaders/`. Run everything from the repository root: results, traces and downloaded data are read and written relative to it.

```
src/
  benchmark.py      Phase 1 single-node benchmark: sends the 9 workloads to one llama-server and records
                    latency, prefill/decode speed, CPU and memory
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
  interference.py   how much two workloads slow each other down on one node: each workload alone,
                    then every pair, one per slot; `summarize` writes the slowdown matrix
                    (docs/phase2-interference.md)
  movement.py       network: time vs size through the request path (`payload`, with `sink` on the
                    other end) and network byte counters per node (`netdev`)
  split.py          prompt on one node, answer on another: every workload alone on every node and
                    split over every pair, KV saved/copied/restored; `summarize` writes costs.json
                    for the routers (docs/phase2-experiment-plan.md)
  stages.py         where an image or audio request's time goes on this node: encoding, prompt
                    processing, decoding (with llama-mtmd-cli); `summarize` gives the encoding share
  run.sh            builds llama.cpp (llama-server, llama-mtmd-cli), downloads the model, starts
                    llama-server with slot saving to ~/slots (runs on each instance)
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
python src/interference.py run --runs 02 04 07 --pairs 02+04 04+07   # text+image, image+audio
```

Scripts import each other by file name (`from benchmark import ...`, `from loaders.coco import ...`), which works because Python puts the running script's folder (`src/`) on the import path. New code goes directly in `src/`.

Don't name a module after a library or standard-library module (for example `datasets.py`, `io.py`, `queue.py`, `random.py`): a file in `src/` would hide the real one for every script.
