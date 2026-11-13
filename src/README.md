# Code

All code is in `src/`, with the dataset loaders in `src/loaders/`. Run everything from the repository root: results, traces and downloaded data are read and written relative to it.

```
src/
  benchmark.py      Phase 1 single-node benchmark: sends the 9 workloads to one llama-server and records
                    latency, prefill/decode speed, CPU and memory
  quality.py        answer quality per quantization (ROUGE-L vs f16, word error rate)
  loadgen.py        routing: replays a trace over the cluster with a routing policy (rr, least, ect);
                    also `capacity` (predicted cluster capacity) and `trace` (simple Poisson traces)
  tracegen.py       Phase 2 traffic generator: steady, bursty, step, ramp and replayed arrivals, mixes,
                    deadlines (docs/phase2-traffic-generation.md)
  run.sh            builds llama.cpp, downloads the model, starts llama-server (runs on each instance)
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
python src/loadgen.py run --trace traces/phase2/... --nodes nodes.json --policy ect --cluster C
```

Scripts import each other by file name (`from benchmark import ...`, `from loaders.coco import ...`), which works because Python puts the running script's folder (`src/`) on the import path. New code goes directly in `src/`.

Don't name a module after a library or standard-library module (for example `datasets.py`, `io.py`, `queue.py`, `random.py`): a file in `src/` would hide the real one for every script.
