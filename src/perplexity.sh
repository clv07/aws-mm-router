#!/usr/bin/env bash
# Run on the AWS instance, after quality.py: ./src/perplexity.sh [quant ...]
# Perplexity of each downloaded Qwen2.5-Omni-3B quantization on the wikitext-2 test text,
# the standard llama.cpp measure of quantization quality (lower is better; f16 is the reference).
# Stops llama-server first so the two don't compete for CPU and memory.
# Appends to results/quality/perplexity.csv. CHUNKS x 512 tokens are scored (default 20 = ~10k tokens).
set -euo pipefail

QUANTS=${*:-"Q8_0 Q4_K_M f16"}
CHUNKS=${CHUNKS:-20}
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
BUILD=${BUILD:-build}
MODELS_DIR=${MODELS_DIR:-$HOME/models}
PY=${PY:-.venv/bin/python}
OUT=results/quality/perplexity.csv
TEXT=dataset/wikitext/wiki.test.raw

# 1. llama-perplexity, same llama.cpp build as llama-server
if [ ! -x "$LLAMA_DIR/$BUILD/bin/llama-perplexity" ]; then
  echo "building llama-perplexity"
  cmake --build "$LLAMA_DIR/$BUILD" -j"$(nproc)" --target llama-perplexity >>~/llama-build.log 2>&1 ||
    { echo "build failed, last lines of ~/llama-build.log:" >&2; tail -20 ~/llama-build.log >&2; exit 1; }
fi

# 2. wikitext-2 test text
if [ ! -f "$TEXT" ]; then
  mkdir -p "$(dirname "$TEXT")"
  $PY -c "
from datasets import load_dataset
rows = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test', cache_dir='dataset/wikitext/.cache')
open('$TEXT', 'w').write(''.join(r['text'] for r in rows))"
fi

# 3. one run per quantization
tmux kill-session -t llama 2>/dev/null || true
pkill -x llama-server || true
mkdir -p "$(dirname "$OUT")"
[ -s "$OUT" ] || echo "timestamp_utc,model,chunks,ctx,ppl,ppl_error,llama_build" > "$OUT"
# e.g. "version: 0.7.0-dev (build 11960, commit 8642541b2)" -> b11960-8642541b2, as llama-server reports it
BUILD_INFO=$("$LLAMA_DIR/$BUILD/bin/llama-perplexity" --version 2>&1 | sed -nE 's/.*build ([0-9]+), commit ([0-9a-f]+).*/b\1-\2/p' | head -1)
for q in $QUANTS; do
  MODEL=$MODELS_DIR/Qwen2.5-Omni-3B-$q.gguf
  [ -f "$MODEL" ] || { echo "skipping $q: $MODEL not downloaded"; continue; }
  echo "perplexity $q ($CHUNKS chunks)"
  LOG=results/quality/perplexity-$q.log
  "$LLAMA_DIR/$BUILD/bin/llama-perplexity" -m "$MODEL" -f "$TEXT" -c 512 --chunks "$CHUNKS" -t "$(nproc)" >"$LOG" 2>&1
  # e.g. "Final estimate: PPL = 9.8765 +/- 0.12345"
  read -r PPL ERR < <(grep 'Final estimate' "$LOG" | sed -E 's/.*PPL = ([0-9.]+) \+\/- ([0-9.]+).*/\1 \2/')
  echo "  PPL = $PPL +/- $ERR"
  echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00),Qwen2.5-Omni-3B-$q,$CHUNKS,512,$PPL,$ERR,$BUILD_INFO" >> "$OUT"
done
echo "Appended to $OUT; llama-server is stopped (start it again with ./src/run.sh qwen)"
