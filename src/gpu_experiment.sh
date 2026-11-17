#!/usr/bin/env bash
# Run on a GPU instance (e.g. g4dn.xlarge), inside tmux:
#   [REPEATS=5] [BATCH_REPEATS=3] [SWEEP="8 16 32"] [NODE=label] ./src/gpu_experiment.sh <qwen|gemma> [quant]
# The GPU version of experiment.sh: llama.cpp built with CUDA (GPU=1 ./src/run.sh), every layer
# and the image/audio encoder on the GPU, and a batch-size sweep.
#   1. 2 slots: concurrency 1, then 2        natural and fixed output   (same as Phase 1, so loadgen.py
#                                                                        can use this node's profile)
#   2. 4 slots: concurrency 4                natural and fixed output   (same as Phase 1)
#   3. N slots: concurrency N, N in SWEEP    fixed output               (the batching sweep)
# Concurrency always equals the slot count, so every request in flight is in the batch and none
# waits in the queue. Fixed output (128 tokens each) keeps every request in the batch for the same
# number of decode steps, so throughput at different batch sizes is comparable.
# Requests per workload = repeats x concurrency (BATCH_REPEATS=3 at 32 slots -> 96 requests).
# GPU utilization, memory, power and clocks are sampled every 0.5 s into
# <results folder>/logs/gpu-<run_id>.csv and summarized per workload by gpu_summary.py.
# NODE defaults to <instance type>-gpu (e.g. g4dn.xlarge-gpu), so a CPU-only run on the same
# instance type can use another label.
set -euo pipefail
cd "$(dirname "$0")/.."   # repository root: results/ and dataset/ are relative to it

MODEL_KEY=${1:?usage: [REPEATS=5] [BATCH_REPEATS=3] [SWEEP="8 16 32"] ./src/gpu_experiment.sh <qwen|gemma> [Q8_0|Q4_K_M|f16]}
QUANT=${2:-Q8_0}
REPEATS=${REPEATS:-5}
BATCH_REPEATS=${BATCH_REPEATS:-3}
SWEEP=${SWEEP:-8 16 32}
export GPU=1 CUDA_ARCH=${CUDA_ARCH:-75}

if [ -z "${NODE:-}" ]; then
  TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token --max-time 2 \
               -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)
  ITYPE=$(curl -sf --max-time 2 -H "X-aws-ec2-metadata-token: $TOKEN" \
               http://169.254.169.254/latest/meta-data/instance-type || hostname)
  NODE=$ITYPE-gpu
fi
# "<slots>|<concurrency levels>|<output modes>|<repeats>"
CONFIGS=("2|1 2|natural fixed|$REPEATS"
         "4|4|natural fixed|$REPEATS")
for n in $SWEEP; do CONFIGS+=("$n|$n|fixed|$BATCH_REPEATS"); done
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

mkdir -p results/phase1/logs
LOG=results/phase1/logs/gpu-experiment-$RUN_ID.log
exec > >(tee -a "$LOG") 2>&1
OUT=""
SAMPLER=""
finish() {
  status=$?
  [ -z "$SAMPLER" ] || kill "$SAMPLER" 2>/dev/null || true
  if [ -n "$OUT" ] && [ -f "$LOG" ]; then
    mkdir -p "$OUT/logs" && mv "$LOG" "$OUT/logs/" && LOG=$OUT/logs/$(basename "$LOG")
  fi
  if [ $status -ne 0 ]; then
    echo "experiment stopped (exit $status); results written so far are kept. Log: $LOG"
  fi
}
trap finish EXIT

echo "gpu experiment run_id=$RUN_ID node=$NODE model=$MODEL_KEY quant=$QUANT repeats=$REPEATS" \
     "batch_repeats=$BATCH_REPEATS sweep='$SWEEP' started $(date -u)"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv
EXPECTED=()
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r slots concurrency outputs repeats <<< "$cfg"
  echo "=== llama-server with $slots slots on the GPU: concurrency $concurrency, output $outputs, repeats $repeats"
  CTX="" SLOTS=$slots ./src/run.sh "$MODEL_KEY" "$QUANT"
  if [ -z "$OUT" ]; then
    OUT=$(python3 src/benchmark.py --where --node "$NODE" | tail -1)
    mkdir -p "$OUT/logs"
    # TZ=UTC: nvidia-smi writes local time, requests_v2.csv is in UTC.
    TZ=UTC nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm \
      --format=csv,nounits -lms 500 > "$OUT/logs/gpu-$RUN_ID.csv" &
    SAMPLER=$!
  fi
  # How much of the model llama-server actually put on the GPU, and the VRAM it reserved.
  grep -E 'offloaded|CUDA0|KV self size|KV buffer' ~/llama-server.log \
    > "$OUT/logs/gpu-offload-$RUN_ID-np$slots.txt" 2>/dev/null || true
  grep -h 'offloaded' "$OUT/logs/gpu-offload-$RUN_ID-np$slots.txt" || echo "  warning: no 'offloaded' line in ~/llama-server.log"
  # shellcheck disable=SC2086  # $concurrency and $outputs are word lists
  python3 src/benchmark.py --repeats "$repeats" --concurrency $concurrency \
                      --output $outputs --run-id "$RUN_ID" --node "$NODE"
  for c in $concurrency; do
    for o in $outputs; do EXPECTED+=("$slots $c $o $repeats"); done
  done
done
kill "$SAMPLER" 2>/dev/null || true; SAMPLER=""

echo "=== rows written by run_id=$RUN_ID (expected: 9 workloads x repeats x concurrency)"
python3 - "$OUT/requests_v2.csv" "$RUN_ID" "${EXPECTED[@]}" <<'EOF'
import collections, csv, sys
path, run_id = sys.argv[1], sys.argv[2]
repeats = {tuple(e.split()[:3]): int(e.split()[3]) for e in sys.argv[3:]}
counts, failed = collections.Counter(), collections.Counter()
with open(path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['run_id'] == run_id:
            key = (row['slots'], row['concurrency'], row['output_mode'])
            counts[key] += 1
            failed[key] += row['status'] != 'ok'
for (slots, conc, mode) in sorted(set(counts) | set(repeats), key=lambda k: (int(k[0]), int(k[1]), k[2])):
    n = counts[(slots, conc, mode)]
    expected = 9 * repeats.get((slots, conc, mode), 0) * int(conc)
    flag = '' if n == expected and not failed[(slots, conc, mode)] else '   <-- check'
    print(f'  slots={slots} concurrency={conc} output={mode}: {n}/{expected} requests, '
          f'{failed[(slots, conc, mode)]} failed{flag}')
EOF
python3 src/gpu_summary.py "$OUT"
echo "done $(date -u). Results: $OUT/summary_v2.csv, requests_v2.csv, gpu_summary.csv; log: $OUT/logs/gpu-experiment-$RUN_ID.log"
echo "llama-server is left running. Copy the folder off the instance before terminating it."
