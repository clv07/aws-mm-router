#!/usr/bin/env bash
# Run on the AWS instance, inside tmux:  [REPEATS=10] ./experiment.sh <qwen|gemma> [quant]
# The whole Phase 1 sweep in one command, each pass with natural and fixed (ignore_eos) output:
#   llama-server with 2 slots: concurrency 1, then 2
#   llama-server with 4 slots: concurrency 4
# Results are only ever appended (results/phase1/<node>__<model>/*_v2.csv); the log of this
# experiment goes to <that folder>/logs/experiment-<run_id>.log.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_KEY=${1:?usage: [REPEATS=10] ./experiment.sh <qwen|gemma> [Q8_0|Q4_K_M|f16]}
QUANT=${2:-Q8_0}
REPEATS=${REPEATS:-10}
CONFIGS=("2:1 2" "4:4")    # "<slots>:<concurrency levels>"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Log everything; the file moves into the node's results folder once that is known.
mkdir -p results/phase1/logs
LOG=results/phase1/logs/experiment-$RUN_ID.log
exec > >(tee -a "$LOG") 2>&1
OUT=""
finish() {
  status=$?
  if [ -n "$OUT" ] && [ -f "$LOG" ]; then
    mkdir -p "$OUT/logs" && mv "$LOG" "$OUT/logs/" && LOG=$OUT/logs/$(basename "$LOG")
  fi
  if [ $status -ne 0 ]; then
    echo "experiment stopped (exit $status); results written so far are kept. Log: $LOG"
  fi
}
trap finish EXIT

echo "experiment run_id=$RUN_ID model=$MODEL_KEY quant=$QUANT repeats=$REPEATS started $(date -u)"
for cfg in "${CONFIGS[@]}"; do
  slots=${cfg%%:*}
  concurrency=${cfg#*:}
  echo "=== llama-server with $slots slots: concurrency $concurrency, output natural + fixed"
  CTX="" SLOTS=$slots ./run.sh "$MODEL_KEY" "$QUANT"
  [ -n "$OUT" ] || OUT=$(python3 benchmark.py --where | tail -1)
  # shellcheck disable=SC2086  # $concurrency is a list of numbers
  python3 benchmark.py --repeats "$REPEATS" --concurrency $concurrency \
                      --output natural fixed --run-id "$RUN_ID"
done

echo "=== rows written by run_id=$RUN_ID (expected per row: 9 workloads x $REPEATS x concurrency)"
python3 - "$OUT/requests_v2.csv" "$RUN_ID" "$REPEATS" <<'EOF'
import collections, csv, sys
path, run_id, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3])
counts, failed = collections.Counter(), collections.Counter()
with open(path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['run_id'] == run_id:
            key = (row['slots'], row['concurrency'], row['output_mode'])
            counts[key] += 1
            failed[key] += row['status'] != 'ok'
for (slots, conc, mode), n in sorted(counts.items()):
    expected = 9 * repeats * int(conc)
    flag = '' if n == expected and not failed[(slots, conc, mode)] else '   <-- check'
    print(f'  slots={slots} concurrency={conc} output={mode}: {n}/{expected} requests, '
          f'{failed[(slots, conc, mode)]} failed{flag}')
EOF
echo "done $(date -u). Results: $OUT/summary_v2.csv and requests_v2.csv; log: $OUT/logs/experiment-$RUN_ID.log"
echo "llama-server is left running with 4 slots. Copy the folder off the instance before terminating it."
