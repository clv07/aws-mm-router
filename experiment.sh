#!/usr/bin/env bash
# Run on the AWS instance, inside tmux:  [REPEATS=5] [QUEUE_REPEATS=2] [NODE=label] ./experiment.sh <qwen|gemma> [quant]
# NODE labels a variant of the same instance type, e.g. NODE=c7i.2xlarge-noamx (default: the instance type).
# The whole Phase 1 sweep in one command:
#   llama-server with 2 slots: concurrency 1, then 2           natural and fixed (ignore_eos) output
#   llama-server with 2 slots: concurrency 5, then 10 (queue)  fixed output; 3 and 8 requests wait
#   llama-server with 4 slots: concurrency 4                    natural and fixed output
# Requests per workload = repeats x concurrency, so the queue steps use QUEUE_REPEATS (20 at c=10).
# Results are only ever appended (results/phase1/<node>__<model>/*_v2.csv); the log of this
# experiment goes to <that folder>/logs/experiment-<run_id>.log.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_KEY=${1:?usage: [REPEATS=5] [QUEUE_REPEATS=2] ./experiment.sh <qwen|gemma> [Q8_0|Q4_K_M|f16]}
QUANT=${2:-Q8_0}
REPEATS=${REPEATS:-5}
QUEUE_REPEATS=${QUEUE_REPEATS:-2}
NODE_ARGS=()
[ -z "${NODE:-}" ] || NODE_ARGS=(--node "$NODE")
# "<slots>|<concurrency levels>|<output modes>|<repeats>|<extra benchmark.py flags>"; consecutive
# steps with the same slot count share one llama-server (run.sh restarts it only on a change).
CONFIGS=("2|1 2|natural fixed|$REPEATS|"
         "2|5 10|fixed|$QUEUE_REPEATS|--allow-queue"
         "4|4|natural fixed|$REPEATS|")
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

echo "experiment run_id=$RUN_ID model=$MODEL_KEY quant=$QUANT repeats=$REPEATS" \
     "queue_repeats=$QUEUE_REPEATS started $(date -u)"
EXPECTED=()   # "<slots> <concurrency> <output mode> <repeats>" for the final check
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r slots concurrency outputs repeats flags <<< "$cfg"
  echo "=== llama-server with $slots slots: concurrency $concurrency, output $outputs, repeats $repeats"
  CTX="" SLOTS=$slots ./run.sh "$MODEL_KEY" "$QUANT"
  [ -n "$OUT" ] || OUT=$(python3 benchmark.py --where ${NODE_ARGS[@]+"${NODE_ARGS[@]}"} | tail -1)
  # shellcheck disable=SC2086  # $concurrency, $outputs and $flags are word lists
  python3 benchmark.py --repeats "$repeats" --concurrency $concurrency \
                      --output $outputs --run-id "$RUN_ID" $flags ${NODE_ARGS[@]+"${NODE_ARGS[@]}"}
  for c in $concurrency; do
    for o in $outputs; do EXPECTED+=("$slots $c $o $repeats"); done
  done
done

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
echo "done $(date -u). Results: $OUT/summary_v2.csv and requests_v2.csv; log: $OUT/logs/experiment-$RUN_ID.log"
echo "llama-server is left running with 4 slots. Copy the folder off the instance before terminating it."
