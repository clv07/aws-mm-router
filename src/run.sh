#!/usr/bin/env bash
# Run on the AWS instance: [SLOTS=2|4] ./src/run.sh <qwen|gemma> [quant]
#   quant = weights file: Q8_0 (default), Q4_K_M or f16. The image/audio encoder (mmproj)
#   stays Q8_0 for every quant, so only the language model's weights change.
#   SLOTS = requests llama-server processes at once (-np, default 2); the context grows with it
#   so every slot keeps 4096 tokens. Override the total with CTX=... if needed.
#   GPU=1 = build llama.cpp with CUDA (into ~/llama.cpp/build-cuda) and offload every layer and the
#   image/audio encoder to the GPU (-ngl 99). Needs nvcc and nvidia-smi: use the AWS "Deep Learning
#   Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)", or run ./src/setup.sh once. CUDA_ARCH = GPU
#   compute capability to compile for (default: read from nvidia-smi, e.g. 75 = Tesla T4 on g4dn,
#   86 = A10G on g5, 89 = L4 on g6).
#   EXTRA_ARGS = more llama-server flags, e.g. EXTRA_ARGS="-ub 1024" (default: none).
#   CACHE_RAM = llama-server --cache-ram in MiB (default: llama-server's own); 0 on 8 GB nodes and for split.py.
#   Slot KV files (split.py, /slots?action=save|restore) go to ~/slots.
#   llama.cpp is built with patches/llama-stage-profiling.patch (encoding time per request).
# Build llama.cpp and download the model if missing, (re)start llama-server in tmux only if it
# isn't already serving this model with these flags, and wait until it is ready.
set -euo pipefail

MODEL_KEY=${1:?usage: ./src/run.sh <qwen|gemma> [Q8_0|Q4_K_M|f16]}
QUANT=${2:-Q8_0}
case $MODEL_KEY in
  qwen)  REPO=ggml-org/Qwen2.5-Omni-3B-GGUF; NAME=Qwen2.5-Omni-3B ;;
  gemma) REPO=ggml-org/gemma-4-E2B-it-GGUF;  NAME=gemma-4-E2B-it ;;
  *) echo "unknown model '$MODEL_KEY' (qwen or gemma)" >&2; exit 1 ;;
esac
MODEL=$NAME-$QUANT.gguf
MMPROJ=mmproj-$NAME-Q8_0.gguf
LLAMA_COMMIT=8642541b2     # the llama.cpp build in README.md
SLOTS=${SLOTS:-2}               # llama-server -np: requests processed at once
CTX=${CTX:-$((4096 * SLOTS))}   # llama-server -c: each slot gets CTX / SLOTS tokens (4096 by default)
GPU=${GPU:-0}
if [ "$GPU" = 1 ]; then
  export PATH=/usr/local/cuda/bin:$PATH
  nvidia-smi >/dev/null 2>&1 || { echo "GPU=1 but nvidia-smi fails: no NVIDIA driver (run ./src/setup.sh, or use the Deep Learning Base GPU AMI)" >&2; exit 1; }
  CUDA_ARCH=${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.[:space:]')}
  BUILD=build-cuda   # separate from the CPU build, so one machine can run both
  CMAKE_FLAGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
  OFFLOAD="-ngl 99"  # every layer on the GPU; the encoder (mmproj) goes to the GPU by default
else
  BUILD=build; CMAKE_FLAGS=(); OFFLOAD=""
fi

SLOT_DIR=$HOME/slots

# 1. llama.cpp at $LLAMA_COMMIT plus patches/llama-stage-profiling.patch, which makes llama-server
#    report each request's image/audio encoding time (timings.encoding_ms). The stamp file names the
#    patch's hash, so a changed patch triggers a rebuild. Local edits in ~/llama.cpp are discarded.
PATCH=$(cd "$(dirname "$0")/.." && pwd)/patches/llama-stage-profiling.patch
STAMP=~/llama.cpp/$BUILD/.stage-profiling-$(sha1sum <"$PATCH" | cut -c1-12)
if [ ! -x ~/llama.cpp/$BUILD/bin/llama-server ] || [ ! -f "$STAMP" ]; then
  echo "building llama.cpp $LLAMA_COMMIT + stage-profiling patch ($BUILD)"
  sudo apt-get update -qq && sudo apt-get install -y -qq git cmake build-essential tmux curl python3-venv
  if [ "$GPU" = 1 ] && ! command -v nvcc >/dev/null; then
    echo "GPU=1 but nvcc (CUDA toolkit) is not installed: run ./src/setup.sh" >&2; exit 1
  fi
  [ -d ~/llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp ~/llama.cpp
  (cd ~/llama.cpp && git checkout -q -f "$LLAMA_COMMIT" && git apply "$PATCH" &&
   cmake -B $BUILD -DLLAMA_CURL=OFF ${CMAKE_FLAGS[@]+"${CMAKE_FLAGS[@]}"} >~/llama-build.log 2>&1 &&
   cmake --build $BUILD -j"$(nproc)" --target llama-server >>~/llama-build.log 2>&1) ||
    { echo "llama.cpp build failed, last lines of ~/llama-build.log:" >&2; tail -20 ~/llama-build.log >&2; exit 1; }
  rm -f ~/llama.cpp/$BUILD/.stage-profiling-*; touch "$STAMP"
fi

# 2. model and its multimodal projector
mkdir -p ~/models
for f in "$MODEL" "$MMPROJ"; do
  if [ ! -f ~/models/"$f" ]; then
    echo "downloading $f"
    curl -fL --retry 3 -o ~/models/"$f".part "https://huggingface.co/$REPO/resolve/main/$f"
    mv ~/models/"$f".part ~/models/"$f"
  fi
done

# 3. llama-server, listening on localhost only (benchmark.py runs on the node; main.py reaches it through SSH tunnels)
mkdir -p "$SLOT_DIR"
CMD="$HOME/llama.cpp/$BUILD/bin/llama-server -m $HOME/models/$MODEL --mmproj $HOME/models/$MMPROJ --host 127.0.0.1 --port 8080 -c $CTX -np $SLOTS -t $(nproc) --slot-save-path $SLOT_DIR/${CACHE_RAM:+ --cache-ram $CACHE_RAM}${OFFLOAD:+ $OFFLOAD}${EXTRA_ARGS:+ $EXTRA_ARGS}"
PID=$(pgrep -x llama-server || true)
if [ -z "$PID" ] || [ "$(ps -o args= -p "$PID")" != "$CMD" ]; then
  echo "starting llama-server"
  tmux kill-session -t llama 2>/dev/null || true
  pkill -x llama-server || true
  while pgrep -x llama-server >/dev/null; do sleep 0.5; done
  tmux new-session -d -s llama "$CMD 2>&1 | tee ~/llama-server.log"
  sleep 2  # let the process appear before step 4 checks that it is still alive
else
  echo "llama-server already running this model"
fi

# 4. wait for the model to load (/health answers 503 until then)
until curl -sf localhost:8080/health >/dev/null; do
  if ! pgrep -x llama-server >/dev/null; then
    echo "llama-server exited:" >&2; tail -20 ~/llama-server.log >&2; exit 1
  fi
  sleep 1
done
echo "llama-server ready (pid $(pgrep -x llama-server), $SLOTS slots, context $CTX, $([ "$GPU" = 1 ] && echo GPU || echo CPU)); watch it with: tmux attach -t llama"
