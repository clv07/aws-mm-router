#!/usr/bin/env bash
# Run on the AWS instance: ./run.sh <qwen|gemma> [quant]
#   quant = weights file: Q8_0 (default), Q4_K_M or f16. The image/audio encoder (mmproj)
#   stays Q8_0 for every quant, so only the language model's weights change.
# Build llama.cpp and download the model if missing, (re)start llama-server in tmux only if it
# isn't already serving this model with these flags, and wait until it is ready.
set -euo pipefail

MODEL_KEY=${1:?usage: ./run.sh <qwen|gemma> [Q8_0|Q4_K_M|f16]}
QUANT=${2:-Q8_0}
case $MODEL_KEY in
  qwen)  REPO=ggml-org/Qwen2.5-Omni-3B-GGUF; NAME=Qwen2.5-Omni-3B ;;
  gemma) REPO=ggml-org/gemma-4-E2B-it-GGUF;  NAME=gemma-4-E2B-it ;;
  *) echo "unknown model '$MODEL_KEY' (qwen or gemma)" >&2; exit 1 ;;
esac
MODEL=$NAME-$QUANT.gguf
MMPROJ=mmproj-$NAME-Q8_0.gguf
LLAMA_COMMIT=8642541b2     # the llama.cpp build in README.md
CTX=8192 SLOTS=2           # llama-server -c / -np: each slot gets CTX / SLOTS tokens

# 1. llama.cpp
if [ ! -x ~/llama.cpp/build/bin/llama-server ]; then
  echo "building llama.cpp $LLAMA_COMMIT"
  sudo apt-get update -qq && sudo apt-get install -y -qq git cmake build-essential tmux curl python3-venv
  [ -d ~/llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp ~/llama.cpp
  (cd ~/llama.cpp && git checkout -q "$LLAMA_COMMIT" &&
   cmake -B build -DLLAMA_CURL=OFF >~/llama-build.log 2>&1 &&
   cmake --build build -j"$(nproc)" --target llama-server >>~/llama-build.log 2>&1) ||
    { echo "llama.cpp build failed, last lines of ~/llama-build.log:" >&2; tail -20 ~/llama-build.log >&2; exit 1; }
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

# 3. llama-server, listening on localhost only (benchmark.py runs on the node; roundrobin.py uses SSH tunnels)
CMD="$HOME/llama.cpp/build/bin/llama-server -m $HOME/models/$MODEL --mmproj $HOME/models/$MMPROJ --host 127.0.0.1 --port 8080 -c $CTX -np $SLOTS -t $(nproc)"
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
echo "llama-server ready (pid $(pgrep -x llama-server)); watch it with: tmux attach -t llama"
