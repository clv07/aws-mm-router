#!/usr/bin/env bash
# One-time setup of a new benchmark node (CPU or GPU), from a fresh clone:
#   ./src/setup.sh [--cpu|--gpu] [--no-server] [qwen|gemma] [Q8_0|Q4_K_M|f16]
# Safe to re-run: every step is skipped if it is already done.
#   1. system packages (build tools, tmux, python venv)
#   2. GPU only: NVIDIA driver and CUDA toolkit. On the AWS "Deep Learning Base OSS Nvidia Driver
#      GPU AMI" they are already there. On plain Ubuntu this installs them and asks for a reboot;
#      run the script again after the reboot.
#   3. Python virtual environment (.venv) with requirements.txt
#   4. the four datasets into dataset/ (about 3 GB)
#   5. llama.cpp build (with CUDA on a GPU), model download, and llama-server started via run.sh,
#      which checks that everything works. --no-server skips this step.
# A GPU is detected automatically (an NVIDIA PCI device); --cpu forces a CPU-only setup on a GPU
# instance, --gpu forces a GPU setup.
set -euo pipefail
cd "$(dirname "$0")/.."   # repository root

MODE=auto
SERVER=1
POSITIONAL=()
for arg in "$@"; do
  case $arg in
    --cpu) MODE=cpu ;;
    --gpu) MODE=gpu ;;
    --no-server) SERVER=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    -*) echo "unknown option $arg" >&2; exit 1 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done
MODEL_KEY=${POSITIONAL[0]:-qwen}
QUANT=${POSITIONAL[1]:-Q8_0}

if [ "$MODE" = auto ]; then
  MODE=cpu
  # 0x10de = NVIDIA's PCI vendor id; visible even before a driver is installed.
  grep -qs 0x10de /sys/bus/pci/devices/*/vendor && MODE=gpu
fi
step() { echo; echo "=== $*"; }
echo "setup: mode=$MODE model=$MODEL_KEY quant=$QUANT ($(. /etc/os-release && echo "$PRETTY_NAME"), $(uname -m), $(nproc) vCPUs)"

# 1. System packages
step "1/5 system packages"
PKGS=(git cmake build-essential tmux curl python3-venv python3-dev)
MISSING=()
for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
if [ ${#MISSING[@]} -gt 0 ]; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${MISSING[@]}"
else
  echo "already installed"
fi

# 2. NVIDIA driver and CUDA toolkit
if [ "$MODE" = gpu ]; then
  step "2/5 NVIDIA driver and CUDA toolkit"
  export PATH=/usr/local/cuda/bin:$PATH
  NEED_REBOOT=0
  if ! nvidia-smi >/dev/null 2>&1; then
    if lsmod | grep -q '^nvidia ' || dpkg -l 'nvidia-driver-*' 'nvidia-headless-*' 2>/dev/null | grep -q '^ii'; then
      echo "an NVIDIA driver is installed but not loaded"
      NEED_REBOOT=1
    else
      echo "no NVIDIA driver: installing the recommended server (compute-only) driver"
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ubuntu-drivers-common
      sudo ubuntu-drivers install --gpgpu
      # --gpgpu installs the kernel module; nvidia-smi is in the matching utils package.
      VER=$(dpkg -l 'nvidia-headless-no-dkms-*-server' 'nvidia-headless-*-server' 2>/dev/null \
              | awk '/^ii/ {print $2}' | grep -oE '[0-9]+' | head -1 || true)
      if [ -n "$VER" ] && ! command -v nvidia-smi >/dev/null; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "nvidia-utils-$VER-server"
      fi
      NEED_REBOOT=1
    fi
  else
    echo "driver: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
  fi
  if ! command -v nvcc >/dev/null; then
    echo "no nvcc: installing Ubuntu's CUDA toolkit (nvidia-cuda-toolkit)"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-cuda-toolkit
  fi
  echo "nvcc: $(nvcc --version | tail -1)"
  if [ "$NEED_REBOOT" = 1 ]; then
    echo
    echo "The NVIDIA driver was just installed and needs a reboot to load. Run:"
    echo "  sudo reboot"
    echo "then reconnect and run ./src/setup.sh $* again; it continues from here."
    exit 0
  fi
  CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.[:space:]')
  echo "GPU compute capability: $CUDA_ARCH (llama.cpp is compiled for it)"
else
  step "2/5 NVIDIA driver: skipped (CPU node)"
fi

# 3. Python environment
step "3/5 Python environment (.venv)"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
if ! pip install -q --disable-pip-version-check -r requirements.txt; then
  echo "pip install failed with $(python3 --version). requirements.txt pins versions tested on" \
       "Python 3.14; on another Python, a pinned version may not exist." >&2
  exit 1
fi
echo "$(python3 --version), requirements installed"

# 4. Datasets. A marker file per dataset records a finished download, so an interrupted one is
# retried and a finished one is skipped.
step "4/5 datasets (dataset/)"
mkdir -p dataset
for loader in dolly cnn_dailymail coco librispeech; do
  if [ -f "dataset/.$loader.done" ]; then
    echo "$loader: already downloaded"
  else
    echo "$loader: downloading"
    python3 "src/loaders/$loader.py" --download
    touch "dataset/.$loader.done"
  fi
done
du -sh dataset | awk '{print "datasets: " $1}'

# 5. llama.cpp, the model, and llama-server
if [ "$SERVER" = 1 ]; then
  step "5/5 llama.cpp build, model download, llama-server (via run.sh)"
  if [ "$MODE" = gpu ]; then
    GPU=1 CUDA_ARCH=$CUDA_ARCH ./src/run.sh "$MODEL_KEY" "$QUANT"
    grep -h 'offloaded' ~/llama-server.log || echo "warning: no 'offloaded' line in ~/llama-server.log"
  else
    ./src/run.sh "$MODEL_KEY" "$QUANT"
  fi
else
  step "5/5 llama-server: skipped (--no-server)"
fi

echo
echo "Setup done. Next, inside tmux (tmux new -s exp):"
if [ "$MODE" = gpu ]; then
  echo "  REPEATS=2 BATCH_REPEATS=1 SWEEP=\"8\" ./src/gpu_experiment.sh $MODEL_KEY $QUANT   # quick smoke test"
  echo "  ./src/gpu_experiment.sh $MODEL_KEY $QUANT                                         # full run"
else
  echo "  NODE=<label> ./src/experiment.sh $MODEL_KEY $QUANT"
fi
