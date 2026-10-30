# Phase 1 Benchmark

A single Python script runs on the **benchmark client (Mac)**, sending HTTP requests to AWS EC2 inference nodes and collecting CPU/RAM metrics via SSH.

## 1. Setup

All scripts are in the project root:

```
benchmark.py
cnn_dailymail.py
coco.py
librispeech.py
dataset/
results/
```

Create a virtual environment and install dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip install requests datasets transformers soundfile
```

Download datasets:

```
python cnn_dailymail.py --download
python coco.py --download
python librispeech.py --download
```

## 2. SSH Configuration

Edit `~/.ssh/config` on your Mac:

```
Host qwen-large
    HostName <EC2_PUBLIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/key.pem

Host qwen-small
    HostName <EC2_PUBLIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/key.pem
```

Set key permissions:

```
chmod 400 ~/.ssh/key.pem
```

Create SSH tunnels in separate terminals:

```
ssh -N -L 8080:127.0.0.1:8080 qwen-large
ssh -N -L 8081:127.0.0.1:8080 qwen-small
```

Verify:

```
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8081/health
```

Keep both tunnels running during benchmarking.

## 3. Benchmark Configuration

Edit `benchmark.py`:

```
MODEL = "qwen"

NODES = {
    "large": {
        "url": "http://127.0.0.1:8080",
        "ssh": "qwen-large",
    },
    "small": {
        "url": "http://127.0.0.1:8081",
        "ssh": "qwen-small",
    },
}
```

Default: 10 measured requests, 1 warm-up, 128 max output tokens, concurrency = 1.

## 4. Workloads

| Run   | Modality | Size                     | Dataset       |
| ----- | -------- | ------------------------ | ------------- |
| 01–03 | Text     | 512 / 1024 / 3072 tokens | CNN/DailyMail |
| 04–06 | Image    | 1 / 2 / 4 images         | COCO          |
| 07–09 | Audio    | 5 / 10 / 20 seconds      | LibriSpeech   |

Text sizes assume a 4096-token context window.

## 5. Run

Single workload:

```
python benchmark.py --run 01
```

All workloads:

```
python benchmark.py
```

Results are **appended**, not overwritten:

```
results/benchmark.csv
```

## 6. Metrics

| Metric                                           | Description                           |
| ------------------------------------------------ | ------------------------------------- |
| `model`, `node`, `modality`, `size`, `size_unit` | Experiment configuration              |
| `median_e2e_s`, `p95_s`                          | Request latency                       |
| `req_s`                                          | Sequential throughput                 |
| `ttft_s`                                         | Time to first token                   |
| `decode_tok_s`                                   | Decode throughput                     |
| `peak_ram_gb`                                    | Peak sampled process RSS              |
| `cpu_pct`                                        | CPU usage normalized to node capacity |
| `success`, `failed`                              | Request outcomes                      |

CPU/RAM are sampled remotely via SSH every 0.2 seconds. P95 with 10 repetitions is exploratory. Keep the model, quantization, and runtime settings consistent across nodes.