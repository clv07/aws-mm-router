# Phase 2 Profiling Plan

**Goal:** measure what the router's cost model still guesses, which is how requests slow each other down when they
share a node, and do it under bursty multimodal traffic.

| | |
|---|---|
| Serving nodes (AWS us-west-2) | c7a.2xlarge (8 vCPU, 2 slots), c7i.xlarge (4 vCPU, 1 slot), m8g.xlarge (4 vCPU, 2 slots) |
| Router / client | lab server s41, connected to the nodes over SSH tunnels |
| vCPU quota | **24 vCPU**: the cluster uses 16, so 8 are spare (see §3, "Using the spare 8 vCPU") |
| Cost | about $0.77/h for the three nodes. Must-have work comes to about **3 h / $2.70**; everything, about **5 h / $3.90** |

Code layout: the routing algorithm is in `src/router.py`, the CLI (`capacity`, `run`, `sim`) is in `src/main.py`, and
the mix helpers are in `src/tracegen.py`.

## Time per step

| Step | What it measures | Node(s) | Priority | Time |
|---|---|---|---|---|
| Setup (§2) | Start nodes, tunnels, datasets, checks | all | must | 45 min |
| Traces (§5) | Generate the bursty traces | laptop | must | 5 min |
| **E0** network | s41 → AWS overhead vs request size | all | must | 5 min probe (also comes free from the solo rows) |
| **E1a** interference, base grid | `slowdown(A\|B)` for chat / text / image / audio | c7a | must | 35–45 min |
| | | m8g | must | 50–60 min |
| **E1b** interference, largest sizes | chat and large × large | c7a | must | 20–25 min |
| | chat × large | m8g | nice | 40 min |
| **E2** dynamic concurrency | cost and queue wait vs k = 1..4 mixed requests | c7a | must | 15–20 min |
| | | m8g | must | 25–30 min |
| | solo + E0 | c7i | must | 7 min |
| | queueing with more requests than slots | c7i | nice | 10 min |
| **E3** bursty replay | live `onoff` trace, `ect` and `least` | cluster | must | 40 min (2 × 18–22) |
| | `step` × {ect, least}, `rr`, Poisson twin | cluster | nice | 80 min |
| Analysis (§6) | summarize, network fit, sim vs live | laptop | must | 15–20 min |
| Copy results + stop (§8) | | all | must | 10 min |

**Critical path (must): about 3 h.** Setup takes 45 min. The three node lanes then run in parallel, and the slowest,
m8g, takes about 85 min. E3 follows at about 40 min, then analysis and stop at about 25 min.

**Suggested schedule** (T = setup start):

| When | What |
|---|---|
| T+0:45 | Lanes start; `concurrency.py` gets written in parallel |
| T+1:30 | c7a E1 done; start c7a E2 |
| T+1:45 | m8g E1a done; start m8g E2 |
| T+2:15 | Start E3 |
| T+3:00 | Stop, or run the nice-to-haves |

---

## 1. What each experiment feeds

- **E0, network overhead.**
  - Formula: `overhead_s = e2e_s − (server_prompt_ms + server_predicted_ms)/1000`, fitted per node as
    `α + β · MB`.
  - It all comes from durations, so the s41 and AWS clocks don't need to be synced.
  - Feeds a dispatch-delay term in `W_n` (`Router.cost_model`, `src/router.py:267`) and an e2e offset in `sim`.
- **E1, pairwise interference.**
  - The `slowdown(A|B)` matrix per node replaces `k / gains[n]` in `cost_model` (`src/router.py:263-264`) and in
    the simulator's `rate()` (`src/main.py:241-242`).
  - It also gives the "added interference" a request causes for the one already running, and the memory limits.
- **E2, dynamic concurrency.**
  - A cost-vs-k curve per node with mixed co-runners, including k > slots, where requests queue in llama-server.
  - Checks whether the pair matrix still predicts the slowdown under random mixes.
  - Gives a measured gain to replace the m8g's borrowed `slot_gain` of 1.01.
  - Checks the FIFO wait model that the baselines use.
- **E3, bursty replay.**
  - Live ground truth for calibrating the simulator. After the update, `sim` should reproduce live `ect`/`least`
    p50/p95 on the same trace.
  - `slo` is not wired into live `run` yet, so it is evaluated in `sim`.

Why k > slots matters only for the baselines:
- `slo` dispatches only to a free slot, so under `slo` a node never runs more than 2 requests.
- `rr`, `least` and `ect` commit on arrival, so under bursts they overfill llama-server's queue. E2 and E3 measure
  that.

---

## 2. Setup (45 min; do 2.1 and 2.2 in parallel)

Placeholders: `<C7A_IP> <C7I_IP> <M8G_IP>` are the public IPs, which change on every start. `<*_ID>` are instance IDs,
`$SG` is the security group, and `<S41_EGRESS>` is s41's public IP.

### 2.1 AWS nodes (15 min)

```bash
# check what is running (the 3 nodes use 16 of the 24 vCPU), then start them
aws ec2 describe-instances --region us-west-2 --no-cli-pager \
  --filters Name=instance-state-name,Values=running,pending,stopped \
  --query 'Reservations[].Instances[].[InstanceType,InstanceId,State.Name,PublicIpAddress]' --output table
aws ec2 start-instances --region us-west-2 --instance-ids <C7A_ID> <C7I_ID> <M8G_ID>
# if they no longer exist, set them up as in docs/routing-cluster-runbook.md §2-3

# allow SSH from s41 only (on s41: curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region us-west-2 --group-id $SG --protocol tcp --port 22 --cidr <S41_EGRESS>/32
```

On each node (`ssh -i ~/.ssh/<KEY>.pem ubuntu@<IP>`):

```bash
# c7a and m8g: 2 slots (the run.sh default; pins llama.cpp 8642541b2)
cd ~/aws-mm-router && git pull && ./src/run.sh qwen

# c7i: 1 slot, prompt cache off (do NOT run run.sh afterwards)
tmux kill-session -t llama 2>/dev/null; pkill -x llama-server
tmux new -d -s llama "$HOME/llama.cpp/build/bin/llama-server -m $HOME/models/Qwen2.5-Omni-3B-Q8_0.gguf \
  --mmproj $HOME/models/mmproj-Qwen2.5-Omni-3B-Q8_0.gguf --host 127.0.0.1 --port 8080 \
  -c 4096 -np 1 -t 4 --cache-ram 0 2>&1 | tee ~/llama-server.log"
until curl -sf localhost:8080/health; do sleep 2; done; echo

# every node: auto-stop after 6 h (cancel: sudo shutdown -c)
sudo shutdown -h +360
```

llama-server listens on 127.0.0.1 only, so s41 reaches it through SSH tunnels and port 8080 stays closed.

### 2.2 s41 (30 min, mostly the dataset copy)

```bash
# 1. key for s41 → AWS (never copy the .pem to s41); add the .pub line to ~/.ssh/authorized_keys on all 3 nodes
ssh-keygen -t ed25519 -N '' -f ~/.ssh/aws-routing && cat ~/.ssh/aws-routing.pub

# 2. ~/.ssh/config on s41 (update HostName after every restart)
Host aws-c7a
    HostName <C7A_IP>
Host aws-c7i
    HostName <C7I_IP>
Host aws-m8g
    HostName <M8G_IP>
Host aws-*
    User ubuntu
    IdentityFile ~/.ssh/aws-routing
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ServerAliveInterval 15
    ServerAliveCountMax 4

# 3. self-restarting tunnels (check the ports are free: ss -ltn | grep 1808)
tmux new -d -s tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 18080:127.0.0.1:8080 aws-c7a; sleep 3; done'
tmux split-window -t tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 18081:127.0.0.1:8080 aws-c7i; sleep 3; done'
tmux split-window -t tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 18082:127.0.0.1:8080 aws-m8g; sleep 3; done'

# 4. code and venv
git clone https://github.com/clv07/aws-mm-router ~/aws-mm-router && cd ~/aws-mm-router   # or git pull
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Datasets.** s41 builds every request, with images and audio sent inline as base64, so it needs `dataset/`.
- Copy it from your laptop (about 4.3 GB):
  `rsync -a --info=progress2 --exclude '.cache' dataset/ s41:aws-mm-router/dataset/`
- Use a writable copy, because `librispeech.py` writes clips into the folder.
- The text loader also needs the HF tokenizer `Qwen/Qwen2.5-Omni-3B`. If s41 has no HF access, copy
  `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B` from your laptop.

**`nodes.s41.json`** goes in the repo root on s41. It uses the same profiles, prices and overrides as
`results/routing/nodes.json`; entry order is the round-robin order.

```json
{
  "c7a.2xlarge": {"url": "http://127.0.0.1:18080", "profile": "c7a.2xlarge-p2__Qwen2.5-Omni-3B-Q8_0",
                  "ssh": "aws-c7a", "price_per_hour": 0.41},
  "c7i.xlarge":  {"url": "http://127.0.0.1:18081", "profile": "c7i.xlarge-p2__Qwen2.5-Omni-3B-Q8_0",
                  "ssh": "aws-c7i", "price_per_hour": 0.18, "slot_gain": 1.0},
  "m8g.xlarge":  {"url": "http://127.0.0.1:18082", "profile": "m8g.xlarge-p2__Qwen2.5-Omni-3B-Q8_0",
                  "ssh": "aws-m8g", "price_per_hour": 0.18, "slot_gain": 1.01}
}
```

### 2.3 Checks and network probe, E0 (5 min)

```bash
for p in 18080 18081 18082; do      # expect 2 / 1 / 2 slots, build ...8642541b
  curl -s http://127.0.0.1:$p/props | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("total_slots"), d.get("build_info"))'
done
for p in 18080 18081 18082; do for i in $(seq 10); do curl -s -o /dev/null -w '%{time_total}\n' http://127.0.0.1:$p/health; done | sort -n | sed -n 5p; done   # median RTT, expect 20-40 ms
for h in aws-c7a aws-c7i aws-m8g; do /usr/bin/time -f "$h 20MB upload %es" sh -c "head -c 20000000 /dev/zero | ssh $h 'cat >/dev/null'"; done
for h in aws-c7a aws-c7i aws-m8g; do ssh -o BatchMode=yes $h 'hostname; free -g | sed -n 2p'; done   # --ssh monitoring works
python -c "import sys; sys.path.insert(0,'src'); from interference import build_pools; p=build_pools(['00','03','06','08'],1,'fixed'); import json; print({k: len(json.dumps(v[1][1])) for k,v in p.items()})"   # datasets OK; largest ~1.7 MB
```

### 2.4 Smoke test (2 min)

```bash
python src/interference.py run --node smoke-c7a --url http://127.0.0.1:18080 --ssh aws-c7a --runs 00 --repeats 1 --pool 2
rm -rf results/interference/smoke-c7a__*
```

Always pass `--node`. Without it, results are labeled with s41's hostname.

---

## 3. Run the experiments (on s41, one tmux window per lane, lanes in parallel)

Run `mkdir -p ~/plogs` first. Steps inside a lane are joined with `;`, so one failure doesn't stop the rest.

### Lane c7a: about 75–90 min, all must

```bash
N=c7a.2xlarge-p2-s41; U=http://127.0.0.1:18080; S=aws-c7a
( # E1a base grid: 4 solo + 10 pairs (35-45 min)
  python src/interference.py run --node $N --url $U --ssh $S --runs 00 02 04 07 --repeats 5 --pool 60 --run-id 20261119-s41-base ;
  # E1b largest sizes (20-25 min); large self-pairs skipped, since Phase 1's c=2 rows already cover them
  python src/interference.py run --node $N --url $U --ssh $S --runs 00 03 06 08 \
        --pairs 00+03 00+06 00+08 03+06 03+08 06+08 --repeats 3 --pool 40 --run-id 20261119-s41-large ;
  # E2 dynamic concurrency, k = 1..4 (15-20 min)
  python src/concurrency.py run --node $N --url $U --ssh $S --mix default --repeats 4 --allow-queue \
        --groups 00+mix 00+mix+mix 00+mix+mix+mix mix+mix mix+mix+mix mix+mix+mix+mix --run-id 20261119-s41-conc
) 2>&1 | tee -a ~/plogs/c7a.log
```

### Lane m8g: about 75–90 min must, plus 40 min nice (the critical path)

```bash
N=m8g.xlarge-p2-s41; U=http://127.0.0.1:18082; S=aws-m8g
( # E1a base grid (50-60 min; repeats 4 because large inputs take 100-130 s alone here)
  python src/interference.py run --node $N --url $U --ssh $S --runs 00 02 04 07 --repeats 4 --pool 60 --run-id 20261119-s41-base ;
  # E2 dynamic concurrency (25-30 min)
  python src/concurrency.py run --node $N --url $U --ssh $S --mix default --repeats 3 --allow-queue \
        --groups 00+mix 00+mix+mix 00+mix+mix+mix mix+mix+mix mix+mix+mix+mix --run-id 20261119-s41-conc ;
  # NICE, E1b chat x large (40 min): only if it ends before E3, otherwise run it after E3
  python src/interference.py run --node $N --url $U --ssh $S --runs 00 03 06 08 \
        --pairs 00+03 00+06 00+08 --repeats 3 --pool 40 --run-id 20261119-s41-large
) 2>&1 | tee -a ~/plogs/m8g.log
```

### Lane c7i: 7 min must, plus 10 min nice

The c7i has 1 slot, so it gets no pairs (`interference.py` needs 2 slots). For the cost model, f = 1 and W is the
running request's remaining time.

```bash
N=c7i.xlarge-p2-s41; U=http://127.0.0.1:18081; S=aws-c7i
( # E2 solo + E0 (7 min)
  python src/concurrency.py run --node $N --url $U --ssh $S --groups 00 04 08 --repeats 3 --run-id 20261119-s41-solo ;
  # NICE, queueing with more requests than slots (10 min)
  python src/concurrency.py run --node $N --url $U --ssh $S --mix default --repeats 3 --allow-queue \
        --groups mix+mix mix+mix+mix --run-id 20261119-s41-queue
) 2>&1 | tee -a ~/plogs/c7i.log
```

**If `concurrency.py` isn't ready,** use `benchmark.py` instead. E3 then serves as the mixed-load ground truth.

```bash
# c7i solo / E0 (7 min)
for r in 00 04 08; do python src/benchmark.py --node $N --url $U --ssh $S --run $r --repeats 3 --output fixed; done
# c7a / m8g single-workload cost vs k (15-25 min per node)
python src/benchmark.py --node $N --url $U --ssh $S --run 00 --concurrency 3 4 --allow-queue --repeats 2 --output fixed   # then --run 04
```

### Using the spare 8 vCPU (optional, saves about 30 min)

The 24-vCPU quota leaves room for a second m8g.xlarge (4 vCPU, $0.18/h) to split the critical m8g lane:
- **Setup:** set it up like the first (about 15 min, in parallel with §2) and tunnel it to local port 18083.
- **Split:** run m8g E1a on node 1 and m8g E2 on node 2 at the same time, under separate labels
  (`m8g.xlarge-p2-s41` and `m8g.xlarge-p2-s41-b`).
- **Effect:** the critical path becomes the c7a lane (about 85 min) instead of m8g E1a + E2.
- **E3 must use the original three nodes only,** so the cluster matches Phase 1.

### E3, bursty replay: about 40 min must, plus 80 min nice (start once the c7a and m8g must-steps finish)

Live `run` accepts `random`/`rr`/`least`/`ect`. Use `ect` and `least` as ground truth. Keep `--output natural`
(the default), because `sim` samples Phase 1 natural-output times.

```bash
# smoke test (1 min; rows have cluster=smoke, filter them out later)
python src/main.py run --trace traces/phase2/onoff-default-load0.7-dur900-seed1.csv --nodes nodes.s41.json --policy rr --cluster smoke --requests 4

# MUST: onoff x {ect, least} (18-22 min each)
T=traces/phase2/onoff-default-load0.7-dur900-seed1.csv
for p in ect least; do
  python src/main.py run --trace $T --nodes nodes.s41.json --policy $p --cluster s41-c7a-c7i-m8g
done 2>&1 | tee -a ~/plogs/e3.log
# NICE: step-uniform x {ect, least} (40 min), rr on onoff (20 min), Poisson twin with ect (20 min)
```

Rows are appended to `results/routing/requests.csv` and `summary.csv`. While E3 runs, watch the c7i for OOM:
`ssh aws-c7i 'free -m | sed -n 2p; sudo dmesg | grep -i oom | tail -2'`.

---

## 4. New script: `src/concurrency.py` (about 1–1.5 h to write; needed by T+1:30)

It reuses `interference.py` by import rather than extending it: that script hard-codes two streams, and its CSV
header can't change.

**Imports:** `benchmark` (`request_once`, `server_info`, `RemoteSamples`, `append_rows`, `RUNS`, …),
`interference` (`build_pools`, `token_gaps`, `read_csv`) and `tracegen` (`MIXES`, `parse_mix`).

```
python src/concurrency.py run --groups G [G ...] [--mix default] [--repeats 4] [--allow-queue]
       [--max-wall 2400] [--max-requests 200] [--pool 30] [--jitter 0.5] [--seed 1]
       [--output fixed|natural (default fixed)] [--run-id ID] [--node LABEL] [--url URL] [--ssh HOST]
python src/concurrency.py summarize FOLDER [--interference INTERF_FOLDER] [--network results/concurrency/network.csv]
python src/concurrency.py network FOLDER [FOLDER ...]
```

### `run`

- **Groups.** A group is `+`-joined streams. Each stream is a run ID (`00`–`08`) or `mix`, which draws each
  request's workload from `--mix`. k is the number of streams, so `00+mix+mix` is chat plus two random co-runners.
- **Slot check.** Exit if k > slots and `--allow-queue` isn't set.
- **Streams.**
  - One closed-loop thread per stream, each with one request in flight.
  - When k ≥ 2, a seeded random pause of 0 to `--jitter` s comes before each request.
  - Pool index `(c*k + i) % len(pool)`, so streams never send the same input.
  - One warm-up request per run first.
- **Stopping.** A group ends when every stream has finished `--repeats` requests, or when `--max-wall` has passed.
- **Recorded per request:** `inflight_at_send` and `others_at_send`, the co-runners at send time.
- **Output** goes to `results/concurrency/<node>__<model>/`, append-only and keyed by `run_id`:
  - **`requests.csv`** has the `interference.py` request fields, plus `group, k, stream, spec, request_bytes,
    service_s, overhead_s, svc_start_s, inflight_at_send, others_at_send`, where
    `service_s = (server_prompt_ms + server_predicted_ms)/1000`, `overhead_s = e2e_s − service_s` and
    `svc_start_s = end_s − service_s`.
  - **`groups.csv`** has `run_id, timestamp_utc, model, node, slots, output_mode, group, k, mix, wall_s, requests,
    failed, req_s, peak_ram_gb, cpu_pct`.

### `network`

- Fits `overhead_s = α + β · MB` per node on requests that ran alone.
- Writes `results/concurrency/network.csv` (`node, requests, alpha_s, beta_s_per_mb, r2`), overwritten on each run.

### `summarize` (writes `curve.csv`)

- One row per (node, group, run), plus aggregate rows per (k, run).
- **Columns:** `solo_service_s`, median/p95 `e2e_s`, `service_slowdown`, `e2e_slowdown`, median/p95 `queue_s`,
  `mean_corunners`, and median/p10/p90 `work_ratio`.
- **Queue wait:** `queue_s = overhead_s − (α + β·MB)`.
- **`work_ratio`** checks whether the pair matrix explains mixed load:
  - Split each request's service time into segments by co-runner.
  - Compute Σ segment / slowdown(run | co-runner from `load_matrix`) ÷ solo time.
  - A value of 1.0 means the pair matrix predicts mixed-load cost correctly.
- **Printout:** a cost-vs-k table per run.

---

## 5. Traces (5 min; deterministic, so generate once and commit)

```bash
# MUST: bursty, default mix, 70% load; bursts at 3x rate for 20% of the time, mean 60 s (~95 requests)
python src/tracegen.py make --arrivals onoff --load 0.7 --duration 900 --seed 1 --nodes nodes.s41.json \
       --burst-factor 3 --burst-fraction 0.2 --burst-length 60
# NICE: flash crowd, uniform mix, 50% then 3x for 180 s
python src/tracegen.py make --arrivals step --mix uniform --load 0.5 --step-at 300 --step-length 180 --step-factor 3 \
       --duration 900 --seed 1 --nodes nodes.s41.json
# NICE: steady twin of the onoff trace
python src/tracegen.py make --arrivals poisson --load 0.7 --duration 900 --seed 1 --nodes nodes.s41.json
python src/tracegen.py describe traces/phase2/*.csv
```

Use 70% load, not 90%: bursts already reach about 2× capacity, and at 90% the `least`/`rr` backlogs would take
too long to drain.

---

## 6. Analysis and router inputs

### Analysis (15–20 min)

```bash
python src/interference.py summarize results/interference/c7a.2xlarge-p2-s41__Qwen2.5-Omni-3B-Q8_0
python src/interference.py summarize results/interference/m8g.xlarge-p2-s41__Qwen2.5-Omni-3B-Q8_0
python src/concurrency.py network results/interference/*-s41__* results/concurrency/*-s41__*
for n in c7a.2xlarge m8g.xlarge; do
  python src/concurrency.py summarize results/concurrency/$n-p2-s41__Qwen2.5-Omni-3B-Q8_0 \
     --interference results/interference/$n-p2-s41__Qwen2.5-Omni-3B-Q8_0 --network results/concurrency/network.csv
done
python src/main.py sim --trace traces/phase2/onoff-default-load0.7-dur900-seed1.csv --nodes nodes.s41.json --policies rr,least,ect,slo
```

**Sanity checks.**
- Solo times are within about 10% of Phase 1 plus network overhead.
- `failed = 0`.
- `solo_max_gap_s` is under 0.5 s. If it isn't, the tunnel is batching tokens: use server-side times.

### Router update (after profiling)

1. **`nodes.s41.json`:** add `"interference": "<E1 folder>"` to the c7a and m8g entries.
2. **`Router.__init__`** (`src/router.py:184`): load the matrix (`load_matrix`) and `(α, β)` per node, and track
   which runs are executing on each node.
3. **`cost_model`, new option `slowdown='matrix'`:**
   - Use f = 1 when alone, otherwise `M[(r.run, co_run)]`.
   - Fall back to the modality pair, then the E2 curve at k = 2, then `k/gain`.
   - Optionally add the slowdown r causes to the running request.
4. **`W_n`:** add `α_n + β_n · MB(r)`.
5. **Simulator `rate(n)` → `rate(n, i)`** (`src/main.py:241`): use the same matrix, and add network time to the
   simulated e2e.
6. **m8g `slot_gain`:** replace 1.01 with the measured gain, then regenerate the evaluation traces with the same
   seeds.

---

## 7. Done when

- [ ] **Interference matrices:** the c7a and m8g `matrix.csv` have every base cell with ≥ 5 requests (≥ 3 for the
  c7a's largest sizes), no failures, and solo times consistent with Phase 1.
- [ ] **Concurrency curves:** `curve.csv` exists for the c7a and m8g with k = 1..4 and reports `work_ratio`.
- [ ] **Network fit:** `network.csv` has α and β for all 3 nodes.
- [ ] **Live bursty runs:** E3 onoff × {ect, least} completed, all requests ok, no OOM.
- [ ] **Router calibrated:** with the matrix, sim p50/p95 for `ect`/`least` on the onoff trace are within about
  20–25% of live and closer than with the old `gains/k`. `slo` is simulated on the same trace.
- [ ] **Wrap-up:** results committed, instances stopped, a short write-up in
  `analysis/phase2-profiling/README.md`.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| c7i OOM (8 GB) | Keep 1 slot and `--cache-ram 0`; no pairs. Check `dmesg`. If it crashes, restart and rerun that policy (`least` keeps routing to a dead node). |
| RAM growth on c7a/m8g (prompt cache) | If `peak_ram_gb` passes about 13 GB, restart with `EXTRA_ARGS="--cache-ram 0" ./src/run.sh qwen`. Speed is unchanged. |
| Tunnel drops or jitter | The tunnels restart themselves. Failed requests are excluded; rerun only those pairs with a new `--run-id`. |
| `concurrency.py` late | Use the `benchmark.py` fallback (§3). |
| Missing noon | Drop the nice-to-haves first, then cut m8g E1a to `--repeats 3`. Or use the spare vCPU (§3). |
| Public IPs change | Update `~/.ssh/config` on s41; the tunnels reconnect. |
| Cost | `shutdown -h +360` on every node is a hard stop. |

---

## 9. Stop checklist (10 min)

1. All lanes finished: `tail ~/plogs/*.log`, `failed = 0`, `tmux ls`.
2. Copy results to your laptop:
   ```bash
   for d in interference concurrency routing; do rsync -av s41:aws-mm-router/results/$d/ results/$d/; done
   rsync -av s41:aws-mm-router/traces/phase2/ traces/phase2/
   scp s41:aws-mm-router/nodes.s41.json results/routing/
   mkdir -p results/phase2-profiling && scp 's41:plogs/*.log' results/phase2-profiling/
   ```
3. Stop the nodes and confirm nothing is still running:
   ```bash
   aws ec2 stop-instances --region us-west-2 --instance-ids <C7A_ID> <C7I_ID> <M8G_ID>
   aws ec2 describe-instances --region us-west-2 --filters Name=instance-state-name,Values=running,pending \
     --query 'Reservations[].Instances[].[InstanceType,InstanceId]' --output table   # should be empty
   ```
4. On s41: `tmux kill-session -t tunnels`.
5. After the final evaluation: revoke the s41 SSH rule and terminate the instances. Stopped instances still cost
   about $7/month for storage.
