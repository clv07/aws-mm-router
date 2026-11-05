# Routing Benchmarks: Cluster Runbook

Step-by-step commands to reproduce the routing benchmarks in [`routing-benchmark.md`](routing-benchmark.md): set up a 3-node cluster, profile each node, generate the traces, and replay them with the `rr`, `least` and `ect` policies at two loads.

| Node                        | Role                                              | ~$/h |
| --------------------------- | ------------------------------------------------- | ---- |
| c7a.2xlarge (8 vCPU, 16 GB) | images, audio, long text; **also runs the router** | 0.41 |
| c7i.xlarge (4 vCPU, 8 GB)   | long text (AMX), overflow                          | 0.18 |
| m8g.xlarge (4 vCPU, 16 GB)  | chat (fastest decode)                              | 0.18 |

All three run Qwen2.5-Omni-3B Q8_0 in AWS us-west-2: 16 vCPUs in total, ~$0.77/h.

**Time:** ~40 min setup, ~1.5 h profiling, ~1.7 h for the 6 routing runs. **Cost:** about $3.

**Placeholders.** `<KEY>` is your EC2 key pair (`~/.ssh/<KEY>.pem` on your laptop). `<C7A_IP>`, `<C7I_IP>`, `<M8G_IP>` are public IPs, which change on every start; `<C7I_PRIV>`, `<M8G_PRIV>` are private IPs, which stay the same. `<C7A_ID>`, `<C7I_ID>`, `<M8G_ID>` are instance IDs. Run the `aws` commands in **CloudShell** (AWS console, `>_` icon, top right) or anywhere the AWS CLI is configured.

**Security.** Never copy a `.pem` private key to an instance or share it. llama-server listens on localhost only; the router reaches the other nodes through SSH tunnels, so port 8080 is never opened.

---

## 1. Security groups

The three instances must be in the same VPC and availability zone, so they can reach each other over private IPs. Use one security group for all three (or adjust the rules below per group):

```bash
SG=sg-...             # the security group for the three nodes
MY_IP=x.x.x.x/32      # your laptop's IP: curl -s https://checkip.amazonaws.com

aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr $MY_IP    # SSH from your laptop
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --source-group $SG  # SSH between nodes (router tunnels)
```

`InvalidPermission.Duplicate` means the rule already exists; that's fine.

## 2. Launch the three instances

The account's on-demand limit is 16 vCPUs, so nothing else may be running.

```bash
SUBNET=subnet-...     # one subnet, so all three are in the same zone
X86=resolve:ssm:/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id
ARM=resolve:ssm:/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id
for spec in c7a.2xlarge:$X86 c7i.xlarge:$X86 m8g.xlarge:$ARM; do
  aws ec2 run-instances --region us-west-2 --instance-type ${spec%%:*} --image-id ${spec#*:} \
    --key-name <KEY> --subnet-id $SUBNET --security-group-ids $SG \
    --instance-initiated-shutdown-behavior stop \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=routing-${spec%%.*}}]" \
    --query 'Instances[0].InstanceId' --output text
done

# public and private IPs (rerun after every start)
aws ec2 describe-instances --region us-west-2 --no-cli-pager \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[InstanceType,InstanceId,PublicIpAddress,PrivateIpAddress]' --output table
```

`VcpuLimitExceeded` means another instance is still running.

## 3. Set up each node (~25 min, all three in parallel)

On each node (`ssh -i ~/.ssh/<KEY>.pem ubuntu@<NODE_IP>`):

```bash
git clone https://github.com/clv07/aws-mm-router && cd aws-mm-router
./run.sh qwen                        # builds llama.cpp, downloads Q8_0 + mmproj, starts llama-server (2 slots)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python dolly.py --download && python cnn_dailymail.py --download && python coco.py --download && python librispeech.py --download
```

## 4. Profile all three in parallel (~1–1.5 h, set by the slowest node)

On **each** node, in tmux (`tmux new -s profile`), with its own label:

```bash
L=c7a.2xlarge-p2     # c7i.xlarge-p2 on the c7i, m8g.xlarge-p2 on the m8g
python benchmark.py --node $L --repeats 2 --concurrency 1 --output natural && \
python benchmark.py --node $L --repeats 2 --concurrency 1 2 --output fixed
```

- The first command gives `ect` its costs: the time of each workload when a request runs alone.
- The second gives the slot gain: how much more throughput the node has with 2 requests at once.
- Results go to `results/phase1/<label>__Qwen2.5-Omni-3B-Q8_0/`, separate from the baseline runs. The repository already has our profiles under the `-p2` labels and `benchmark.py` appends to existing files, so to profile from scratch use your own labels (e.g. `c7a.2xlarge-mine`) and put them in `nodes.json` in step 6.

Nothing else may run on a node while it is being profiled.

## 5. Restart the c7i.xlarge's server for routing (8 GB of RAM)

With 8 GB, the c7i.xlarge runs out of memory with two image requests at once, and llama-server's default prompt cache (up to 8 GB) also fills its memory over a long run. After profiling, restart it with **1 slot and the prompt cache off**:

```bash
tmux kill-session -t llama; pkill -x llama-server
tmux new -d -s llama "$HOME/llama.cpp/build/bin/llama-server -m $HOME/models/Qwen2.5-Omni-3B-Q8_0.gguf \
  --mmproj $HOME/models/mmproj-Qwen2.5-Omni-3B-Q8_0.gguf --host 127.0.0.1 --port 8080 \
  -c 4096 -np 1 -t 4 --cache-ram 0 2>&1 | tee ~/llama-server.log"
until curl -sf localhost:8080/health; do sleep 2; done; echo
```

Don't run `./run.sh` on this node afterwards: it would restart the server with 2 slots and the cache on. Every input in a trace is different, so the cache never gets a hit, and turning it off does not change speed.

## 6. Router setup on the c7a (~10 min; can start while step 4 runs)

```bash
# the router's own key: it stays on this instance and is never copied anywhere else
ssh-keygen -t ed25519 -N '' -f ~/.ssh/router
cat ~/.ssh/router.pub >> ~/.ssh/authorized_keys          # lets the router sample this node's own CPU/RAM
cat ~/.ssh/router.pub                                     # copy this line
```

Append that line to `~/.ssh/authorized_keys` on the c7i and on the m8g (SSH into each from your laptop). Then on the c7a:

```bash
cat >> ~/.ssh/config <<'EOF'
Host node-c7a
    HostName 127.0.0.1
Host node-c7i
    HostName <C7I_PRIV>
Host node-m8g
    HostName <M8G_PRIV>
Host node-*
    User ubuntu
    IdentityFile ~/.ssh/router
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
EOF
ssh node-c7i hostname && ssh node-m8g hostname && ssh node-c7a hostname   # each prints a name, no password prompt
```

Tunnels to the two remote llama-servers, in tmux so they survive logout:

```bash
tmux new -d -s tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 8081:127.0.0.1:8080 node-c7i; sleep 5; done'
tmux split-window -t tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 8082:127.0.0.1:8080 node-m8g; sleep 5; done'
curl -s 127.0.0.1:8080/health; curl -s 127.0.0.1:8081/health; curl -s 127.0.0.1:8082/health   # {"status":"ok"} x3
```

`~/aws-mm-router/nodes.json` on the c7a. The entry order is also the round-robin order; `slot_gain` overrides the measured gain (1.0 for the c7i, which now has one slot; we used 1.01 for the m8g, measured on the c8g.2xlarge, the same CPU):

```json
{
  "c7a.2xlarge": {"url": "http://127.0.0.1:8080", "profile": "c7a.2xlarge-p2__Qwen2.5-Omni-3B-Q8_0",
                  "ssh": "node-c7a", "price_per_hour": 0.41},
  "c7i.xlarge":  {"url": "http://127.0.0.1:8081", "profile": "c7i.xlarge-p2__Qwen2.5-Omni-3B-Q8_0",
                  "ssh": "node-c7i", "price_per_hour": 0.18, "slot_gain": 1.0},
  "m8g.xlarge":  {"url": "http://127.0.0.1:8082", "profile": "m8g.xlarge-p2__Qwen2.5-Omni-3B-Q8_0",
                  "ssh": "node-m8g", "price_per_hour": 0.18, "slot_gain": 1.01}
}
```

Our exact file is in `results/routing/nodes.json`.

## 7. Capacity and traces (after step 4 has finished on all three)

Copy the two remote profiles to the router, then check the cost model:

```bash
scp -r node-c7i:aws-mm-router/results/phase1/c7i.xlarge-p2__Qwen2.5-Omni-3B-Q8_0 results/phase1/
scp -r node-m8g:aws-mm-router/results/phase1/m8g.xlarge-p2__Qwen2.5-Omni-3B-Q8_0 results/phase1/
python loadgen.py capacity --nodes nodes.json
```

Check that every node has a row, that nothing important is marked `*` (estimated), and that the gains look plausible (1.0–1.4). Take the printed 60% and 90% rates (ours: 0.089 and 0.134 req/s):

```bash
python loadgen.py trace --rate <R60> --duration 900 --seed 1 --out traces/cluster-load60.csv
python loadgen.py trace --rate <R90> --duration 900 --seed 1 --out traces/cluster-load90.csv
```

Both traces use the same seed: the same requests in the same order, only closer together at 90%. To replay exactly our traffic, use the traces in `traces/` instead.

## 8. Quick test (~5 min)

The repository already contains our results in `results/routing/`, and `loadgen.py` appends to them. Move them aside first so your runs start fresh files:

```bash
mv results/routing results/routing-original
python loadgen.py run --trace traces/cluster-load60.csv --nodes nodes.json --policy rr --cluster test --requests 6
```

You should see 6 `ok` requests, two per node, and a summary. Then delete the test output (only the new folder; ours is in `results/routing-original`): `rm -rf results/routing`.

## 9. The 6 routing runs (~1.7 h)

On the c7a, in tmux (`tmux new -s routing`):

```bash
for load in 60 90; do
  for p in rr least ect; do
    python loadgen.py run --trace traces/cluster-load$load.csv --nodes nodes.json --policy $p --cluster c7a-c7i-m8g \
      2>&1 | tee -a routing-console.log
  done
done
```

Detach with `Ctrl-b d`. Each run prints a summary when it finishes, and its results are appended to `results/routing/requests.csv` and `summary.csv`. If a run fails, the loop goes on to the next; repeat the failed run afterwards (the analysis uses the latest run of each load and policy). Check every request finished with `grep -c error routing-console.log` and, on the c7i, `sudo dmesg | grep -i oom`.

## 10. Collect the results and stop the cluster

From your laptop, in your clone of the repository:

```bash
scp -i ~/.ssh/<KEY>.pem -r ubuntu@<C7A_IP>:aws-mm-router/results/routing results/
scp -i ~/.ssh/<KEY>.pem -r 'ubuntu@<C7A_IP>:aws-mm-router/results/phase1/*-p2__*' results/phase1/
scp -i ~/.ssh/<KEY>.pem -r ubuntu@<C7A_IP>:aws-mm-router/traces .
scp -i ~/.ssh/<KEY>.pem ubuntu@<C7A_IP>:aws-mm-router/routing-console.log results/routing/
```

Then stop all three:

```bash
aws ec2 stop-instances --region us-west-2 --instance-ids <C7A_ID> <C7I_ID> <M8G_ID>
```

Analyse with `python analysis/routing/analyze_routing.py`.

## Afterwards

- Terminate the instances once nothing more needs to run on them (stopped instances still pay for their disks).
- Remove the security group rule for SSH between nodes if it is no longer needed: same command as step 1 with `revoke-security-group-ingress`.
