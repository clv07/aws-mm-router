# Phase 2 Cluster Runbook: c7a.2xlarge + c7i.xlarge + m8g.xlarge

The goal: route an open-loop Poisson trace over three CPU nodes with different strengths, comparing four policies (`random`, `rr`, `least`, `ect`) at two load levels. Each node is profiled on itself right before the routing runs.

| Node | Instance | Owner / key | Role | ~$/h |
|---|---|---|---|---|
| c7a.2xlarge (8 vCPU, 16 GB) | `i-0cb24cd5df02b1336` `qwen-3B-c7a` | Shofiya / `shofiya-291a` | images, audio, long text; **runs the router** | 0.41 |
| c7i.xlarge (4 vCPU, 8 GB) | `i-05c94f4537607634c` `qwen-omni-8b-c7i-1xlarge` | Liangyuan? / `cse291a-benchmark` | overflow for heavy requests (AMX) | 0.18 |
| m8g.xlarge (4 vCPU, 16 GB) | **new**, `sushila-m8g-cluster` | Sushila / `sushila-291a` | chat (fast decode) | 0.18 |

All three run Qwen2.5-Omni-3B Q8_0 with 2 slots. Total: 16 vCPUs (the account limit), ~$0.77/h.

**Time:** ~40 min setup, ~1.5 h profiling, ~2.7 h routing runs. **Cost:** about $3.50, against the account's $20/day.

Placeholders: `<C7A_IP>`, `<C7I_IP>`, `<M8G_IP>` are public IPs, which change on every start. `<C7I_PRIV>`, `<M8G_PRIV>` are private IPs, which stay the same. Run the `aws` commands in **CloudShell** (console, `>_` icon, top right).

---

## 0. Before the day

1. **Livia's c6i.2xlarge must be stopped.** It uses 8 of the 16 vCPUs. Running instances count against the limit; stopped ones don't.
2. **Get access to the two teammate instances without sharing private keys.** On your laptop, print your public key:
   ```bash
   ssh-keygen -y -f ~/.ssh/sushila-291a.pem
   ```
   Send that one line (it starts with `ssh-rsa` or `ssh-ed25519`; it is safe to share) to Shofiya (c7a) and the c7i.xlarge owner. Each runs this once on their instance:
   ```bash
   echo '<your public key line>' >> ~/.ssh/authorized_keys
   ```
3. Ask both owners where their clone is (usually `~/aws-mm-router`) and whether `git status` is clean.

## 1. Network check and security group rules (CloudShell)

```bash
aws ec2 describe-instances --region us-west-2 --no-cli-pager \
  --instance-ids i-0cb24cd5df02b1336 i-05c94f4537607634c i-0abbd4495725e4854 \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,VpcId,SubnetId,join(`,`,SecurityGroups[].GroupId)]' \
  --output table
```

Note each instance's **security group** (`sg-...`) and the c7a's **subnet**. All three should share the same VPC (the default one); otherwise private IPs won't connect.

Your laptop's IP: `curl -s https://checkip.amazonaws.com` (on the laptop).

```bash
C7A_SG=sg-...      # c7a.2xlarge's group
C7I_SG=sg-...      # c7i.xlarge's group
MY_SG=sg-...       # your c7i.2xlarge's group (reused for the new m8g)
MY_IP=x.x.x.x/32

# you can SSH into the two teammate instances (skip if a group already allows your IP)
aws ec2 authorize-security-group-ingress --group-id $C7A_SG --protocol tcp --port 22 --cidr $MY_IP
aws ec2 authorize-security-group-ingress --group-id $C7I_SG --protocol tcp --port 22 --cidr $MY_IP
# the router (on the c7a) can SSH into the other two; SSH only, port 8080 stays closed
aws ec2 authorize-security-group-ingress --group-id $C7I_SG --protocol tcp --port 22 --source-group $C7A_SG
aws ec2 authorize-security-group-ingress --group-id $MY_SG  --protocol tcp --port 22 --source-group $C7A_SG
```

If two instances share a group, the rule is already there and AWS says so (`InvalidPermission.Duplicate`). That's fine.

## 2. Start the two existing nodes and launch the m8g (CloudShell)

```bash
aws ec2 start-instances --region us-west-2 --instance-ids i-0cb24cd5df02b1336 i-05c94f4537607634c

C7A_SUBNET=subnet-...   # from step 1, so all nodes are in the same zone
aws ec2 run-instances --region us-west-2 --instance-type m8g.xlarge \
  --image-id resolve:ssm:/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
  --key-name sushila-291a --subnet-id $C7A_SUBNET --security-group-ids $MY_SG \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=sushila-m8g-cluster}]' \
  --query 'Instances[0].InstanceId' --output text

# public and private IPs of all three (rerun after every start)
aws ec2 describe-instances --region us-west-2 --no-cli-pager \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[InstanceType,PublicIpAddress,PrivateIpAddress]' --output table
```

If `run-instances` fails with `VcpuLimitExceeded`, something else is still running (step 0.1).

## 3. Set up the m8g (new node, ~25 min)

```bash
ssh -i ~/.ssh/sushila-291a.pem ubuntu@<M8G_IP>
git clone https://github.com/clv07/aws-mm-router && cd aws-mm-router
git checkout phase2-loadgen          # PR #6; use main once it is merged
./run.sh qwen                        # builds llama.cpp, downloads Q8_0 + mmproj, starts the server
python3 -m venv .venv && source .venv/bin/activate
pip install requests datasets transformers soundfile
python dolly.py --download && python cnn_dailymail.py --download && python coco.py --download && python librispeech.py --download
```

## 4. Bring the two existing nodes up to date (~5 min each)

On the c7a (`ssh -i ~/.ssh/sushila-291a.pem ubuntu@<C7A_IP>`) and on the c7i.xlarge:

```bash
cd ~/aws-mm-router
git status                           # if teammates' files would block the pull, move them aside (don't delete)
git fetch && git checkout phase2-loadgen && git pull
source .venv/bin/activate
./run.sh qwen                        # restarts llama-server after the stop/start (Q8_0, 2 slots)
```

## 5. Audio test on the c7i.xlarge (~5 min)

Its 40 s audio failed in Phase 1, probably because it ran out of memory. Test the worst case, two 40 s clips at once:

```bash
python benchmark.py --node test-c7i --run 08 --repeats 1 --concurrency 2 --output fixed
free -g; sudo dmesg | grep -i -E 'oom|killed' | tail -3
rm -rf results/phase1/test-c7i__*
```

- **Both `ok`:** continue.
- **Errors or an OOM line:** `SLOTS=1 ./run.sh qwen` and test again with `--concurrency 1`. If it still fails, stop this instance and launch an **m7i.xlarge** like the m8g in step 2: x86 AMI path `.../24.04/stable/current/amd64/hvm/ebs-gp3/ami-id`, then step 3.

## 6. Profile all three in parallel (~1–1.5 h, set by the slowest node)

On **each** node, in tmux (`tmux new -s profile`), with its own label:

```bash
L=c7a.2xlarge-p2     # c7i.xlarge-p2 on the c7i, m8g.xlarge-p2 on the m8g
python benchmark.py --node $L --repeats 2 --concurrency 1 --output natural && \
python benchmark.py --node $L --repeats 2 --concurrency 1 2 --output fixed
```

- The first command gives `ect` its costs: time per workload when a request runs alone, natural output.
- The second gives the slot gain: how much faster the node is with 2 requests at once.
- The `-p2` labels keep these runs in their own folders, `results/phase1/<label>__Qwen2.5-Omni-3B-Q8_0/`, separate from the Phase 1 sweeps.

Nothing else may run on a node while it is being profiled.

## 7. Router setup on the c7a (~10 min; can start while step 6 runs)

```bash
# the router's own key: it stays on this instance and is never copied anywhere else
ssh-keygen -t ed25519 -N '' -f ~/.ssh/router
cat ~/.ssh/router.pub >> ~/.ssh/authorized_keys          # lets the router sample this node's own CPU/RAM
cat ~/.ssh/router.pub                                     # copy this line
```

Add that line to `~/.ssh/authorized_keys` on the c7i.xlarge and on the m8g (you can SSH into both). Then on the c7a:

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
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
EOF
ssh node-c7i hostname && ssh node-m8g hostname && ssh node-c7a hostname   # each prints a name, no password prompt
```

Tunnels: the router reaches each remote llama-server through SSH, so port 8080 stays closed. Run them in a tmux session so they survive logout:

```bash
tmux new -d -s tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 8081:127.0.0.1:8080 node-c7i; sleep 5; done'
tmux split-window -t tunnels 'while true; do ssh -N -o ExitOnForwardFailure=yes -L 8082:127.0.0.1:8080 node-m8g; sleep 5; done'
curl -s 127.0.0.1:8080/health; curl -s 127.0.0.1:8081/health; curl -s 127.0.0.1:8082/health   # {"status":"ok"} x3
```

`nodes.json`, in `~/aws-mm-router` on the c7a. The entry order is also the round-robin order:

```json
{
  "c7a.2xlarge": {"url": "http://127.0.0.1:8080", "profile": "c7a.2xlarge-p2__Qwen2.5-Omni-3B-Q8_0",
                  "ssh": "node-c7a", "price_per_hour": 0.41},
  "c7i.xlarge":  {"url": "http://127.0.0.1:8081", "profile": "c7i.xlarge-p2__Qwen2.5-Omni-3B-Q8_0",
                  "ssh": "node-c7i", "price_per_hour": 0.18},
  "m8g.xlarge":  {"url": "http://127.0.0.1:8082", "profile": "m8g.xlarge-p2__Qwen2.5-Omni-3B-Q8_0",
                  "ssh": "node-m8g", "price_per_hour": 0.18}
}
```

## 8. Pick the load levels (after step 6 has finished on all three)

Copy the two remote profiles to the router, then check the cost model:

```bash
scp -r node-c7i:aws-mm-router/results/phase1/c7i.xlarge-p2__Qwen2.5-Omni-3B-Q8_0 results/phase1/
scp -r node-m8g:aws-mm-router/results/phase1/m8g.xlarge-p2__Qwen2.5-Omni-3B-Q8_0 results/phase1/
python loadgen.py capacity --nodes nodes.json
```

Check that every node has a row, that nothing important is marked `*` (estimated), and that the gains look plausible (1.0–1.4). Take the printed 60% and 90% rates:

```bash
python loadgen.py trace --rate <R60> --duration 900 --seed 1 --out traces/cluster-load60.csv
python loadgen.py trace --rate <R90> --duration 900 --seed 1 --out traces/cluster-load90.csv
```

Both traces use the same seed: the same requests in the same order, only closer together at 90%.

## 9. Quick test (~5 min)

```bash
python loadgen.py run --trace traces/cluster-load60.csv --nodes nodes.json --policy rr --cluster test --requests 6
```

You should see 6 `ok` requests, two per node, and a summary. Then delete the test output: `rm -rf results/phase2`.

## 10. The 8 routing runs (~2.7 h), then shut everything down

On the c7a, in tmux (`tmux new -s routing`):

```bash
for load in 60 90; do
  for p in random rr least ect; do
    python loadgen.py run --trace traces/cluster-load$load.csv --nodes nodes.json --policy $p --cluster c7a-c7i-m8g \
      2>&1 | tee -a results/routing-console.log
  done
done; \
ssh node-c7i sudo shutdown -h now; ssh node-m8g sudo shutdown -h now; sudo shutdown -h now
```

Detach with `Ctrl-b d`. Each run prints its summary when it finishes. If one run fails, the loop goes on to the next, and finished requests are already saved.

**Shutdown behavior:** all three must be set to **Stop**, not Terminate (Actions → Instance settings → Change shutdown behavior).

## 11. Collect the results (next session, ~5 min, ~$0.03)

Start only the c7a, then from your laptop:

```bash
cd ~/Desktop/UCSD/claude/291a/aws-mm-router
scp -i ~/.ssh/sushila-291a.pem -r ubuntu@<C7A_IP>:aws-mm-router/results/phase2 results/
scp -i ~/.ssh/sushila-291a.pem -r 'ubuntu@<C7A_IP>:aws-mm-router/results/phase1/*-p2__*' results/phase1/
scp -i ~/.ssh/sushila-291a.pem -r ubuntu@<C7A_IP>:aws-mm-router/traces .
scp -i ~/.ssh/sushila-291a.pem ubuntu@<C7A_IP>:aws-mm-router/results/routing-console.log results/phase2/
```

Then stop the c7a again.

## Afterwards

- Remove the extra security group rules if the owners want them gone: same commands as step 1 with `revoke-security-group-ingress`.
- Remove the router key from the c7i.xlarge's `authorized_keys`.
- Keep the m8g stopped; terminate it once nothing more needs to run on it.
- Tell Livia the vCPUs are free again.
