# Cluster Access (Phase 2)

How the team shares the Phase 2 cluster without re-authorizing each other every time. Set up once; afterwards the only thing anyone updates is their own IP rule.

| Role | Instance | vCPU |
| --- | --- | --- |
| Router (the only machine reachable from the internet) | c7i.large | 2 |
| Node | c7a.2xlarge | 8 |
| Node | c7i.2xlarge | 8 |
| Node | m8g.xlarge | 4 |

All four are in us-west-2d. Together they use 22 of the account's 24 vCPUs, and the quota is shared by everyone's instances (GPU instances count separately).

## How access works

- **Laptop → router:** your own SSH key. The router's security group has one SSH rule per teammate IP, and the router has an Elastic IP, so its address never changes.
- **Router → nodes, node → node:** one shared cluster key that exists only on the four instances, plus an SSH config with the private IPs (fixed for the life of an instance). `split.py --transport direct` uses this for KV copies.
- **Laptop → node:** jumps through the router (`ProxyJump`) with your own key, which is on every machine.

Everyone logs in as `ubuntu`. Private keys (`.pem` files, `~/.ssh/id_*`) never leave their owner's laptop; only `.pub` lines are shared.

## One-time setup

### 1. Everyone: share your public key

Post the single line of your `~/.ssh/id_ed25519.pub` in the team channel, with your name at the end of the line (`ssh-ed25519 AAAA… yourname`). Collect all lines into a file `team_keys.pub`.

### 2. AWS console (one person, with the instance owners' OK)

1. **Security group `mm-cluster`**: inbound SSH with source = `mm-cluster` itself. Attach it to all four instances (Actions → Security → Change security groups → add; works while running, and an instance can have several groups).
2. **Security group `mm-entry`**: one inbound SSH rule per teammate, source = their IP/32, description = their name. Attach it to the **router only**.
3. **Elastic IP**: allocate one and associate it with the router (about $3.60/month).
4. Afterwards, the node owners remove per-person IP rules from the nodes' own groups, so the router is the only way in.

Never use 0.0.0.0/0 as a source, and never open port 8080: llama-server listens on 127.0.0.1 and is reached through SSH tunnels.

### 3. Each machine's owner: authorize the keys

On the router, its owner creates the cluster key and posts the `.pub` line:

```bash
ssh-keygen -t ed25519 -N '' -C mm-cluster -f ~/.ssh/cluster && cat ~/.ssh/cluster.pub
```

On every machine (router and the three nodes), its owner appends the team keys and the cluster key:

```bash
cat team_keys.pub cluster.pub >> ~/.ssh/authorized_keys
```

This survives stop/start; it only needs repeating for a new teammate.

### 4. On the router: give every machine the cluster key and address book

```bash
cat > ~/.ssh/cluster_config <<'EOF'
Host router
    HostName <ROUTER_PRIVATE_IP>
Host node-c7a
    HostName <C7A_PRIVATE_IP>
Host node-c7i
    HostName <C7I_PRIVATE_IP>
Host node-m8g
    HostName <M8G_PRIVATE_IP>
Host router node-*
    User ubuntu
    IdentityFile ~/.ssh/cluster
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
EOF
include='touch ~/.ssh/config; grep -q "^Include cluster_config" ~/.ssh/config || { echo "Include cluster_config"; cat ~/.ssh/config; } > ~/.ssh/c.tmp && mv ~/.ssh/c.tmp ~/.ssh/config; chmod 600 ~/.ssh/config ~/.ssh/cluster'
eval "$include"
for h in node-c7a node-c7i node-m8g; do
  scp ~/.ssh/cluster ~/.ssh/cluster.pub ~/.ssh/cluster_config $h:.ssh/ && ssh $h "$include"
done
# every machine can reach every other one
for a in router node-c7a node-c7i node-m8g; do for b in router node-c7a node-c7i node-m8g; do
  ssh $a ssh -o BatchMode=yes $b true && echo "$a -> $b ok" || echo "$a -> $b FAIL"; done; done
```

Every machine now has the same aliases, so a nodes file can use them for both `ssh` and `peer` (e.g. `"ssh": "node-m8g", "peer": "node-m8g"`).

### 5. Everyone: laptop SSH config

```
Host mm-router
    HostName <ELASTIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
Host mm-c7a
    HostName <C7A_PRIVATE_IP>
Host mm-c7i
    HostName <C7I_PRIVATE_IP>
Host mm-m8g
    HostName <M8G_PRIVATE_IP>
Host mm-c7a mm-c7i mm-m8g
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump mm-router
```

Then `ssh mm-router`, `ssh mm-c7a`, `rsync -a mm-c7a:aws-mm-router/results/stages/ results/stages/` all work.

### 6. Optional: start servers and tunnels on boot

After testing each command by hand, add it to `crontab -e`:

- each node: `@reboot cd ~/aws-mm-router && CACHE_RAM=0 ./src/run.sh qwen >> ~/boot.log 2>&1`
- router: a `~/tunnels.sh` with the tunnel loops (ports 18080–18082 → the nodes' 8080), started the same way

## What changes, and what to do

| What | Changes when | What to do |
| --- | --- | --- |
| Your public IP | You change network (home, campus, hotspot) | Console → `mm-entry` → edit your own rule → My IP. Nothing else. |
| Router public IP | Every start, unless it has an Elastic IP | Elastic IP (step 2). Without one, everyone updates `HostName`. |
| Node public IPs | Every start | Not used. |
| Private IPs | Only when an instance is terminated and replaced | Update `cluster_config` (step 4) and everyone's laptop config. |
| Host keys | Only when an instance is replaced | `ssh-keygen -R <ip>`, then reconnect. |
| llama-server, tunnels | Gone after every stop/start | Rerun `run.sh` and the tunnels, or use step 6. |
| Auto-stop timer | Gone after every start | `sudo shutdown -h +360` after starting, if wanted. |
| New teammate | When they join | Their key into `authorized_keys` on all four machines (one loop from the router), and an `mm-entry` rule. |

## Sharing rules

1. **One experiment at a time.** Concurrent runs on the same nodes distort each other's measurements. Before starting, on the router: `cat ~/CLUSTER_BUSY; tmux ls`. Claim it with `echo "<name>: <experiment> until <time>" > ~/CLUSTER_BUSY`; delete the file when done.
2. **Your own clone on the router:** `git clone https://github.com/clv07/aws-mm-router ~/<name>/aws-mm-router`, then `ln -s ~/aws-mm-router/dataset ~/<name>/aws-mm-router/dataset`, so branches and results don't collide and the 4.3 GB dataset exists once.
3. **Standard server settings:** `CACHE_RAM=0 ./src/run.sh qwen` from `main` on every node. Restore them if your experiment changes them.
4. **Whoever finishes last stops all four instances.**
