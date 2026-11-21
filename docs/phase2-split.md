# Phase 2: Splitting a Request Across Nodes (`src/split.py`)

A request can be processed in two places: the prompt (with its image or audio) on one node, and the answer on another. The KV cache built for the prompt is moved between them. This is **EP+D disaggregation**: encode + prefill on node A, decode on node B.

It can pay off because our nodes are good at different stages. In the Phase 1 profiles the c7a processes prompts fastest, while the m8g generates tokens about twice as fast (23.6 vs 11.9 tok/s). Whether the split wins depends on the size of the KV cache, the link between the nodes and the request's workload. `split.py` measures exactly that, and writes the numbers the routing policies use.

The experiment plan is in [phase2-experiment-plan.md](phase2-experiment-plan.md) (experiment 3).

## What we split, and what we don't

A request goes through three stages:

| Stage | What happens | What limits it |
| --- | --- | --- |
| **E**: encode | The image/audio encoder (the `mmproj` file) turns the media into embeddings, a few hundred "tokens" per image | CPU compute |
| **P**: prefill | The language model reads the whole prompt (text + media embeddings) in one pass and builds the **KV cache**, about 36.9 KB per prompt token for this model (36 layers × 2 KV heads × 128 dims × 2 bytes × K and V) | CPU compute |
| **D**: decode | The model generates the answer one token at a time, reading the KV cache at each step | Memory bandwidth |

- **We split at the P → D boundary (EP+D).** E and P run on node A; the KV cache moves; D runs on node B. llama-server supports this as it is, through slot save/restore.
- **Not E → P (E+PD).** That would move the encoder's embeddings instead of the KV cache. llama-server has no way to accept embeddings as input, so it would need changes to llama.cpp.
- **Not the model's layers.** Splitting layers across nodes (llama.cpp RPC) only helps when a model doesn't fit in one node's memory. For a 3B model it would just add a network hop to every token.

The model is whole on every node. Only one request at a time is split, and the router decides per request whether to split.

## How it works

llama-server has slots: each slot holds one request's KV cache. Started with `--slot-save-path DIR`, it accepts `POST /slots/<id>?action=save|restore|erase`. A saved slot file holds the KV cache **and the request's media chunks**, so after a restore the server knows the image or audio is already processed and does not encode it again.

`split_request(src_url, dst_url, body, filename, copy, cleanup)` runs one split:

| Step | Where | What happens | Why this way |
| --- | --- | --- | --- |
| 1 | A and B | `POST /slots/0?action=erase` | Both slots start empty, so an old cache can't make a step look fast |
| 2 | A | The request with `max_tokens: 1`, `cache_prompt: true`, `id_slot: 0` | A encodes and prefills the whole prompt, then stops after one token: the cheapest way to make llama-server build the KV cache. `id_slot` pins it to a known slot |
| 3 | A | `POST /slots/0?action=save` | A writes the slot to `<slot dir>/<filename>`; the reply's `n_written` is the KV size in bytes |
| 4 | A → B | The `copy` command | A real file copy over the network (transports below) |
| 5 | B | `POST /slots/0?action=restore` | B loads the KV cache into its slot 0 |
| 6 | B | The same request with `cache_prompt: true`, `id_slot: 0` | B finds the whole prompt already in its cache, processes about 1 new token and generates the answer |
| – | A and B | The `cleanup` command | Deletes the slot file on both nodes, even if a step failed |

Every step is timed. The split's total is `split_s` = prefill on A + save + copy + restore + B's continuation.

**Transports** (`--transport`), for step 4:

| Transport | Command | Use |
| --- | --- | --- |
| `direct` | `ssh <A> scp <file> <B's peer>:<slot dir>/<file>` | The real setup: node to node inside AWS. Needs SSH from A to B ([cluster-access.md](cluster-access.md)) and `peer` in the nodes file |
| `relay` (default) | `scp -3 <A>:<file> <B>:<file>` | Through the machine running `split.py`. Crosses the network twice, so only for when nodes can't reach each other |
| `local` | `cp` | Two servers on one machine (lab-server tests) |

## How we know the split really happened

llama-server reports what it actually computed for each request, so every split row checks itself:

| Field | Expected | If not |
| --- | --- | --- |
| `dst_prompt_n` | ≈ 1 (B processed one new token) | Close to the prompt length = B recomputed the prompt. Prints `WARNING: dst recomputed the prompt` |
| `src_prompt_n` | = the prompt length | Much smaller = A hit a prompt cache. Prints a warning; start the servers with `CACHE_RAM=0` |
| `dst_encoding_ms` | 0 (patched llama-server) | > 0 = B encoded the image/audio again. Prints `WARNING: dst encoded the image/audio again` |
| `src_encoding_ms` | The encoder's time inside A's prompt time | – |
| `same_prefix_chars` | Close to `answer_chars` | The split answer drifts from the one generated alone. Thread counts differ between nodes, so small differences are expected; this is a sanity check, not a strict test |

`encoding_ms` comes from the stage-profiling patch that `run.sh` builds into llama-server; on an unpatched server it is empty.

## Running it

**Servers:** every node runs `CACHE_RAM=0 ./src/run.sh qwen`. That gives the patched llama-server with `--slot-save-path ~/slots` and no prompt cache in RAM. All nodes must serve the same model and quantization (a KV cache is specific to the model); `split.py` refuses otherwise.

**Nodes file:** the routing `nodes.json` format, plus two optional fields per node.
```json
{
  "c7a.2xlarge": {"url": "http://127.0.0.1:18080", "ssh": "node-c7a", "peer": "node-c7a"},
  "c7i.2xlarge": {"url": "http://127.0.0.1:18081", "ssh": "node-c7i", "peer": "node-c7i"},
  "m8g.xlarge":  {"url": "http://127.0.0.1:18082", "ssh": "node-m8g", "peer": "node-m8g"}
}
```
- `url`: the node's llama-server, as the machine running `split.py` reaches it (here, the router's SSH tunnels).
- `ssh`: how that machine reaches the node, to run the copy and cleanup commands.
- `peer`: how the *other nodes* reach this one, for `direct` copies. With the shared SSH config in [cluster-access.md](cluster-access.md) it's the same alias.
- `slot_dir` (optional): the node's `--slot-save-path`, default `slots` (i.e. `~/slots`).

**Commands**, on the router:
```bash
# quick check, ~5 min: x86 -> Arm restore, copy and restore; no WARNING lines expected
python src/split.py run --nodes nodes.aws.json --label aws-check --runs 00 04 --repeats 1 \
  --pairs c7a.2xlarge:m8g.xlarge --transport direct

# full run, ~1.5 h
python src/split.py run --nodes nodes.aws.json --label aws-p2 --runs 00 02 03 04 06 07 08 --transport direct \
  --pairs c7a.2xlarge:m8g.xlarge c7a.2xlarge:c7i.2xlarge c7i.2xlarge:m8g.xlarge
python src/split.py summarize results/split/aws-p2
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--runs` | all 9 | Workloads (`benchmark.py` `RUNS`) |
| `--pairs` | every ordered pair | `src:dst` pairs to split over |
| `--only` | all nodes | Use only these nodes from the file |
| `--repeats` | 3 | Requests per workload, alone and per pair |
| `--transport` | `relay` | `direct`, `relay` or `local` |
| `--slot` | 0 | The slot used on both nodes |

`run` has two phases, one request at a time, so nothing else runs alongside:
1. **Alone:** every workload start to finish on every node (non-streamed, nothing cached). This is the no-split baseline, measured on the same servers in the same hour.
2. **Split:** every pair A → B and every workload through `split_request()`.

Each request uses the same inputs and prompts as `benchmark.py`, with fixed output (128 tokens). Every node gets one unrecorded warm-up request per modality first. Don't run it while other experiments use the nodes.

## Output

Results go to `results/split/<label>/`.

| File | One row per | Contents |
| --- | --- | --- |
| `alone.csv` | request on one node | `e2e_s`, `prompt_s`, `encoding_ms`, `decode_s`, `decode_tok_s`, prompt tokens, request and response bytes |
| `split.csv` | split request | Every step: `src_prefill_s`, `src_prompt_s`, `src_encoding_ms`, `save_s`, `kv_bytes`, `copy_s`, `restore_s`, `dst_continue_s`, `dst_prompt_n`, `dst_encoding_ms`, `dst_decode_s`, `dst_decode_tok_s`, `split_s`, and the answer check |
| `costs.json` (from `summarize`) | – | Medians per node and workload alone; per pair: save + restore time, the copy link fitted as latency + MB / speed, and per workload the split time, prefill, continuation, KV MB and copy time; KV bytes per prompt token |

`summarize` also prints a table of seconds per request: each node alone, each split, and the best option per workload. It repeats the best option with the copy replaced by one at other link speeds (`--mb-s`, default 125 / 12.5 / 2.5 MB/s, i.e. 1 Gbit, 100 Mbit and 20 Mbit). That shows the break-even bandwidth below which splitting stops paying.

## How a router uses it

```python
from split import SplitCosts, split_request
c = SplitCosts('results/split/aws-p2/costs.json')
c.alone('m8g.xlarge', '04')                               # s, the request all on one node
c.split('c7a.2xlarge', 'm8g.xlarge', '04')                # s, measured split
c.split('c7a.2xlarge', 'm8g.xlarge', '04', mb_s=12.5)     # s, the same over a 100 Mbit/s link
c.options('04')                                           # every option, fastest first
```

All of these are times on idle nodes; queueing and interference are the router's to add. An `ect + split` policy estimates the finish time of every option (each node alone, each split A → B) for an arriving request and picks the earliest. To carry out a split it calls `split_request()` with the two nodes' URLs and the transport's copy and cleanup commands. Not yet written: the policy, and the split path in `main.py run`.

## Results so far (lab servers, not AWS)

- **s41** (two llama-servers on one machine, 8 and 4 cores, `local` copies): the split works for every modality, with `dst_prompt_n` = 1. Save + copy + restore took under 0.1 s. The split never won there, because the 8-core server was faster at both stages. That's expected: a split only pays when the nodes are good at different stages.
- **s26** (patched llama-server, 12 and 6 cores, `local` copies on a hard disk):

  | Workload | Prompt tokens | KV size | Encode on A | Save | Copy | Restore | `dst_prompt_n` | `dst_encoding_ms` |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | 1 image | 426 | 15.7 MB | 2.92 s | 0.01 s | 0.32 s | 1.86 s | 1 | 0 |
  | 5 s audio | 775 | 28.6 MB | 3.31 s | 0.01 s | 0.70 s | 0.91 s | 1 | 0 |

  KV size is 36.9 KB per prompt token, as computed above. Restores read from a hard disk there; the AWS nodes use SSDs.

**The KV cache is much bigger than the input.** One image is a few hundred KB of base64 in the request, but about 16 MB of KV cache after prefill. A split moves far more data between nodes than the request itself moved from the router.

## Limits

- **One request at a time, on idle nodes.** This measures what a split costs, not how splits behave under load. Copies competing with request traffic on a node's network link, and save/restore competing with other requests for CPU, show up only in routing runs with a split policy.
- **x86 → Arm** (c7a → m8g) restore is not tested yet; the AWS quick check is the first test.
- **TTFT of a split request** is counted at B's first token, after save, copy and restore. A already produced one token, but that is not what the user sees streaming.
- **The slot file goes through the disk** on both nodes (save writes it, restore reads it). An in-memory transfer would be faster but needs changes to llama.cpp.
- **Encode and prefill stay together on A.** Moving only the encoder's work would need llama-server to take embeddings as input.
