# Phase 2: Traffic Generation

How we generate the request traffic for Phase 2, what is implemented in `src/tracegen.py`, and what still needs changes in the runner (`src/loadgen.py run`) or the router.

The Phase 1 routing benchmarks used one kind of trace: Poisson arrivals, one 4-modality mix, the 9 benchmark workloads, 15 minutes, one seed. Phase 2 must evaluate "steady and bursty arrivals and held-out workload mixtures", and the new router needs deadlines and, eventually, real token counts.

## Plan

| Addition | Why | How | Status |
| --- | --- | --- | --- |
| **Bursty arrivals** | Required by the project. A router queue only pays off when arrivals bunch up. | On/off process with the same mean rate as the steady trace, so steady and bursty are directly comparable. Parameters: burst rate multiplier, share of time in bursts, mean burst length. Plus a flash-crowd step and a ramp. | Done: `--arrivals onoff / step / ramp` |
| **Load sweep** | One 60% and one 90% run is too thin. | Rates from ~30% to ~110% of capacity, including periods above capacity (bursts, steps). | Done: `--load` with `--nodes` (capacity for the chosen mix) |
| **Held-out mixes** | Required: tune on one mix, test on others. | Training mix (`default`) plus held-out mixes (`chat-heavy`, `text-heavy`, `image-heavy`, `audio-heavy`, `uniform`). Decide which are held out before anyone tunes. | Done: `--mix` presets or a custom mix |
| **Deadlines in the trace** | The SLO-aware router needs them, and every policy must be judged against the same deadlines. | `deadline_s` per request = factor × the workload's fastest solo time on any node, factor per modality. | Done: `--slo`; needs the runner to log SLO met |
| **Continuous request sizes** | The new cost model uses actual prompt tokens; fixed sizes favour a lookup table like `ect`'s. | Text ~100–3,500 tokens, 1–4 images, 5–40 s audio, drawn from distributions. | To do: needs the dataset scripts to accept any size, and the runner to read `size` |
| **Multi-turn sessions** | The project asks for "the cost of sending accumulated context" and to "avoid moving an active session". | `session_id`, `turn`, think time between turns; each turn's prompt includes the earlier turns. Turns within a session run one after another; sessions arrive independently. | To do: needs runner support (send turn k+1 after turn k's answer) |
| **Real trace replay** | Realistic burstiness. | Replay arrival times from a public trace (e.g. Azure LLM inference traces, BurstGPT), rescaled to our cluster's rate. Check the current format of each trace before using it. | Done for arrival times: `--arrivals replay`; token counts to map later |
| **Seeds and length** | 85–122 requests per run gives a noisy p95. | 3 seeds per configuration, longer runs, and leave out a warm-up and the drain at the end when computing statistics. | Done: `--seed`, `--duration`; warm-up/drain trimming belongs in the analysis |

## Trace format

A trace is a CSV plus a JSON file with the same name. `loadgen.py run` replays the CSV unchanged: it reads `id`, `arrival_s`, `run`, `modality`, `size`, `input`, and ignores the rest.

| Column | Meaning |
| --- | --- |
| `id` | Request number, in arrival order |
| `arrival_s` | Arrival time in seconds from the start of the run |
| `run`, `modality`, `size` | Workload (one of the 9 in `benchmark.py` `RUNS`) |
| `input` | Which input of that modality's dataset; a new one per request (0 is the warm-up) |
| `slo_factor` | Deadline as a multiple of the workload's fastest solo time |
| `deadline_s` | Deadline relative to arrival, s (empty if no `--nodes` was given) |
| `burst` | 1 if the request arrived during a burst or step, for analysing latency in bursts |

The JSON file records every parameter, the mix, the rate, the capacity used, the rate segments (when bursts happened), and summary statistics: requests, mean rate, interarrival CV (1 = Poisson, above 1 = bursty), peak 60 s load relative to the mean, and requests per modality.

**Comparability.** Every arrival process is a time-rescaling of the same unit-rate Poisson stream, and workloads are drawn from a separate random stream in arrival order. With one seed, request *k* has the same workload and input in every trace (steady, bursty, any load); traces differ only in when requests arrive. Steady and bursty traces at the same load have exactly the same requests.

## Usage

```bash
# steady and bursty at 90% of capacity for the default mix (capacity and deadlines from nodes.json)
python src/tracegen.py make --arrivals poisson --load 0.9 --nodes nodes.json --seed 1
python src/tracegen.py make --arrivals onoff   --load 0.9 --nodes nodes.json --seed 1 \
       --burst-factor 3 --burst-fraction 0.2 --burst-length 60

# flash crowd: 60% load with a 3x step for 2 minutes from t = 300 s
python src/tracegen.py make --arrivals step --load 0.6 --nodes nodes.json --step-at 300 --step-length 120 --step-factor 3

# held-out mix
python src/tracegen.py make --arrivals poisson --load 0.9 --nodes nodes.json --mix image-heavy

# arrival times from a real trace, rescaled to 0.13 req/s
python src/tracegen.py make --arrivals replay --replay real.csv --replay-column timestamp --rate 0.13

# summary of any trace, including the Phase 1 ones
python src/tracegen.py describe traces/cluster-load90.csv traces/phase2/*.csv
```

Traces go to `traces/phase2/<arrivals>-<mix>-<load>-<duration>-<seed>.csv` unless `--out` is given. Without `--nodes`, use `--rate` (or `--load` with `--capacity`), and `deadline_s` is left empty.

**What depends on the cluster.** Arrival patterns, mixes and the request sequence don't depend on the hardware. Two things do: the rate for a given `--load` (capacity comes from the node profiles) and `deadline_s` (from the fastest node's solo times). If the cluster changes, regenerate with the new `nodes.json` and the same seeds: the requests stay the same, only rates and deadlines move.
