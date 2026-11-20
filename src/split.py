#!/usr/bin/env python3
"""Split benchmark: is a request faster when its prompt is processed on one node and its answer
generated on another, with its KV cache moved in between?

  python src/split.py run --nodes split-nodes.json --label aws-p3            # from the router
  python src/split.py summarize results/split/aws-p3                          # -> costs.json for the routers

`run` has two phases, one request at a time, so nothing interferes:
  1. alone: every workload on every node, start to finish (the "no split" options);
  2. split: for every pair of nodes A -> B and every workload, the prompt (with image/audio
     encoding) is processed on A, A's slot is saved to a file, the file is copied to B, B restores
     it and generates the answer. Each step is timed; dst_prompt_n, the prompt tokens B still had to
     process, shows that B used the restored KV (close to 0) instead of recomputing.
`summarize` writes costs.json: per node and workload the time alone, prompt time and decode time; per
pair the KV size, save + restore time, copy speed and the measured split time. SplitCosts reads it,
so a router can compare "run it all on node n" with "split A -> B", also at other link speeds.

Servers: started by src/run.sh (it passes --slot-save-path ~/slots), and with CACHE_RAM=0 for this
benchmark, so llama-server's prompt cache in RAM cannot turn a repeated prompt into a cache hit.

The nodes file is the routing one (nodes.json) with two optional fields per node:
  "slot_dir": where its llama-server saves slots (default "slots", i.e. ~/slots over SSH)
  "peer":     how the other nodes reach it over SSH, for --transport direct
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import requests

from benchmark import OUTPUT_TOKENS, RUNS, TIMEOUT_SECONDS, append_rows, median, payload, server_info

OUT_DIR = Path("results/split")
ALONE_FIELDS = ["run_id", "timestamp_utc", "label", "node", "model", "run", "modality", "size", "input",
                "repeat", "status", "error", "request_bytes", "response_bytes", "e2e_s", "prompt_tokens",
                "prompt_n", "prompt_s", "encoding_ms", "decode_s", "decode_tok_s", "output_tokens"]
SPLIT_FIELDS = ["run_id", "timestamp_utc", "label", "src", "dst", "model", "run", "modality", "size",
                "input", "repeat", "status", "error", "prompt_tokens", "src_prompt_n", "src_prefill_s",
                "src_prompt_s", "src_encoding_ms", "save_s", "kv_bytes", "copy_s", "restore_s",
                "dst_continue_s", "dst_prompt_n", "dst_encoding_ms", "dst_decode_s", "dst_decode_tok_s", "split_s", "same_prefix_chars",
                "answer_chars"]
ALONE_CSV, SPLIT_CSV, COSTS_JSON = "alone.csv", "split.csv", "costs.json"
TRANSPORTS = {
    # copy the slot file from src's slot_dir to dst's, then delete both copies
    'relay':  ("scp -q -3 {src_ssh}:{src_dir}/{file} {dst_ssh}:{dst_dir}/{file}",
               "ssh {src_ssh} rm -f {src_dir}/{file}; ssh {dst_ssh} rm -f {dst_dir}/{file}"),
    'direct': ("ssh {src_ssh} scp -q {src_dir}/{file} {dst_peer}:{dst_dir}/{file}",
               "ssh {src_ssh} rm -f {src_dir}/{file}; ssh {dst_ssh} rm -f {dst_dir}/{file}"),
    'local':  ("cp {src_dir}/{file} {dst_dir}/{file}", "rm -f {src_dir}/{file} {dst_dir}/{file}"),
}


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------- one split request

def complete(url, body):
    """One non-streamed chat completion: (wall seconds, response JSON, response bytes)."""
    t = time.perf_counter()
    r = requests.post(url + '/v1/chat/completions', json=body, timeout=(15, TIMEOUT_SECONDS))
    r.raise_for_status()
    return time.perf_counter() - t, r.json(), len(r.content)


def slot_action(url, slot, action, filename=None):
    t = time.perf_counter()
    r = requests.post(f'{url}/slots/{slot}?action={action}',
                      json={'filename': filename} if filename else {}, timeout=(15, TIMEOUT_SECONDS))
    r.raise_for_status()
    return time.perf_counter() - t, r.json()


def shell(command):
    t = time.perf_counter()
    subprocess.run(command, shell=True, check=True)
    return time.perf_counter() - t


def split_request(src_url, dst_url, body, filename, copy, cleanup='', slot=0):
    """Process body's prompt on src, move the slot's KV to dst (copy: shell command), and generate
    the answer there. body is a non-streamed chat request; returns the timings and dst's response.
    Usable from a router: main.py can call it for a request it decides to split."""
    for url in (src_url, dst_url):
        slot_action(url, slot, 'erase')
    first_body = {**body, 'max_tokens': 1, 'cache_prompt': True, 'id_slot': slot}
    out = {}
    out['src_prefill_s'], first, _ = complete(src_url, first_body)
    out['src_prompt_n'] = first.get('timings', {}).get('prompt_n')
    out['src_prompt_s'] = first.get('timings', {}).get('prompt_ms', 0) / 1000
    # Encoder time inside prompt_s (patched llama-server only); dst's should be 0 after a restore.
    out['src_encoding_ms'] = first.get('timings', {}).get('encoding_ms')
    out['prompt_tokens'] = first.get('usage', {}).get('prompt_tokens')
    out['save_s'], saved = slot_action(src_url, slot, 'save', filename)
    out['kv_bytes'] = saved['n_written']
    try:
        out['copy_s'] = shell(copy)
        out['restore_s'], _ = slot_action(dst_url, slot, 'restore', filename)
    finally:
        if cleanup:
            subprocess.run(cleanup, shell=True)
    out['dst_continue_s'], rest, _ = complete(dst_url, {**body, 'cache_prompt': True, 'id_slot': slot})
    timings = rest.get('timings', {})
    out.update({'dst_prompt_n': timings.get('prompt_n'), 'dst_encoding_ms': timings.get('encoding_ms'),
                'dst_decode_s': timings.get('predicted_ms', 0) / 1000,
                'dst_decode_tok_s': timings.get('predicted_per_second')})
    out['split_s'] = sum(out[k] for k in ('src_prefill_s', 'save_s', 'copy_s', 'restore_s', 'dst_continue_s'))
    return out, rest


# ---------------------------------------------------------------- benchmark

def load_nodes(path):
    nodes = json.loads(Path(path).read_text())
    for name, cfg in nodes.items():
        if 'url' not in cfg:
            raise SystemExit(f'{path}: node {name!r} has no "url"')
        cfg['url'] = cfg['url'].rstrip('/')
        cfg.setdefault('slot_dir', 'slots')
    return nodes


def answer(response):
    return response['choices'][0]['message'].get('content') or ''


def run(args):
    from main import load_datasets
    nodes = load_nodes(args.nodes)
    if args.only:
        nodes = {n: nodes[n] for n in args.only}
    pairs = ([tuple(p.split(':')) for p in args.pairs] if args.pairs
             else [(a, b) for a in nodes for b in nodes if a != b])
    for a, b in pairs:
        if a not in nodes or b not in nodes:
            raise SystemExit(f'pair {a}:{b}: both must be in {args.nodes}')
    models = {n: server_info(cfg['url'])[0] for n, cfg in nodes.items()}
    if len(set(models.values())) != 1:
        raise SystemExit(f'the nodes serve different models: {models}; the KV cache is per model')
    model = next(iter(models.values()))
    copy, cleanup = TRANSPORTS[args.transport]
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = OUT_DIR / args.label
    datasets = load_datasets({RUNS[r][0] for r in args.runs})
    print(f'run_id={run_id} model={model} nodes={list(nodes)} pairs={pairs} transport={args.transport} '
          f'-> {out_dir}', flush=True)

    def body(run, index, max_tokens=OUTPUT_TOKENS):
        modality, setting = RUNS[run]
        b = payload(datasets[modality], modality, setting, index, 'fixed')
        b.pop('stream_options', None)
        b.update({'stream': False, 'max_tokens': max_tokens, 'cache_prompt': False})
        return b

    labels = {'run_id': run_id, 'label': args.label, 'model': model}
    for n, cfg in nodes.items():   # unrecorded warm-up, one request per modality (input 0)
        for m in {RUNS[r][0] for r in args.runs}:
            first = next(r for r in args.runs if RUNS[r][0] == m)
            complete(cfg['url'], body(first, 0, 8))
        print(f'  warmed up {n}', flush=True)

    answers = {}
    print('[alone]', flush=True)
    for n, cfg in nodes.items():
        for run in args.runs:
            modality, setting = RUNS[run]
            for k in range(args.repeats):
                row = {**labels, 'node': n, 'run': run, 'modality': modality, 'size': setting,
                       'input': 1 + k, 'repeat': k + 1, 'status': 'ok', 'error': ''}
                try:
                    b = body(run, 1 + k)
                    row['request_bytes'] = len(json.dumps(b))
                    row['e2e_s'], resp, row['response_bytes'] = complete(cfg['url'], b)
                    t = resp.get('timings', {})
                    row.update({'prompt_tokens': resp.get('usage', {}).get('prompt_tokens'),
                                'prompt_n': t.get('prompt_n'), 'prompt_s': t.get('prompt_ms', 0) / 1000,
                                'encoding_ms': t.get('encoding_ms'),
                                'decode_s': t.get('predicted_ms', 0) / 1000,
                                'decode_tok_s': t.get('predicted_per_second'),
                                'output_tokens': t.get('predicted_n')})
                    answers[(n, run, 1 + k)] = answer(resp)
                except (requests.RequestException, KeyError) as exc:
                    row.update({'status': 'error', 'error': str(exc)[:200]})
                row['timestamp_utc'] = now_utc()
                append_rows(out_dir / ALONE_CSV, ALONE_FIELDS, [row])
                print(f"  {n} {run} {modality}={setting} #{k + 1}: " + (row['error'] or
                      f"{row['e2e_s']:.1f}s (prompt {row['prompt_s']:.1f} + decode {row['decode_s']:.1f})"),
                      flush=True)

    print('[split]', flush=True)
    for a, b in pairs:
        fmt = {'src_ssh': nodes[a].get('ssh'), 'dst_ssh': nodes[b].get('ssh'), 'dst_peer': nodes[b].get('peer'),
               'src_dir': nodes[a]['slot_dir'], 'dst_dir': nodes[b]['slot_dir']}
        for run in args.runs:
            modality, setting = RUNS[run]
            for k in range(args.repeats):
                filename = f'split-{run_id}-{run}-{1 + k}.bin'
                row = {**labels, 'src': a, 'dst': b, 'run': run, 'modality': modality, 'size': setting,
                       'input': 1 + k, 'repeat': k + 1, 'status': 'ok', 'error': ''}
                try:
                    timings, resp = split_request(nodes[a]['url'], nodes[b]['url'], body(run, 1 + k), filename,
                                                  copy.format(file=filename, **fmt),
                                                  cleanup.format(file=filename, **fmt), args.slot)
                    row.update(timings)
                    alone = answers.get((a, run, 1 + k), '')
                    got = answer(resp)
                    row['same_prefix_chars'] = len(os.path.commonprefix([alone, got]))
                    row['answer_chars'] = len(alone)
                except (requests.RequestException, subprocess.CalledProcessError, KeyError) as exc:
                    row.update({'status': 'error', 'error': str(exc)[:200]})
                row['timestamp_utc'] = now_utc()
                append_rows(out_dir / SPLIT_CSV, SPLIT_FIELDS, [row])
                if row['error']:
                    print(f'  {a}->{b} {run} #{k + 1}: ERROR {row["error"]}', flush=True)
                    continue
                note = ('  WARNING: dst recomputed the prompt'
                        if row['dst_prompt_n'] and row['prompt_tokens'] and row['dst_prompt_n'] > 0.5 * row['prompt_tokens']
                        else '')
                note += ('  WARNING: src hit a prompt cache (start the servers with CACHE_RAM=0)'
                         if row['src_prompt_n'] is not None and row['prompt_tokens'] and
                         row['src_prompt_n'] < 0.5 * row['prompt_tokens'] else '')
                note += ('  WARNING: dst encoded the image/audio again'
                         if row['dst_encoding_ms'] else '')
                print(f"  {a}->{b} {run} {modality}={setting} #{k + 1}: {row['split_s']:.1f}s = prompt "
                      f"{row['src_prefill_s']:.1f} + save {row['save_s']:.2f} + copy {row['copy_s']:.2f} "
                      f"+ restore {row['restore_s']:.2f} + answer {row['dst_continue_s']:.1f}  "
                      f"KV {row['kv_bytes'] / 1e6:.1f} MB{note}", flush=True)
    print(f'appended results to {out_dir}; next: python src/split.py summarize {out_dir}')


# ---------------------------------------------------------------- cost model for the routers

def read_csv(path):
    import csv
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fnum(rows, field):
    return [float(r[field]) for r in rows if r.get(field) not in (None, '')]


def fit_link(rows):
    """copy_s = latency + MB / speed over the copies of one pair; (latency s, MB/s)."""
    pts = [(float(r['kv_bytes']) / 1e6, float(r['copy_s'])) for r in rows]
    if len({x for x, _ in pts}) < 2:
        mb, s = median([x for x, _ in pts]), median([y for _, y in pts])
        return 0.0, mb / s
    mx, my = sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    if slope <= 0:   # copies too fast to see a size effect (e.g. same machine): latency only
        return median([y for _, y in pts]), None
    return max(0.0, my - slope * mx), 1 / slope


def ms_to_s(ms):
    return ms / 1000 if ms is not None else None


def summarize(args):
    folder = Path(args.folder)
    alone = [r for r in read_csv(folder / ALONE_CSV) if r['status'] == 'ok']
    split = [r for r in read_csv(folder / SPLIT_CSV) if r['status'] == 'ok']
    costs = {'label': folder.name, 'model': (alone or split)[0]['model'], 'output_tokens': OUTPUT_TOKENS,
             'nodes': {}, 'pairs': {}}
    groups = defaultdict(list)
    for r in alone:
        groups[(r['node'], r['run'])].append(r)
    for (n, run), rows in sorted(groups.items()):
        costs['nodes'].setdefault(n, {})[run] = {
            'e2e_s': median(fnum(rows, 'e2e_s')), 'prompt_s': median(fnum(rows, 'prompt_s')),
            'decode_s': median(fnum(rows, 'decode_s')), 'prompt_tokens': median(fnum(rows, 'prompt_tokens')),
            # encoder share of prompt_s; None unless the servers run the stage-profiling patch
            'encode_s': ms_to_s(median(fnum(rows, 'encoding_ms'))),
            'request_bytes': median(fnum(rows, 'request_bytes')), 'requests': len(rows)}
    kv_per_token = median([float(r['kv_bytes']) / float(r['prompt_tokens']) for r in split if r['prompt_tokens']])
    costs['kv_bytes_per_prompt_token'] = kv_per_token
    by_pair = defaultdict(list)
    for r in split:
        by_pair[(r['src'], r['dst'])].append(r)
    for (a, b), rows in sorted(by_pair.items()):
        latency, speed = fit_link(rows)
        entry = {'save_restore_s': median([float(r['save_s']) + float(r['restore_s']) for r in rows]),
                 'copy_latency_s': latency, 'copy_mb_s': speed, 'runs': {}}
        per_run = defaultdict(list)
        for r in rows:
            per_run[r['run']].append(r)
        for run, rs in sorted(per_run.items()):
            entry['runs'][run] = {'split_s': median(fnum(rs, 'split_s')),
                                  'prefill_s': median(fnum(rs, 'src_prefill_s')),
                                  'src_encode_s': ms_to_s(median(fnum(rs, 'src_encoding_ms'))),
                                  'continue_s': median(fnum(rs, 'dst_continue_s')),
                                  'kv_mb': median(fnum(rs, 'kv_bytes')) / 1e6,
                                  'copy_s': median(fnum(rs, 'copy_s')),
                                  'dst_prompt_n': median(fnum(rs, 'dst_prompt_n')), 'requests': len(rs)}
        costs['pairs'][f'{a}->{b}'] = entry
    (folder / COSTS_JSON).write_text(json.dumps(costs, indent=1))
    print_table(SplitCosts(folder / COSTS_JSON), args.mb_s)
    print(f'\nwrote {folder / COSTS_JSON}')


class SplitCosts:
    """Reads costs.json. For a router:
        c = SplitCosts('results/split/aws-p3/costs.json')
        c.alone('m8g.xlarge', '04')                      # s, the request all on one node, nothing else running
        c.split('c7a.2xlarge', 'm8g.xlarge', '04')       # s, measured split
        c.split('c7a.2xlarge', 'm8g.xlarge', '04', mb_s=12.5)   # s, the same over a 100 Mbit/s link
        c.options('04')                                  # every option, fastest first
    All times are for one request on idle nodes; queueing and interference are the router's to add."""
    def __init__(self, path):
        self.c = json.loads(Path(path).read_text())

    def alone(self, node, run):
        return self.c['nodes'][node][run]['e2e_s']

    def split(self, src, dst, run, mb_s=None):
        p = self.c['pairs'][f'{src}->{dst}']
        r = p['runs'][run]
        if mb_s is None:
            return r['split_s']
        # Measured split with the copy replaced by one at mb_s (MB/s).
        return r['split_s'] - r['copy_s'] + p['copy_latency_s'] + r['kv_mb'] / mb_s

    def options(self, run, mb_s=None):
        opts = [(n, self.alone(n, run)) for n in self.c['nodes'] if run in self.c['nodes'][n]]
        for pair, p in self.c['pairs'].items():
            if run in p['runs']:
                a, b = pair.split('->')
                opts.append((pair, self.split(a, b, run, mb_s)))
        return sorted(opts, key=lambda o: o[1])


def print_table(c, mb_s_list):
    nodes, pairs = list(c.c['nodes']), list(c.c['pairs'])
    runs = sorted({r for n in nodes for r in c.c['nodes'][n]})
    name = lambda r: f'{r} {RUNS[r][0]}{"" if RUNS[r][0] == "chat" else RUNS[r][1]}'
    print('\nSeconds per request, one at a time: each node alone, then each split src->dst (measured)')
    head = nodes + pairs
    print(f'  {"":<12}' + ''.join(f'{h[:22]:>24}' for h in head) + f'{"best":>26}')
    for r in runs:
        cells = [c.c['nodes'][n].get(r, {}).get('e2e_s') for n in nodes] + \
                [c.c['pairs'][p]['runs'].get(r, {}).get('split_s') for p in pairs]
        best = c.options(r)[0]
        print(f'  {name(r):<12}' + ''.join(f'{v:>24.1f}' if v is not None else f'{"-":>24}' for v in cells)
              + f'{best[0][:22]:>26}')
    for p in pairs:
        e = c.c['pairs'][p]
        speed = f'{e["copy_mb_s"]:.0f} MB/s' if e['copy_mb_s'] else 'too fast to fit'
        print(f'  {p}: save + restore {e["save_restore_s"]:.2f} s, copy {e["copy_latency_s"]:.2f} s + MB / {speed}')
    print(f'  KV cache: {c.c["kv_bytes_per_prompt_token"] / 1e3:.1f} KB per prompt token')
    for mb_s in mb_s_list:
        print(f'\nIf the copy ran at {mb_s:g} MB/s ({mb_s * 8:g} Mbit/s), best option per workload:')
        for r in runs:
            opts = c.options(r, mb_s)
            print(f'  {name(r):<12} {opts[0][0]:<34} {opts[0][1]:6.1f} s'
                  + (f'   (next: {opts[1][0]} {opts[1][1]:.1f} s)' if len(opts) > 1 else ''))


def main():
    parser = argparse.ArgumentParser(description='Split benchmark: prompt on one node, answer on another.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='measure every workload alone on every node, then split over every pair')
    p.add_argument('--nodes', required=True, help='nodes file (url, ssh, optional slot_dir and peer per node)')
    p.add_argument('--label', required=True, help='results folder name, e.g. aws-p3')
    p.add_argument('--only', nargs='+', help='use only these nodes from the file')
    p.add_argument('--pairs', nargs='+', help='src:dst pairs (default: every ordered pair)')
    p.add_argument('--runs', nargs='+', choices=RUNS.keys(), default=list(RUNS), help='workloads (default: all 9)')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--transport', choices=TRANSPORTS, default='relay',
                   help='how slot files are copied: relay = scp through this machine (default), '
                        'direct = node to node (needs "peer" and SSH between the nodes), local = cp')
    p.add_argument('--slot', type=int, default=0)
    p.add_argument('--run-id')
    p.set_defaults(func=run)
    s = sub.add_parser('summarize', help='costs.json and the comparison table')
    s.add_argument('folder')
    s.add_argument('--mb-s', type=float, nargs='*', default=[125, 12.5, 2.5],
                   help='also show the best option if copies ran at these MB/s (default 1 Gbit, 100 Mbit, 20 Mbit)')
    s.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
