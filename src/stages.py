#!/usr/bin/env python3
"""Stage breakdown: where an image or audio request's time goes on this node (encoding, prompt
processing, decoding), for the 5 image and audio workloads of benchmark.py. Chat and text have no
encoder; their prompt and decode times are in the benchmark.py profiles (ttft_s, decode_tok_s).

  python src/stages.py run --node c7a.2xlarge-p3          # on the node, in tmux
  python src/stages.py summarize results/stages/c7a.2xlarge-p3__Qwen2.5-Omni-3B-Q8_0

llama-server reports one prompt time that includes encoding, so this runs each request through
llama-mtmd-cli (built by src/run.sh), which logs the stages separately:
  encode_s         the vision/audio encoder ("mtmd batch encoding done in X ms")
  media_prefill_s  the language model over the image/audio tokens ("image decoded ... in X ms")
  text_prefill_s   the language model over the text tokens (prompt time - the two above)
  decode_s         generating the answer (OUTPUT_TOKENS tokens, end of answer ignored)
Same model files, inputs (input 1, 2, ... of each dataset) and prompts as benchmark.py. Each request
is a fresh process, so nothing is cached; loading the model is not part of any stage. Run it with
llama-server stopped (or idle), as it uses the same cores.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import time

from benchmark import OUTPUT_TOKENS, RUNS, append_rows, instance_type, median

OUT_DIR = Path("results/stages")
MEDIA_RUNS = [r for r, (m, _) in RUNS.items() if m in ('image', 'audio')]
HOME = Path.home()
FIELDS = ["run_id", "timestamp_utc", "node", "model", "threads", "run", "modality", "size", "input",
          "repeat", "status", "error", "prompt_tokens", "media_tokens", "encode_s", "media_prefill_s",
          "text_prefill_s", "prompt_s", "decode_s", "output_tokens", "decode_tok_s", "wall_s"]
SUMMARY_FIELDS = ["node", "model", "run", "modality", "size", "requests", "prompt_tokens", "media_tokens",
                  "encode_s", "media_prefill_s", "text_prefill_s", "prompt_s", "decode_s",
                  "encode_share_of_prompt", "encode_share_of_total"]
PATTERNS = {
    'encode': re.compile(r'mtmd batch encoding done in (\d+) ms'),
    'media_prefill': re.compile(r'(?:image|audio) decoded \(batch \d+/\d+\) in (\d+) ms'),
    'media_tokens': re.compile(r'decoding (?:image|audio) batch \d+/\d+, n_tokens_batch = (\d+)'),
    'prompt': re.compile(r'prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens'),
    'decode': re.compile(r'print:\s+eval time =\s+([\d.]+) ms /\s+(\d+) runs'),
}


def load_inputs(runs):
    """(text prompt, media paths) per (run, input index), from the same datasets and prompts as benchmark.py."""
    from main import load_datasets
    datasets = load_datasets({RUNS[r][0] for r in runs})

    def inputs(run, index):
        modality, setting = RUNS[run]
        ds = datasets[modality]
        parts = ds.messages(setting, start=index)[0]['content']
        text = '\n'.join(p['text'] for p in parts if p['type'] == 'text')
        if modality == 'image':
            media = [ds.files[(index + i) % len(ds.files)] for i in range(setting)]
        else:
            media = [ds.clip(setting, index)]
        return text, media
    return inputs


def parse(log):
    found = {k: p.findall(log) for k, p in PATTERNS.items()}
    if not found['prompt'] or not found['decode']:
        raise RuntimeError('no timing lines in the llama-mtmd-cli output')
    prompt_ms, prompt_tokens = float(found['prompt'][-1][0]), int(found['prompt'][-1][1])
    decode_ms, runs = float(found['decode'][-1][0]), int(found['decode'][-1][1])
    encode = sum(int(x) for x in found['encode']) / 1000
    media_prefill = sum(int(x) for x in found['media_prefill']) / 1000
    return {'prompt_tokens': prompt_tokens, 'media_tokens': sum(int(x) for x in found['media_tokens']),
            'encode_s': encode, 'media_prefill_s': media_prefill,
            'text_prefill_s': max(0.0, prompt_ms / 1000 - encode - media_prefill),
            'prompt_s': prompt_ms / 1000, 'decode_s': decode_ms / 1000,
            # the first output token is sampled from the prompt pass, so eval runs = tokens - 1
            'output_tokens': runs + 1, 'decode_tok_s': runs / (decode_ms / 1000) if decode_ms else None}


def run(args):
    binary, model, mmproj = Path(args.bin), Path(args.model), Path(args.mmproj)
    for f in (binary, model, mmproj):
        if not f.exists():
            raise SystemExit(f'{f} not found (run ./src/run.sh qwen first, or pass --bin/--model/--mmproj)')
    node = args.node or instance_type()
    model_name = model.stem
    out = OUT_DIR / f'{node}__{model_name}' / 'stages.csv'
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    inputs = load_inputs(args.runs)
    print(f'run_id={run_id} node={node} model={model_name} threads={args.threads} -> {out}', flush=True)
    base = [str(binary), '-m', str(model), '--mmproj', str(mmproj), '-t', str(args.threads),
            '-c', str(args.ctx), '-n', str(OUTPUT_TOKENS), '--temp', '0', '--ignore-eos', '-lv', '4']
    for run_ in args.runs:
        modality, setting = RUNS[run_]
        for k in range(args.warmups + args.repeats):
            index = 0 if k < args.warmups else k - args.warmups + 1   # input 0 = warm-up, as in benchmark.py
            text, media = inputs(run_, index)
            command = base + ['-p', text, '--image' if modality == 'image' else '--audio',
                              ','.join(str(m) for m in media)]
            t = time.perf_counter()
            proc = subprocess.run(command, capture_output=True, text=True, errors='replace')
            wall = time.perf_counter() - t
            if k < args.warmups:
                continue
            row = {'run_id': run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   'node': node, 'model': model_name, 'threads': args.threads, 'run': run_, 'modality': modality,
                   'size': setting, 'input': index, 'repeat': k - args.warmups + 1, 'status': 'ok', 'error': '',
                   'wall_s': wall}
            try:
                if proc.returncode != 0:
                    raise RuntimeError(f'exit {proc.returncode}: {proc.stderr.strip().splitlines()[-1:]}')
                row.update(parse(proc.stdout + proc.stderr))
            except RuntimeError as exc:
                row.update({'status': 'error', 'error': str(exc)[:200]})
            append_rows(out, FIELDS, [row])
            print(f"  {run_} {modality}={setting} #{row['repeat']}: " + (row['error'] or
                  f"encode {row['encode_s']:.2f} + media prefill {row['media_prefill_s']:.2f} + text prefill "
                  f"{row['text_prefill_s']:.2f} + decode {row['decode_s']:.2f} s ({row['prompt_tokens']} prompt "
                  f"tokens, {row['media_tokens']} from media)"), flush=True)
    print(f'appended results to {out}; next: python src/stages.py summarize {out.parent}')


def summarize(args):
    folder = Path(args.folder)
    with open(folder / 'stages.csv', newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['status'] == 'ok']
    groups = defaultdict(list)
    for r in rows:
        groups[r['run']].append(r)
    summary = []
    for run_, rs in sorted(groups.items()):
        m = lambda field: median([float(r[field]) for r in rs])
        prompt, decode = m('prompt_s'), m('decode_s')
        summary.append({'node': rs[0]['node'], 'model': rs[0]['model'], 'run': run_, 'modality': rs[0]['modality'],
                        'size': rs[0]['size'], 'requests': len(rs), 'prompt_tokens': m('prompt_tokens'),
                        'media_tokens': m('media_tokens'), 'encode_s': m('encode_s'),
                        'media_prefill_s': m('media_prefill_s'), 'text_prefill_s': m('text_prefill_s'),
                        'prompt_s': prompt, 'decode_s': decode,
                        'encode_share_of_prompt': m('encode_s') / prompt if prompt else None,
                        'encode_share_of_total': m('encode_s') / (prompt + decode) if prompt + decode else None})
    with (folder / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    print(f'{summary[0]["node"] if summary else folder.name}: median seconds per request')
    print(f'  {"workload":<13}{"encode":>8}{"media pf":>10}{"text pf":>9}{"decode":>8}{"total":>8}'
          f'{"encode/prompt":>15}{"encode/total":>14}')
    for s in summary:
        label = f'{s["run"]} {s["modality"]}' + ('' if s['modality'] == 'chat' else str(s['size']))
        total = s['prompt_s'] + s['decode_s']
        share = lambda v: f'{v:>14.0%}' if v is not None else f'{"-":>14}'
        print(f'  {label:<13}{s["encode_s"]:>8.2f}{s["media_prefill_s"]:>10.2f}{s["text_prefill_s"]:>9.2f}'
              f'{s["decode_s"]:>8.2f}{total:>8.2f}{share(s["encode_share_of_prompt"])}{share(s["encode_share_of_total"])}')
    print(f'wrote {folder / "summary.csv"}')


def main():
    parser = argparse.ArgumentParser(description='Stage breakdown per workload: encode, prompt, decode.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='time every stage of every workload on this node')
    p.add_argument('--node', help='node label (default: EC2 instance type); match the Phase 1 profile label')
    p.add_argument('--runs', nargs='+', choices=MEDIA_RUNS, default=MEDIA_RUNS,
                   help='image and audio workloads (default: all 5)')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1, help='unrecorded requests per workload before the repeats')
    p.add_argument('--threads', type=int, default=os.cpu_count(), help='default: all vCPUs, like run.sh')
    p.add_argument('--ctx', type=int, default=8192, help='context; enough for 3,072-token text or 4 images')
    p.add_argument('--bin', default=str(HOME / 'llama.cpp/build/bin/llama-mtmd-cli'))
    p.add_argument('--model', default=str(HOME / 'models/Qwen2.5-Omni-3B-Q8_0.gguf'))
    p.add_argument('--mmproj', default=str(HOME / 'models/mmproj-Qwen2.5-Omni-3B-Q8_0.gguf'))
    p.add_argument('--run-id')
    p.set_defaults(func=run)
    s = sub.add_parser('summarize', help='median stage times and the encoding share per workload')
    s.add_argument('folder', help='results/stages/<node>__<model>')
    s.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
