#!/usr/bin/env python3
"""Answer quality per quantization: the other half of the speed results in benchmark.py.

Quality depends on the weights, not the machine, so this runs once per quantization
(on any one node), not on every instance. Run it on the node, next to llama-server:

  python src/quality.py            # collect answers from the model llama-server is serving
  ./src/run.sh qwen Q4_K_M         # switch model, collect again; repeat for every quantization
  python src/quality.py --score    # compare all collected models

Four tasks, the same inputs for every model, temperature 0:
  chat        Dolly questions (run 00)               ROUGE-L vs the reference model's answer
  summarize   512-token CNN/DailyMail texts (run 01) ROUGE-L vs the reference model's answer
  image       1 COCO photo (run 04)                  ROUGE-L vs the reference model's answer
  transcribe  whole LibriSpeech recordings, 5-25 s   word error rate vs the human transcript
The reference model is the unquantized one (f16) unless --reference says otherwise.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import re
import statistics
import time
from pathlib import Path

import requests

from benchmark import append_rows, payload, server_info
from cnn_dailymail import CNNDailyMailDataset
from coco import COCODataset
from dolly import DollyQADataset
from librispeech import LibriSpeechDataset

INPUTS = 10                  # inputs per task
TASKS = ("chat", "summarize", "image", "transcribe")
TIMEOUT_SECONDS = 900
TRANSCRIBE_TOKENS = 256      # transcripts must not be cut off; other tasks keep benchmark.py's 128
OUT_DIR = Path("results/quality")
ANSWER_FIELDS = ["timestamp_utc", "model", "task", "input", "status", "error",
                 "prompt_tokens", "output_tokens", "e2e_s", "answer", "reference"]
SCORE_FIELDS = ["timestamp_utc", "model", "reference_model", "task", "inputs",
                "rouge_l_vs_reference", "identical_to_reference", "wer_vs_transcript",
                "mean_answer_words"]


def recordings(ds, n):
    """First n whole LibriSpeech recordings of 5-25 s (one 30 s audio window), with transcripts.

    Whole recordings, not the cut clips benchmark.py uses, so the transcript matches the audio.
    """
    import soundfile as sf

    transcripts = {}
    for path in (ds.root / "LibriSpeech").rglob("*.trans.txt"):
        for line in path.read_text().splitlines():
            key, _, text = line.partition(" ")
            transcripts[key] = text
    out = []
    for path in ds.files:
        if path.stem in transcripts and 5 <= sf.info(path).duration <= 25:
            wav = ds.root / "clips" / f"{path.stem}-full.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            if not wav.exists():
                samples, rate = sf.read(path)
                sf.write(wav, samples, rate, subtype="PCM_16")
            out.append((wav, transcripts[path.stem]))
            if len(out) == n:
                break
    return out


def transcribe_payload(wav):
    encoded = base64.b64encode(wav.read_bytes()).decode()
    return {'messages': [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{encoded}"}},
                {"type": "text", "text": "Transcribe this speech exactly. Reply with only the transcript."}]}],
            'max_tokens': TRANSCRIBE_TOKENS, 'temperature': 0, 'cache_prompt': False,
            'chat_template_kwargs': {'enable_thinking': False}}


def jobs(n, tasks):
    """(task, input index, request payload, reference text) for every request."""
    out = []
    for task, modality, ds_class, setting in (("chat", "chat", DollyQADataset, "short"),
                                              ("summarize", "text", CNNDailyMailDataset, 512),
                                              ("image", "image", COCODataset, 1)):
        if task not in tasks:
            continue
        ds = ds_class()
        for i in range(n):
            p = payload(ds, modality, setting, i)
            p['stream'] = False
            p.pop('stream_options')
            out.append((task, i, p, ""))
    if "transcribe" in tasks:
        for i, (wav, text) in enumerate(recordings(LibriSpeechDataset(), n)):
            out.append(("transcribe", i, transcribe_payload(wav), text))
    return out


def collect(url, n, tasks):
    model, _ = server_info(url)
    print(f'collecting answers from {model}', flush=True)
    rows = []
    for task, i, p, reference in jobs(n, tasks):
        start = time.perf_counter()
        try:
            r = requests.post(url.rstrip('/') + '/v1/chat/completions', json=p, timeout=TIMEOUT_SECONDS)
            r.raise_for_status()
            body = r.json()
            row = {"status": "ok", "error": "", "answer": body["choices"][0]["message"]["content"],
                   "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
                   "output_tokens": body.get("usage", {}).get("completion_tokens")}
        except Exception as exc:
            row = {"status": "error", "error": str(exc)[:180], "answer": "",
                   "prompt_tokens": None, "output_tokens": None}
        row.update({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    "model": model, "task": task, "input": i,
                    "e2e_s": time.perf_counter() - start, "reference": reference})
        print(f"  {task} {i}: {row['status']} {row['e2e_s']:.1f}s {row['answer'][:60]!r}", flush=True)
        rows.append(row)
    append_rows(OUT_DIR / "answers.csv", ANSWER_FIELDS, rows)
    print(f'Appended {len(rows)} answers to {OUT_DIR}/answers.csv')


def words(text):
    """Upper-case words without punctuation, as in LibriSpeech transcripts."""
    tokens = (w.strip("'") for w in re.sub(r"[^A-Z0-9' ]+", " ", text.upper()).split())
    return [w for w in tokens if w]


def transcript(answer):
    """The transcript itself, without a lead-in such as "The speech transcribed is:"."""
    lead_in = re.match(r"^[^:\n]{0,60}transcri[^:\n]{0,40}:", answer, re.IGNORECASE)
    return answer[lead_in.end():] if lead_in else answer


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def rouge_l(candidate, reference):
    """ROUGE-L F1: longest common word subsequence, as a share of both answers' length."""
    a, b = words(candidate), words(reference)
    if not a or not b:
        return 0.0
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], prev + 1 if x == y else max(row[j], row[j - 1])
    lcs = row[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def score(reference_model):
    with (OUT_DIR / "answers.csv").open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    latest = {}   # the most recent answer per (model, task, input), if collected twice
    for r in rows:
        latest[(r["model"], r["task"], r["input"])] = r
    models = sorted({m for m, _, _ in latest})
    if reference_model is None:
        reference_model = next((m for m in models if m.lower().endswith("f16")), None)
    if reference_model not in models:
        raise SystemExit(f"reference model {reference_model!r} not collected yet; models: {models}")
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    out = []
    for model in models:
        for task in TASKS:
            pairs = [(r, latest.get((reference_model, task, r["input"])))
                     for (m, t, _), r in latest.items() if m == model and t == task]
            pairs = [(r, ref) for r, ref in pairs if ref is not None]
            if not pairs:
                continue
            mean = lambda xs: statistics.mean(xs) if xs else None
            out.append({
                "timestamp_utc": stamp, "model": model, "reference_model": reference_model,
                "task": task, "inputs": len(pairs),
                "rouge_l_vs_reference": mean([rouge_l(r["answer"], ref["answer"]) for r, ref in pairs]),
                "identical_to_reference": mean([float(r["answer"].strip() == ref["answer"].strip())
                                                for r, ref in pairs]),
                "wer_vs_transcript": mean([edit_distance(words(r["reference"]), words(transcript(r["answer"])))
                                           / len(words(r["reference"])) for r, _ in pairs])
                                     if task == "transcribe" else None,
                "mean_answer_words": mean([len(words(r["answer"])) for r, _ in pairs]),
            })
    append_rows(OUT_DIR / "scores.csv", SCORE_FIELDS, out)
    print(f"reference model: {reference_model}")
    print(f"{'model':<28} {'task':<11} {'n':>3} {'ROUGE-L':>8} {'identical':>9} {'WER':>6} {'words':>6}")
    fmt = lambda v, spec: format(v, spec) if v is not None else "-"
    for s in out:
        print(f"{s['model']:<28} {s['task']:<11} {s['inputs']:>3} {fmt(s['rouge_l_vs_reference'], '8.3f')} "
              f"{fmt(s['identical_to_reference'], '9.0%')} {fmt(s['wer_vs_transcript'], '6.1%')} "
              f"{fmt(s['mean_answer_words'], '6.0f')}")
    print(f'Appended scores to {OUT_DIR}/scores.csv')


def main():
    parser = argparse.ArgumentParser(description='Answer quality per quantization.')
    parser.add_argument('--score', action='store_true', help='score the collected answers instead of collecting')
    parser.add_argument('--reference', help='model to compare against; default: the f16 model')
    parser.add_argument('--inputs', type=int, default=INPUTS, help='inputs per task')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS,
                        help='collect only these tasks (e.g. to redo one); default: all four')
    parser.add_argument('--url', default='http://127.0.0.1:8080', help='llama-server address')
    args = parser.parse_args()
    if args.score:
        score(args.reference)
    else:
        collect(args.url, args.inputs, args.tasks)

if __name__ == '__main__':
    main()
