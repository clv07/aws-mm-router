"""Dolly-15k short chat questions for Qwen2.5-Omni profiling.

Short prompt (~40 tokens), longer answer: the decode-heavy text case, unlike the
long-input CNN/DailyMail summaries. Uses open_qa + general_qa questions that need
no reference text, shuffled with a fixed seed so every node gets the same order.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests


class DollyQADataset:
    SIZES = ("short",)
    SEED = 291

    def __init__(self, root: str | Path = "dataset/dolly"):
        path = Path(root) / "qa.jsonl"
        if not path.exists():
            raise FileNotFoundError("No questions found; run with --download")
        self.questions = [json.loads(line)["question"] for line in path.open(encoding="utf-8")]

    @classmethod
    def download(cls, root: str | Path = "dataset/dolly") -> None:
        from datasets import load_dataset

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        rows = load_dataset("databricks/databricks-dolly-15k", split="train",
                            cache_dir=str(root / ".cache"))
        questions = [r["instruction"].strip() for r in rows
                     if r["category"] in ("open_qa", "general_qa") and not r["context"].strip()
                     and 20 <= len(r["instruction"]) <= 400]
        random.Random(cls.SEED).shuffle(questions)
        with (root / "qa.jsonl").open("w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps({"question": q}) + "\n")

    def messages(self, size: str = "short", start: int = 0) -> list[dict]:
        if size not in self.SIZES:
            raise ValueError(f"size must be one of {self.SIZES}")
        question = self.questions[start % len(self.questions)]
        return [{"role": "user", "content": [{"type": "text", "text": question}]}]

    def send(self, endpoint: str, size: str = "short", start: int = 0, timeout: int = 300):
        response = requests.post(
            endpoint,
            json={"messages": self.messages(size, start), "return_audio": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--root", default="dataset/dolly")
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    if args.download:
        DollyQADataset.download(args.root)
    if args.endpoint:
        print(DollyQADataset(args.root).send(args.endpoint))
