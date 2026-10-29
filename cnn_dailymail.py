"""CNN/DailyMail text inputs for Qwen2.5-Omni profiling."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


class CNNDailyMailDataset:
    SIZES = (512, 1024, 3072)

    def __init__(self, root: str | Path = "dataset/cnn_dailymail", split: str = "train"):
        from datasets import load_from_disk
        from transformers import AutoTokenizer

        self.root = Path(root)
        self.data = load_from_disk(str(self.root / "dataset"))[split]
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-3B")

    @classmethod
    def download(cls, root: str | Path = "dataset/cnn_dailymail") -> None:
        from datasets import load_dataset

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(
            "abisee/cnn_dailymail", "3.0.0", cache_dir=str(root / ".cache")
        )
        dataset.save_to_disk(str(root / "dataset"))

    def messages(self, size: int, start: int = 0) -> list[dict]:
        """Return a text-only conversation whose user content is exactly `size` tokens."""
        if size not in self.SIZES:
            raise ValueError(f"size must be one of {self.SIZES}")

        prefix = "Summarize the following news article:\n\n"
        text = prefix
        row = start
        while len(self.tokenizer.encode(text, add_special_tokens=False)) < size:
            text += self.data[row % len(self.data)]["article"] + "\n"
            row += 1
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:size]
        content = self.tokenizer.decode(ids, skip_special_tokens=True)
        actual = len(self.tokenizer.encode(content, add_special_tokens=False))
        if actual != size:
            raise RuntimeError(f"Expected {size} tokens, got {actual}")
        return [{"role": "user", "content": [{"type": "text", "text": content}]}]

    def send(self, endpoint: str, size: int, start: int = 0, timeout: int = 300):
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
    parser.add_argument("--root", default="data/cnn_dailymail")
    parser.add_argument("--size", type=int, choices=CNNDailyMailDataset.SIZES, default=512)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    if args.download:
        CNNDailyMailDataset.download(args.root)
    if args.endpoint:
        print(CNNDailyMailDataset(args.root).send(args.endpoint, args.size))
