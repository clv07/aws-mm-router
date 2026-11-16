"""COCO image inputs for Qwen2.5-Omni profiling."""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

import requests


class COCODataset:
    COUNTS = (1, 2, 4)
    ARCHIVES = {
        "validation": ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip"),
        "test": ("test2017.zip", "http://images.cocodataset.org/zips/test2017.zip"),
    }

    def __init__(self, root: str | Path = "dataset/coco", split: str = "validation"):
        if split not in self.ARCHIVES:
            raise ValueError(f"split must be one of {tuple(self.ARCHIVES)}")
        folder = "val2017" if split == "validation" else "test2017"
        self.files = sorted((Path(root) / folder).glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError("No images found; run with --download")

    @classmethod
    def download(cls, root: str | Path = "dataset/coco", split: str = "validation") -> None:
        import urllib.request
        import zipfile

        if split not in cls.ARCHIVES:
            raise ValueError(f"split must be one of {tuple(cls.ARCHIVES)}")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        name, url = cls.ARCHIVES[split]
        archive = root / name
        if not archive.exists():
            partial = archive.with_suffix(".zip.part")
            urllib.request.urlretrieve(url, partial)
            partial.replace(archive)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(root)

    @staticmethod
    def _data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime};base64,{encoded}"

    def messages(self, count: int, start: int = 0) -> list[dict]:
        if count not in self.COUNTS:
            raise ValueError(f"count must be one of {self.COUNTS}")
        content = [
            {
                "type": "image",
                "image": self._data_url(self.files[(start + i) % len(self.files)]),
            }
            for i in range(count)
        ]
        content.append({"type": "text", "text": "Describe the image or images concisely."})
        return [{"role": "user", "content": content}]

    def send(self, endpoint: str, count: int, start: int = 0, timeout: int = 300):
        response = requests.post(
            endpoint,
            json={"messages": self.messages(count, start), "return_audio": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--root", default="dataset/coco")
    parser.add_argument("--split", choices=tuple(COCODataset.ARCHIVES), default="validation")
    parser.add_argument("--count", type=int, choices=COCODataset.COUNTS, default=1)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    if args.download:
        COCODataset.download(args.root, args.split)
    if args.endpoint:
        print(COCODataset(args.root, args.split).send(args.endpoint, args.count))
