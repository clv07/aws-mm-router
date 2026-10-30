"""Mini LibriSpeech audio inputs for Qwen2.5-Omni profiling."""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

import requests


class LibriSpeechDataset:
    DURATIONS = (5, 10, 20)
    ARCHIVES = {
        "dev-clean-2.tar.gz": "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
        "train-clean-5.tar.gz": "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
    }

    def __init__(self, root: str | Path = "dataset/librispeech"):
        self.root = Path(root)
        self.files = sorted((self.root / "LibriSpeech").rglob("*.flac"))
        if not self.files:
            raise FileNotFoundError(f"No FLAC files under {self.root}; run with --download")

    @classmethod
    def download(cls, root: str | Path = "dataset/librispeech") -> None:
        import tarfile
        import urllib.request

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for name, url in cls.ARCHIVES.items():
            archive = root / name
            if not archive.exists():
                urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter="data")

    def clip(self, duration: int, start: int = 0) -> Path:
        import soundfile as sf

        if duration not in self.DURATIONS:
            raise ValueError(f"duration must be one of {self.DURATIONS}")

        def long_enough(path: Path) -> bool:
            info = sf.info(path)
            return info.frames >= duration * info.samplerate

        source = next(
            (p for p in self.files[start:] + self.files[:start] if long_enough(p)),
            None,
        )
        if source is None:
            raise ValueError(f"No recording is at least {duration} seconds long")
        output = self.root / "clips" / f"{source.stem}-{duration}s.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            info = sf.info(source)
            samples, rate = sf.read(source, stop=duration * info.samplerate)
            sf.write(output, samples, rate, subtype="PCM_16")
        return output

    def messages(self, duration: int, start: int = 0) -> list[dict]:
        path = self.clip(duration, start)
        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        encoded = base64.b64encode(path.read_bytes()).decode()
        content = [
            {"type": "audio", "audio": f"data:{mime};base64,{encoded}"},
            {"type": "text", "text": "Transcribe this speech exactly."},
        ]
        return [{"role": "user", "content": content}]

    def send(self, endpoint: str, duration: int, start: int = 0, timeout: int = 300):
        response = requests.post(
            endpoint,
            json={"messages": self.messages(duration, start), "return_audio": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--root", default="dataset/librispeech")
    parser.add_argument("--duration", type=int, choices=LibriSpeechDataset.DURATIONS, default=5)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    if args.download:
        LibriSpeechDataset.download(args.root)
    if args.endpoint:
        print(LibriSpeechDataset(args.root).send(args.endpoint, args.duration))
