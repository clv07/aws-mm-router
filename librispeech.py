"""Mini LibriSpeech audio inputs for Qwen2.5-Omni profiling."""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

import requests


class LibriSpeechDataset:
    # llama.cpp pads Qwen audio to whole 30 s windows (clip + 30 s of silence, cut into
    # 30 s windows, partial window dropped), so 5/10/20 s all cost one window and 40 s
    # costs two. benchmark.py uses 10 and 40; 5 and 20 stay for roundrobin.py.
    DURATIONS = (5, 10, 20, 40)
    ARCHIVES = {
        "dev-clean-2.tar.gz": "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
        "train-clean-5.tar.gz": "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
    }

    def __init__(self, root: str | Path = "dataset/librispeech"):
        self.root = Path(root)
        self.files = sorted((self.root / "LibriSpeech").rglob("*.flac"))
        self._long = {}  # duration -> recordings at least that long
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

        # The start-th recording that is long enough, so each start gets a different clip
        # (taking the first long-enough one at or after `start` repeats the same clip).
        if duration not in self._long:
            self._long[duration] = [p for p in self.files if long_enough(p)]
        long = self._long[duration]
        if not long:
            # Few recordings are over ~35 s: join consecutive ones and cut to length.
            return self._joined_clip(duration, self.files[start:] + self.files[:start])
        source = long[start % len(long)]
        output = self.root / "clips" / f"{source.stem}-{duration}s.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            info = sf.info(source)
            samples, rate = sf.read(source, stop=duration * info.samplerate)
            sf.write(output, samples, rate, subtype="PCM_16")
        return output

    def _joined_clip(self, duration: int, order: list[Path]) -> Path:
        import numpy as np
        import soundfile as sf

        output = self.root / "clips" / f"{order[0].stem}-joined-{duration}s.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            parts, total, rate = [], 0, None
            for path in order:
                samples, sr = sf.read(path)
                if rate is None:
                    rate = sr
                if sr != rate:
                    continue
                parts.append(samples)
                total += len(samples)
                if total >= duration * rate:
                    break
            if total < duration * rate:
                raise ValueError(f"Not enough audio for a {duration} second clip")
            sf.write(output, np.concatenate(parts)[:duration * rate], rate, subtype="PCM_16")
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
