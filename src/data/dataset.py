"""PyTorch dataset utilities for PiJAMA token caches.

Combines the preprocessed token JSON files with the split manifest to provide
random 15-second PerformanceRNN training clips with optional augmentation.
"""
from __future__ import annotations

import json
import random
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.data.tokenizer import PerformanceTokenizer, tokens_to_time_axis, transpose_tokens
from src.data.utils import list_token_paths, load_split_ids


@dataclass
class TokenSequence:
    tokens: list[int]
    times: list[float]
    duration: float
    path: Path
    metadata: dict

    @classmethod
    def from_json(cls, path: Path, tokenizer: PerformanceTokenizer) -> "TokenSequence":
        payload = json.loads(path.read_text())
        tokens = payload["tokens"]
        times, duration = tokens_to_time_axis(tokens, tokenizer.resolution)
        return cls(tokens=list(tokens), times=times, duration=duration, path=path, metadata=payload.get("meta", {}))

    def clip_by_time(
        self,
        start_time: float,
        clip_seconds: float,
        sequence_length: int,
    ) -> list[int]:
        if not self.tokens:
            raise ValueError(f"Token cache {self.path} is empty")
        start_idx = bisect_left(self.times, max(0.0, start_time))
        end_time = min(self.duration, start_time + clip_seconds)
        end_idx = bisect_left(self.times, end_time) + 1
        if end_idx - start_idx < sequence_length + 1:
            # fallback: grab the last sequence_length+1 tokens
            start_idx = max(0, len(self.tokens) - sequence_length - 1)
            end_idx = len(self.tokens)
        return self.tokens[start_idx:end_idx]


class PerformanceDataset(Dataset):
    """Random window dataset for PerformanceRNN training."""

    def __init__(
        self,
        processed_root: Path,
        split: str,
        sequence_length: int,
        clip_seconds: float = 15.0,
        clips_per_track: int = 32,
        seed: int = 7,
        splits_manifest: Path | None = None,
        transpose_min: int = -4,
        transpose_max: int = 4,
        time_stretch_options: Sequence[float] | None = (0.95, 0.975, 1.0, 1.025, 1.05),
        tokenizer: PerformanceTokenizer | None = None,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.split = split
        self.sequence_length = sequence_length
        self.clip_seconds = clip_seconds
        self.clips_per_track = max(1, clips_per_track)
        self.seed = seed
        self.transpose_min = transpose_min
        self.transpose_max = transpose_max
        self.time_stretch_options: Tuple[float, ...] = tuple(time_stretch_options or ())
        self.tokenizer = tokenizer or PerformanceTokenizer()
        allowed_ids = load_split_ids(Path(splits_manifest), split) if splits_manifest else None
        self.track_files = list_token_paths(self.processed_root, split, allowed_ids)
        if not self.track_files:
            raise ValueError(f"No token files found for split {split}")
        self._cache: dict[int, TokenSequence] = {}
        self.total_clips = len(self.track_files) * self.clips_per_track

    def __len__(self) -> int:  # type: ignore[override]
        return self.total_clips

    def __getitem__(self, index: int):  # type: ignore[override]
        track_idx = index % len(self.track_files)
        rng = random.Random(self.seed + index)
        cache = self._get_cache(track_idx)
        max_start = max(0.0, cache.duration - self.clip_seconds)
        start_time = rng.uniform(0.0, max_start) if cache.duration > 0 else 0.0
        clip_tokens = cache.clip_by_time(start_time, self.clip_seconds, self.sequence_length + 1)
        transpose = 0
        if self.transpose_min is not None and self.transpose_max is not None:
            transpose = rng.randint(self.transpose_min, self.transpose_max)
            clip_tokens = transpose_tokens(clip_tokens, transpose)
        stretch = 1.0
        if self.time_stretch_options:
            stretch = rng.choice(self.time_stretch_options)
            clip_tokens = self.tokenizer.stretch_time(clip_tokens, stretch)
        clip_tokens = self._ensure_length(clip_tokens)
        inputs = torch.tensor(clip_tokens[:-1], dtype=torch.long)
        targets = torch.tensor(clip_tokens[1:], dtype=torch.long)
        return {
            "input_ids": inputs,
            "labels": targets,
            "source_path": str(cache.path),
            "start_time": start_time,
            "transpose": transpose,
            "time_stretch": stretch,
        }

    def _get_cache(self, track_idx: int) -> TokenSequence:
        cache = self._cache.get(track_idx)
        if cache is None:
            cache = TokenSequence.from_json(self.track_files[track_idx], self.tokenizer)
            self._cache[track_idx] = cache
        return cache

    def _ensure_length(self, tokens: Sequence[int]) -> List[int]:
        if len(tokens) < self.sequence_length + 1:
            pad_value = tokens[-1]
            pad = [pad_value] * (self.sequence_length + 1 - len(tokens))
            tokens = list(tokens) + pad
        return list(tokens[: self.sequence_length + 1])


__all__ = ["PerformanceDataset", "TokenSequence"]
