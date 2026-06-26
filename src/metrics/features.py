"""Feature extraction utilities for OA/KLD evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.tokenizer import PerformanceNote, PerformanceTokenizer, tokens_to_time_axis


NOTE_LENGTH_BEATS = np.array(
    [4.0, 2.0, 1.0, 0.5, 0.25, 3.0, 1.5, 0.75, 0.375, 4.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0],
    dtype=np.float32,
)


def _safe_duration(duration: float) -> float:
    return max(duration, 1e-3)


def _normalize(vec: np.ndarray) -> np.ndarray:
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec.astype(np.float32)


@dataclass
class FeatureSet:
    clip_id: str
    features: Dict[str, np.ndarray]


class FeatureExtractor:
    FEATURE_SPECS = {
        "pitch_count": 1,
        "pitch_range": 1,
        "pitch_class_hist": 12,
        "pitch_class_transition": 144,
        "note_density": 1,
        "note_count": 1,
        "avg_pitch_interval": 1,
        "pitch_interval_hist": 12,
        "avg_ioi": 1,
        "ioi_hist": 12,
        "note_length_hist": 12,
        "note_length_transition": 144,
        "velocity_stats": 2,
    }

    def __init__(self, tokenizer: PerformanceTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or PerformanceTokenizer()

    def from_json(self, path: Path) -> FeatureSet:
        payload = json.loads(path.read_text())
        tokens = payload["tokens"]
        clip_id = str(payload.get("meta", {}).get("id", path.stem))
        notes = self.tokenizer.decode(tokens)
        _, duration = tokens_to_time_axis(tokens, self.tokenizer.resolution)
        features = self._compute_features(notes, duration)
        return FeatureSet(clip_id=clip_id, features=features)

    def _compute_features(self, notes: List[PerformanceNote], duration: float) -> Dict[str, np.ndarray]:
        if not notes:
            return {name: np.zeros(size, dtype=np.float32) for name, size in self.FEATURE_SPECS.items()}

        pitches = np.array([note.pitch for note in notes], dtype=np.float32)
        start_times = np.array([note.start for note in notes], dtype=np.float32)
        sorted_idx = np.argsort(start_times)
        start_times = start_times[sorted_idx]
        pitches_sorted = pitches[sorted_idx]
        durations = np.array([max(note.end - note.start, 1e-3) for note in notes], dtype=np.float32)
        velocities = np.array([note.velocity for note in notes], dtype=np.float32)
        duration_sec = _safe_duration(duration)

        pitch_classes = np.mod(pitches, 12)
        pitch_class_hist = _normalize(np.histogram(pitch_classes, bins=12, range=(0, 12))[0].astype(np.float32))
        pitch_class_transition = self._pitch_class_transition(pitch_classes[np.argsort(start_times)])

        intervals = np.abs(np.diff(pitches_sorted))
        pitch_interval_hist = _normalize(np.histogram(intervals, bins=12, range=(0, 12))[0].astype(np.float32))
        avg_interval = np.array([float(np.mean(intervals)) if intervals.size else 0.0], dtype=np.float32)

        iois = np.diff(start_times)
        ioi_hist = _normalize(np.histogram(iois, bins=12, range=(0.0, 1.2))[0].astype(np.float32))
        avg_ioi = np.array([float(np.mean(iois)) if iois.size else 0.0], dtype=np.float32)

        beat_duration = self._estimate_beat_duration(iois, durations)
        note_length_hist = self._note_length_histogram(durations, beat_duration)
        note_length_transition = self._note_length_transition(durations, beat_duration)

        features: Dict[str, np.ndarray] = {
            "pitch_count": np.array([len(np.unique(pitches))], dtype=np.float32),
            "pitch_range": np.array([pitches.max() - pitches.min()], dtype=np.float32),
            "pitch_class_hist": pitch_class_hist,
            "pitch_class_transition": pitch_class_transition,
            "note_density": np.array([len(notes) / duration_sec], dtype=np.float32),
            "note_count": np.array([len(notes)], dtype=np.float32),
            "avg_pitch_interval": avg_interval,
            "pitch_interval_hist": pitch_interval_hist,
            "avg_ioi": avg_ioi,
            "ioi_hist": ioi_hist,
            "note_length_hist": note_length_hist,
            "note_length_transition": note_length_transition,
            "velocity_stats": np.array([float(np.mean(velocities)), float(np.std(velocities))], dtype=np.float32),
        }
        return features

    def _estimate_beat_duration(self, iois: np.ndarray, durations: np.ndarray) -> float:
        positives = iois[iois > 1e-4]
        if positives.size:
            return float(np.median(positives))
        dur_pos = durations[durations > 1e-4]
        if dur_pos.size:
            return float(np.median(dur_pos))
        return 0.5

    def _pitch_class_transition(self, ordered_pitch_classes: np.ndarray) -> np.ndarray:
        if ordered_pitch_classes.size < 2:
            return np.zeros(144, dtype=np.float32)
        matrix = np.zeros((12, 12), dtype=np.float32)
        ordered = ordered_pitch_classes.astype(int) % 12
        for a, b in zip(ordered[:-1], ordered[1:]):
            matrix[a, b] += 1.0
        return _normalize(matrix.reshape(-1))

    def _note_length_histogram(self, durations: np.ndarray, beat_duration: float) -> np.ndarray:
        if beat_duration <= 1e-6:
            beat_duration = 0.5
        buckets = np.zeros(len(NOTE_LENGTH_BEATS), dtype=np.float32)
        for value in durations:
            beats = value / beat_duration
            idx = int(np.argmin(np.abs(NOTE_LENGTH_BEATS - beats)))
            buckets[idx] += 1.0
        return _normalize(buckets)

    def _note_length_transition(self, durations: np.ndarray, beat_duration: float) -> np.ndarray:
        if len(durations) < 2:
            return np.zeros(144, dtype=np.float32)
        if beat_duration <= 1e-6:
            beat_duration = 0.5
        indices: list[int] = []
        for value in durations:
            beats = value / beat_duration
            idx = int(np.argmin(np.abs(NOTE_LENGTH_BEATS - beats)))
            indices.append(idx)
        matrix = np.zeros((12, 12), dtype=np.float32)
        for a, b in zip(indices[:-1], indices[1:]):
            matrix[a, b] += 1.0
        return _normalize(matrix.reshape(-1))


__all__ = ["FeatureExtractor", "FeatureSet"]
