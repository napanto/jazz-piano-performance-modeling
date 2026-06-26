"""PerformanceRNN tokenizer utilities.

This module centralizes the 413-event vocabulary (note-on/off, time-shift, and
velocity bins) described in the PerformanceRNN paper so every stage—dataset
caching, training, decoding—shares the same implementation. The blueprint
expects deterministic MIDI→token and token→MIDI conversions plus unit tests for
round-trips and filtering logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


NOTE_COUNT = 128
TIME_SHIFT_TOKENS = 125
VELOCITY_BINS = 32
VOCAB_SIZE = NOTE_COUNT * 2 + TIME_SHIFT_TOKENS + VELOCITY_BINS  # 413


@dataclass(frozen=True)
class PerformanceNote:
    """Simple representation of a note event used during tokenization."""

    pitch: int
    start: float
    end: float
    velocity: int


class PerformanceTokenizer:
    """Converts expressive notes into PerformanceRNN tokens and back."""

    note_on_base = 0
    note_off_base = NOTE_COUNT
    time_shift_base = NOTE_COUNT * 2
    velocity_base = NOTE_COUNT * 2 + TIME_SHIFT_TOKENS

    def __init__(
        self,
        resolution: float = 0.008,
        time_shift_bins: int = TIME_SHIFT_TOKENS,
        velocity_bins: int = VELOCITY_BINS,
    ) -> None:
        if time_shift_bins != TIME_SHIFT_TOKENS:
            raise ValueError("Time-shift bin count must match the 125-token vocabulary")
        if velocity_bins != VELOCITY_BINS:
            raise ValueError("Velocity bins must match the 32-token vocabulary")
        self.resolution = resolution
        self.time_shift_bins = time_shift_bins
        self.velocity_bins = velocity_bins
        self._bin_size = math.ceil(128 / velocity_bins)

    # ---------------------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------------------
    def encode(self, notes: Sequence[PerformanceNote]) -> list[int]:
        """Encodes sorted notes into PerformanceRNN tokens."""
        if not notes:
            return []
        events = self._notes_to_events(notes)
        tokens: list[int] = []
        current_step = 0
        last_velocity_bin: int | None = None
        for event_time, kind, pitch, velocity in events:
            target_step = max(0, self._seconds_to_steps(event_time))
            delta_steps = target_step - current_step
            if delta_steps < 0:
                delta_steps = 0
            tokens.extend(self._encode_time_shift(delta_steps))
            current_step = target_step
            if kind == "off":
                tokens.append(self.note_off_base + pitch)
            else:
                vel_bin = self._velocity_bin(velocity)
                if vel_bin != last_velocity_bin:
                    tokens.append(self.velocity_base + vel_bin)
                    last_velocity_bin = vel_bin
                tokens.append(self.note_on_base + pitch)
        return tokens

    # ---------------------------------------------------------------------
    # Decoding
    # ---------------------------------------------------------------------
    def decode(self, tokens: Sequence[int]) -> list[PerformanceNote]:
        """Decodes tokens back into approximate notes (for round-trip tests)."""
        notes: list[PerformanceNote] = []
        current_steps = 0
        current_velocity = 64
        active: dict[int, tuple[int, int]] = {}
        for token in tokens:
            if self.note_on_base <= token < self.note_off_base:
                pitch = token - self.note_on_base
                active[pitch] = (current_steps, current_velocity)
            elif self.note_off_base <= token < self.time_shift_base:
                pitch = token - self.note_off_base
                if pitch not in active:
                    continue
                start_steps, velocity = active.pop(pitch)
                start = start_steps * self.resolution
                end = max(start, current_steps * self.resolution)
                notes.append(PerformanceNote(pitch=pitch, start=start, end=end, velocity=velocity))
            elif self.time_shift_base <= token < self.velocity_base:
                delta = (token - self.time_shift_base) + 1
                current_steps += delta
            elif self.velocity_base <= token < self.velocity_base + self.velocity_bins:
                bin_index = token - self.velocity_base
                current_velocity = self._velocity_from_bin(bin_index)
            else:
                raise ValueError(f"Token {token} is outside the PerformanceRNN vocabulary")
        # Flush any lingering notes by closing them at the final step
        for pitch, (start_steps, velocity) in active.items():
            start = start_steps * self.resolution
            end = max(start, current_steps * self.resolution)
            notes.append(PerformanceNote(pitch=pitch, start=start, end=end, velocity=velocity))
        notes.sort(key=lambda note: (note.start, note.pitch))
        return notes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def spec(self) -> dict[str, float | int]:
        return {
            "time_shift_resolution": self.resolution,
            "time_shift_bins": self.time_shift_bins,
            "velocity_bins": self.velocity_bins,
            "vocab_size": VOCAB_SIZE,
        }

    def _notes_to_events(self, notes: Sequence[PerformanceNote]) -> list[tuple[float, str, int, int]]:
        timeline: list[tuple[float, str, int, int]] = []
        for note in notes:
            timeline.append((note.start, "on", note.pitch, note.velocity))
            timeline.append((note.end, "off", note.pitch, note.velocity))
        timeline.sort(key=lambda item: (item[0], 0 if item[1] == "off" else 1, item[2]))
        return timeline

    def _seconds_to_steps(self, seconds: float) -> int:
        return int(round(seconds / self.resolution))

    def _encode_time_shift(self, steps: int) -> list[int]:
        encoded: list[int] = []
        remaining = max(0, steps)
        while remaining > 0:
            chunk = min(remaining, self.time_shift_bins)
            encoded.append(self.time_shift_base + (chunk - 1))
            remaining -= chunk
        return encoded

    def encode_time_steps(self, steps: int) -> list[int]:
        """Expose time-shift chunking for data augmentations."""
        return self._encode_time_shift(steps)

    def stretch_time(self, tokens: Sequence[int], scale: float) -> list[int]:
        """Scale TIME_SHIFT tokens by `scale`, re-quantizing to the 8 ms grid."""
        if math.isclose(scale, 1.0, rel_tol=1e-3):
            return list(tokens)
        stretched: list[int] = []
        for token in tokens:
            if self.time_shift_base <= token < self.velocity_base:
                delta = (token - self.time_shift_base) + 1
                scaled_steps = max(1, int(round(delta * scale)))
                stretched.extend(self._encode_time_shift(scaled_steps))
            else:
                stretched.append(token)
        return stretched

    def _velocity_bin(self, velocity: int) -> int:
        clipped = min(127, max(1, velocity))
        bin_index = min(self.velocity_bins - 1, (clipped - 1) // self._bin_size)
        return bin_index

    def _velocity_from_bin(self, bin_index: int) -> int:
        bin_index = max(0, min(self.velocity_bins - 1, bin_index))
        low = bin_index * self._bin_size + 1
        high = min(127, low + self._bin_size - 1)
        return (low + high) // 2


def tokenize_midi_notes(notes: Iterable[PerformanceNote], tokenizer: PerformanceTokenizer) -> List[int]:
    """Convenience wrapper mirroring the blueprint's dataset plan."""
    return tokenizer.encode(list(notes))


def tokens_to_time_axis(
    tokens: Sequence[int],
    resolution: float = 0.008,
) -> Tuple[List[float], float]:
    """Return the cumulative time (seconds) for every token and total duration."""
    times: list[float] = []
    current_time = 0.0
    for token in tokens:
        times.append(current_time)
        if PerformanceTokenizer.time_shift_base <= token < PerformanceTokenizer.velocity_base:
            delta = (token - PerformanceTokenizer.time_shift_base + 1) * resolution
            current_time += delta
    return times, current_time


def transpose_tokens(tokens: Sequence[int], semitones: int) -> List[int]:
    """Transpose NOTE_ON/OFF events when the shift keeps pitches in range."""
    if semitones == 0:
        return list(tokens)
    shifted: list[int] = []
    for token in tokens:
        if PerformanceTokenizer.note_on_base <= token < PerformanceTokenizer.note_off_base:
            pitch = token - PerformanceTokenizer.note_on_base
            new_pitch = pitch + semitones
            if not 0 <= new_pitch < NOTE_COUNT:
                return list(tokens)
            shifted.append(PerformanceTokenizer.note_on_base + new_pitch)
        elif PerformanceTokenizer.note_off_base <= token < PerformanceTokenizer.time_shift_base:
            pitch = token - PerformanceTokenizer.note_off_base
            new_pitch = pitch + semitones
            if not 0 <= new_pitch < NOTE_COUNT:
                return list(tokens)
            shifted.append(PerformanceTokenizer.note_off_base + new_pitch)
        else:
            shifted.append(token)
    return shifted
