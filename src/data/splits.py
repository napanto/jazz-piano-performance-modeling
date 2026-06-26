"""Album-aware split builder for the PiJAMA dataset.

The blueprint caps usable material to ~10–20 h of training data and 1–2 h for
validation/test while ensuring albums (or artist-album pairs) never straddle
splits. This module ingests the metadata CSV, filters by transcription
agreement, groups tracks by `(artist, album, recording_condition)`, then greedily
fills per-split hour budgets in a deterministic, reproducible order.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

LOGGER = logging.getLogger(__name__)
DEFAULT_MIN_AGREEMENT = 0.90
DEFAULT_BUDGETS = {"train": 16.0, "val": 2.0, "test": 2.0}  # hours
DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass(frozen=True)
class TrackInfo:
    id: str
    artist: str
    album: str
    title: str
    recording_condition: str
    duration_sec: float
    midi_path: str
    agreement: float
    split_hint: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "artist": self.artist,
            "album": self.album,
            "title": self.title,
            "recording_condition": self.recording_condition,
            "duration_sec": self.duration_sec,
            "midi_filepath": self.midi_path,
            "agreement": self.agreement,
            "split_hint": self.split_hint,
        }


@dataclass(frozen=True)
class AlbumGroup:
    key: str
    tracks: list[TrackInfo]

    @property
    def duration_sec(self) -> float:
        return sum(track.duration_sec for track in self.tracks)

    @property
    def split_hint(self) -> str:
        votes: dict[str, int] = {}
        for track in self.tracks:
            votes[track.split_hint] = votes.get(track.split_hint, 0) + 1
        return max(votes, key=votes.get) if votes else "train"


def load_tracks(manifest_path: Path, min_agreement: float) -> list[TrackInfo]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    tracks: list[TrackInfo] = []
    for row in rows:
        try:
            agreement = float(row.get("agreement", 1.0))
        except (TypeError, ValueError):
            agreement = 1.0
        if agreement < min_agreement:
            continue
        duration = float(row.get("duration_sec") or 0.0)
        tracks.append(
            TrackInfo(
                id=str(row.get("id")),
                artist=row.get("artist", ""),
                album=row.get("album", ""),
                title=row.get("title", ""),
                recording_condition=row.get("recording_condition", ""),
                duration_sec=duration,
                midi_path=row.get("midi_filepath", ""),
                agreement=agreement,
                split_hint=row.get("split", "train"),
            )
        )
    return tracks


def group_by_album(tracks: Sequence[TrackInfo]) -> list[AlbumGroup]:
    grouped: dict[str, list[TrackInfo]] = {}
    for track in tracks:
        key = f"{track.artist}::{track.album}::{track.recording_condition}"
        grouped.setdefault(key, []).append(track)
    return [AlbumGroup(key=key, tracks=sorted(group, key=lambda t: t.id)) for key, group in sorted(grouped.items())]


def allocate_groups(
    groups: Sequence[AlbumGroup],
    budgets_hours: Dict[str, float],
    seed: int = 7,
    split_order: Sequence[str] | None = None,
    overflow_tolerance_hours: float = 0.25,
) -> tuple[dict[str, list[AlbumGroup]], list[AlbumGroup]]:
    """Assigns album groups to splits while respecting hour budgets."""
    order = list(split_order) if split_order else list(budgets_hours.keys())
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    budgets_sec = {split: max(0.0, hours * 3600.0) for split, hours in budgets_hours.items()}
    tolerance_sec = max(0.0, overflow_tolerance_hours * 3600.0)
    assignments: dict[str, list[AlbumGroup]] = {split: [] for split in budgets_sec}
    remaining_groups = []
    for group in shuffled:
        preferred = group.split_hint if group.split_hint in budgets_sec else order[0]
        candidate_splits = [preferred] + [split for split in order if split != preferred]
        assigned = False
        for split in candidate_splits:
            if budgets_sec[split] <= 0:
                continue
            prospective = budgets_sec[split]
            if group.duration_sec > prospective + tolerance_sec:
                continue
            assignments[split].append(group)
            budgets_sec[split] = max(0.0, budgets_sec[split] - group.duration_sec)
            assigned = True
            break
        if not assigned:
            remaining_groups.append(group)
        if all(value <= 0 for value in budgets_sec.values()):
            remaining_groups.extend(shuffled[shuffled.index(group) + 1 :])
            break
    return assignments, remaining_groups


def build_manifest(assignments: dict[str, list[AlbumGroup]], config: dict, dropped: list[AlbumGroup]) -> dict:
    summary = {}
    splits_payload: dict[str, list[dict]] = {}
    for split, groups in assignments.items():
        track_entries: list[dict] = []
        total_duration = 0.0
        for group in groups:
            total_duration += group.duration_sec
            for track in group.tracks:
                payload = track.as_dict() | {"album_group": group.key, "split": split}
                track_entries.append(payload)
        splits_payload[split] = track_entries
        summary[split] = {
            "tracks": len(track_entries),
            "hours": total_duration / 3600.0,
            "albums": len(groups),
        }
    manifest = {
        "config": config,
        "summary": summary,
        "splits": splits_payload,
        "dropped_album_groups": [group.key for group in dropped],
    }
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PiJAMA album-aware splits.")
    parser.add_argument("--manifest", required=True, help="Path to the PiJAMA metadata CSV.")
    parser.add_argument(
        "--output",
        default="data/splits/pijama_album_splits.json",
        help="Destination JSON containing the chosen subset.",
    )
    parser.add_argument("--min-agreement", type=float, default=DEFAULT_MIN_AGREEMENT)
    parser.add_argument("--train-hours", type=float, default=DEFAULT_BUDGETS["train"])
    parser.add_argument("--val-hours", type=float, default=DEFAULT_BUDGETS["val"])
    parser.add_argument("--test-hours", type=float, default=DEFAULT_BUDGETS["test"])
    parser.add_argument("--use-ratios", action="store_true", help="Scale budgets to cover the full dataset using ratios instead of fixed hours.")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS["test"])
    parser.add_argument("--seed", type=int, default=7, help="Seed controlling album order shuffling.")
    parser.add_argument(
        "--overflow-tolerance-hours",
        type=float,
        default=0.25,
        help="Allow an album to exceed a split budget by up to this many hours so whole albums stay intact.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    tracks = load_tracks(Path(args.manifest), args.min_agreement)
    budgets = {"train": args.train_hours, "val": args.val_hours, "test": args.test_hours}
    if args.use_ratios:
        total_hours = sum(track.duration_sec for track in tracks) / 3600.0
        ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
        if ratio_sum <= 0:
            raise ValueError("Split ratios must sum to a positive value")
        budgets = {
            "train": total_hours * (args.train_ratio / ratio_sum),
            "val": total_hours * (args.val_ratio / ratio_sum),
            "test": total_hours * (args.test_ratio / ratio_sum),
        }
    groups = group_by_album(tracks)
    assignments, dropped = allocate_groups(
        groups,
        budgets,
        seed=args.seed,
        overflow_tolerance_hours=args.overflow_tolerance_hours,
    )
    manifest = build_manifest(
        assignments,
        config={
            "manifest": str(Path(args.manifest)),
            "min_agreement": args.min_agreement,
            "budgets_hours": budgets,
            "seed": args.seed,
            "overflow_tolerance_hours": args.overflow_tolerance_hours,
            "use_ratios": args.use_ratios,
            "ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
        },
        dropped=dropped,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Album splits saved to %s (train %.2fh, val %.2fh, test %.2fh).",
        output_path,
        manifest["summary"].get("train", {}).get("hours", 0.0),
        manifest["summary"].get("val", {}).get("hours", 0.0),
        manifest["summary"].get("test", {}).get("hours", 0.0),
    )
    if dropped:
        LOGGER.info("Dropped %d album groups once budgets were filled.", len(dropped))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
