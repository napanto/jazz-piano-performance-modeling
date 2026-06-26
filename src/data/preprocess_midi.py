"""PiJAMA preprocessing utilities (filtering + PerformanceRNN tokenization).

The Hawthorne transcription variant we use already bakes in pedal behavior, so
this module simply loads the cleaned notes and converts them into the 413-event
PerformanceRNN vocabulary for caching.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pretty_midi
from tqdm import tqdm

from src.data.tokenizer import PerformanceNote, PerformanceTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessConfig:
    min_agreement: float = 0.90
    splits: tuple[str, ...] = ("train", "val", "test")
    time_shift_resolution: float = 0.008  # 8 ms
    time_shift_bins: int = 125
    velocity_bins: int = 32
    split_overrides: dict[str, str] | None = None


def load_manifest(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def load_split_overrides(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    overrides: dict[str, str] = {}
    for split_name, entries in data.get("splits", {}).items():
        for entry in entries:
            track_id = str(entry.get("id"))
            if track_id:
                overrides[track_id] = split_name
    return overrides


def slugify(value: str | None) -> str:
    value = value or "unknown"
    value = value.lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-") or "unknown"


def resolve_midi_path(raw_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    candidates = [raw_root / relative]
    if relative.parts and relative.parts[0] == "data":
        stripped = Path(*relative.parts[1:])
        candidates.append(raw_root / stripped)
    else:
        stripped = relative
    candidates.append(raw_root / "midi" / stripped)
    for variant_dir in raw_root.glob("midi_*"):
        candidates.append(variant_dir / stripped)
        candidates.append(variant_dir / "midi" / stripped)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve MIDI path for {relative_path} inside {raw_root}")


def extract_notes(pm: pretty_midi.PrettyMIDI) -> list[PerformanceNote]:
    notes: list[PerformanceNote] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append(
                PerformanceNote(
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                    velocity=note.velocity,
                )
            )
    notes.sort(key=lambda note: (note.start, note.pitch))
    return notes


def preprocess_row(
    row: dict[str, str],
    raw_root: Path,
    output_root: Path,
    tokenizer: PerformanceTokenizer,
    config: PreprocessConfig,
) -> Path:
    midi_path = resolve_midi_path(raw_root, row["midi_filepath"])
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = extract_notes(pm)
    tokens = tokenizer.encode(notes)
    artist_slug = slugify(row.get("artist"))
    title_slug = slugify(row.get("title"))
    split_name = row.get("split", "unsplit")
    output_dir = output_root / split_name / artist_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    track_id = row.get("id", "unknown")
    filename = f"{track_id.zfill(5) if track_id.isdigit() else track_id}_{title_slug}.json"
    payload = {
        "meta": {
            "id": row.get("id"),
            "artist": row.get("artist"),
            "album": row.get("album"),
            "title": row.get("title"),
            "split": split_name,
            "agreement": row.get("agreement"),
            "duration_sec": row.get("duration_sec"),
            "tokenizer": tokenizer.spec(),
        },
        "paths": {"source_midi": str(midi_path)},
        "tokens": tokens,
    }
    output_path = output_dir / filename
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path


def should_include(row: dict[str, str], config: PreprocessConfig) -> bool:
    try:
        agreement = float(row.get("agreement", 1.0))
    except (TypeError, ValueError):
        agreement = 1.0
    if agreement < config.min_agreement:
        return False
    if config.split_overrides is not None:
        track_id = str(row.get("id"))
        split_override = config.split_overrides.get(track_id)
        if split_override is None:
            return False
        row["split"] = split_override
    split = row.get("split")
    if config.splits and split not in config.splits:
        return False
    return True


def preprocess_manifest(
    manifest_path: Path,
    raw_root: Path,
    output_root: Path,
    config: PreprocessConfig,
    limit: int | None = None,
) -> dict[str, int]:
    rows = load_manifest(manifest_path)
    tokenizer = PerformanceTokenizer(
        resolution=config.time_shift_resolution,
        time_shift_bins=config.time_shift_bins,
        velocity_bins=config.velocity_bins,
    )
    stats = {"processed": 0, "skipped": 0}
    for row in tqdm(rows, desc="tokenizing", unit="track"):
        if not should_include(row, config):
            stats["skipped"] += 1
            continue
        try:
            preprocess_row(row, raw_root, output_root, tokenizer, config)
        except FileNotFoundError as err:
            LOGGER.warning("Skipping %s (%s)", row.get("title"), err)
            stats["skipped"] += 1
            continue
        stats["processed"] += 1
        if limit and stats["processed"] >= limit:
            break
    LOGGER.info(
        "Preprocessing done: %d processed, %d skipped.", stats["processed"], stats["skipped"]
    )
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pedal-aware PiJAMA preprocessing.")
    parser.add_argument("--manifest", required=True, help="Path to the PiJAMA metadata CSV.")
    parser.add_argument(
        "--raw-root",
        default="data/raw/pijama",
        help="Directory containing the extracted PiJAMA MIDI folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/pijama",
        help="Destination directory where token files are written.",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=PreprocessConfig.min_agreement,
        help="Minimum transcription-agreement score to keep a track.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(PreprocessConfig.splits),
        help="List of split names to export (train/val/test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N matching tracks (useful for smoke tests).",
    )
    parser.add_argument(
        "--splits-manifest",
        default=None,
        help="Optional JSON manifest produced by src.data.splits to restrict track IDs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    split_overrides = None
    if args.splits_manifest:
        split_overrides = load_split_overrides(Path(args.splits_manifest))
    config = PreprocessConfig(
        min_agreement=args.min_agreement,
        splits=tuple(args.splits),
        split_overrides=split_overrides,
    )
    preprocess_manifest(
        manifest_path=Path(args.manifest),
        raw_root=Path(args.raw_root),
        output_root=Path(args.output_dir),
        config=config,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
