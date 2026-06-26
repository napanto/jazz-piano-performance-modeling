"""Build a FMD reference pool from PiJAMA's raw MIDI files.

Goes through ``data/splits/pijama_album_splits.json``, picks all entries
with split == ``test`` (and optionally ``val``), copies the corresponding
raw MIDI files into ``data/fmd_reference/<split>/`` for downstream use by
``src/metrics/fmd.py``.

We use RAW MIDIs (not token round-trip) because FMD's CLaMP-2 backbone
expects the natural musical signal — pedal CC, micro-timing, original
velocities, etc.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT3 = os.environ.get("PROJECT3_ROOT", str(_REPO_ROOT.parent / "project3"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-manifest", default=f"{_PROJECT3}/data/splits/pijama_album_splits.json")
    p.add_argument("--raw-root", default=f"{_PROJECT3}/data/raw/pijama/midi_hawthorne")
    p.add_argument("--out-dir", default=str(_REPO_ROOT / "data" / "fmd_reference"))
    p.add_argument("--splits", nargs="+", default=["test", "val"])
    p.add_argument("--max-per-split", type=int, default=200)
    p.add_argument("--symlink", action="store_true", default=True,
                   help="Symlink instead of copying to save disk")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(Path(args.splits_manifest).read_text())
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    splits = manifest.get("splits", manifest)
    total = 0
    for split in args.splits:
        entries = splits.get(split, [])
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for entry in entries:
            if count >= args.max_per_split:
                break
            rel = entry.get("midi_filepath")
            if not rel:
                continue
            # Strip leading "data/" if present (the manifest uses paths relative
            # to a project3 root).
            for prefix in ("data/", "./", "data/midi/"):
                if rel.startswith(prefix):
                    rel = rel[len(prefix):]
                    break
            src = raw_root / rel
            if not src.exists():
                # Try alternative path layouts.
                alt = raw_root / "midi" / rel.removeprefix("midi/")
                if alt.exists():
                    src = alt
                else:
                    continue
            stem = Path(rel).stem.replace("/", "_")
            dst = out_dir / f"{count:03d}_{stem}.midi"
            if dst.exists():
                count += 1; continue
            if args.symlink:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            count += 1
        print(f"  [{split}] collected {count} reference MIDIs -> {out_dir}")
        total += count
    print(f"\nTotal: {total} reference MIDIs under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
