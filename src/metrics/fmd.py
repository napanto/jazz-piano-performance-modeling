"""Fréchet Music Distance — embedding-space FID-style metric for MIDI.

Uses the official ``frechet-music-distance`` library (CLaMP-2 backbone
by default) to compute FMD between a directory of generated MIDIs and a
PiJAMA reference pool.

Designed to slot alongside ``eval_unified.py``: the same model-checkpoint
JSON gets a parallel ``eval_test_fmd.json`` entry with the score.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute FMD between generated and reference MIDI dirs.")
    p.add_argument("--generated-dir", required=True,
                   help="Directory of generated .mid (or .midi) files.")
    p.add_argument("--reference-dir", default=None,
                   help="Directory of reference MIDIs. If omitted, uses the default "
                        "PiJAMA val+test pool baked into the project.")
    p.add_argument("--output", default=None,
                   help="Where to write the JSON result. Defaults to "
                        "<generated_dir>/../eval_test_fmd.json")
    p.add_argument("--backbone", default="clamp2",
                   choices=["clamp2", "clamp"],
                   help="Embedding backbone to use (FMD library default is clamp2).")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda. cpu is the safe default when the GPU is "
                        "occupied by training; CLaMP-2 is ~300M params and CPU "
                        "inference is feasible.")
    p.add_argument("--max-files-gen", type=int, default=200)
    p.add_argument("--max-files-ref", type=int, default=200)
    return p.parse_args()


def _gather_midis(directory: Path, limit: int) -> list[str]:
    paths = sorted(directory.glob("*.mid")) + sorted(directory.glob("*.midi"))
    return [str(p) for p in paths[:limit]]


def main() -> int:
    args = parse_args()
    gen_paths = _gather_midis(Path(args.generated_dir), args.max_files_gen)
    if not gen_paths:
        raise SystemExit(f"No MIDI files found in {args.generated_dir}")
    # Default reference pool resolution order:
    #   1. --reference-dir (CLI)
    #   2. $FMD_REFERENCE_DIR
    #   3. <repo>/data/fmd_reference/test (created by build_pijama_refpool.py)
    if args.reference_dir is not None:
        ref_root = Path(args.reference_dir)
    elif os.environ.get("FMD_REFERENCE_DIR"):
        ref_root = Path(os.environ["FMD_REFERENCE_DIR"])
    else:
        ref_root = _REPO_ROOT / "data" / "fmd_reference" / "test"
    if not ref_root.exists():
        raise SystemExit(
            f"reference dir {ref_root} does not exist; pass --reference-dir or "
            "set $FMD_REFERENCE_DIR, or run "
            "`python -m src.metrics.build_pijama_refpool` first."
        )
    ref_paths = _gather_midis(ref_root, args.max_files_ref)
    if not ref_paths:
        raise SystemExit(f"No reference MIDIs found at {ref_root}")

    # Build the device string FMD expects.
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Lazy-import so this script works on machines without CLaMP available.
    from frechet_music_distance import FrechetMusicDistance
    fmd = FrechetMusicDistance(feature_extractor=args.backbone, verbose=True)
    print(f"Computing FMD with backbone={args.backbone} device={device}")
    print(f"  generated: {len(gen_paths)} files from {args.generated_dir}")
    print(f"  reference: {len(ref_paths)} files from {ref_root}")

    # FMD.score wants directory PATHS (not lists of files). Pass the parent dirs.
    ref_dir = Path(ref_paths[0]).parent
    gen_dir = Path(gen_paths[0]).parent
    score = fmd.score(reference_path=str(ref_dir), test_path=str(gen_dir))

    out_path = Path(args.output) if args.output \
        else Path(args.generated_dir).parent / "eval_test_fmd.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "fmd": float(score),
        "backbone": args.backbone,
        "device": device,
        "n_generated": len(gen_paths),
        "n_reference": len(ref_paths),
        "generated_dir": str(Path(args.generated_dir).resolve()),
        "reference_dir": str(ref_root.resolve()),
    }, indent=2))
    print(f"FMD = {score:.4f}  →  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
