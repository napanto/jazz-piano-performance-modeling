"""Shared utilities for accessing processed PiJAMA token files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Set


def load_split_ids(manifest_path: Path | None, split: str | None = None) -> set[str] | None:
    if manifest_path is None:
        return None
    data = json.loads(manifest_path.read_text())
    splits = data.get("splits", {})
    if split is None:
        ids = {str(entry.get("id")) for entries in splits.values() for entry in entries}
    else:
        entries = splits.get(split, [])
        ids = {str(entry.get("id")) for entry in entries}
    return {track_id for track_id in ids if track_id and track_id.lower() != "none"}


def list_token_paths(processed_root: Path, split: str, allowed_ids: set[str] | None = None) -> list[Path]:
    split_dir = processed_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Processed split directory {split_dir} not found")
    token_paths: list[Path] = []
    for json_file in sorted(split_dir.rglob("*.json")):
        if not allowed_ids:
            token_paths.append(json_file)
            continue
        payload = json.loads(json_file.read_text())
        track_id = str(payload.get("meta", {}).get("id"))
        if track_id in allowed_ids:
            token_paths.append(json_file)
    return token_paths


__all__ = ["load_split_ids", "list_token_paths"]
