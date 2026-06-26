"""PiJAMA downloader that wraps zenodo_get for reproducible data staging.

The PiJAMA blueprint mandates that we fetch the Zenodo release programmatically,
record the checksum of every artifact, and keep extraction logic deterministic.
This module exposes both a CLI (`python -m src.data.pijama_download`) and a
`download_pijama` helper that other scripts (e.g., preprocessing notebooks) can
re-use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse
import zipfile

import requests
from zenodo_get import download as zenodo_download

LOGGER = logging.getLogger(__name__)
DEFAULT_RECORD_ID = "8354955"
DEFAULT_METADATA_URL = "https://almostimplemented.github.io/PiJAMA/pijama.csv"
DEFAULT_FILE_GLOB = ("midi_hawthorne.zip",)


@dataclass(frozen=True)
class ArtifactInfo:
    """Describes a fetched Zenodo file."""

    filename: str
    bytes: int
    md5: str
    variant: str


def md5sum(path: Path) -> str:
    """Compute the MD5 checksum for a file."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _download_archives(
    record_id: str,
    output_dir: Path,
    file_glob: Sequence[str],
    force: bool,
    max_retries: int,
) -> list[Path]:
    """Download PiJAMA zip archives via zenodo_get."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading record %s into %s", record_id, output_dir)
    zenodo_download(
        record=record_id,
        output_dir=str(output_dir),
        file_glob=tuple(file_glob),
        md5=False,
        start_fresh=force,
        max_http_retries=max_retries,
        verbosity=1,
    )
    archives = sorted(output_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(
            f"No *.zip files were downloaded for record {record_id} into {output_dir}."
        )
    return archives


def _extract_archives(archives: Iterable[Path], target_root: Path, force: bool) -> list[Path]:
    """Extract each archive under a variant-specific folder."""
    extracted_dirs: list[Path] = []
    for archive in archives:
        variant_dir = target_root / archive.stem
        if variant_dir.exists():
            if force:
                LOGGER.info("Removing existing extracted folder %s", variant_dir)
                shutil.rmtree(variant_dir)
            else:
                LOGGER.info("Skipping extraction for %s (already present)", variant_dir)
                extracted_dirs.append(variant_dir)
                continue
        LOGGER.info("Extracting %s → %s", archive.name, variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zipped:
            zipped.extractall(variant_dir)
        extracted_dirs.append(variant_dir)
    return extracted_dirs


def _download_metadata(metadata_url: str, metadata_dir: Path, force: bool) -> Path:
    """Download the CSV metadata that reports agreement and duration."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    filename = "pijama_metadata.csv"
    destination = metadata_dir / filename
    if destination.exists() and not force:
        LOGGER.info("Metadata already exists at %s; skipping download.", destination)
        return destination
    LOGGER.info("Fetching metadata CSV from %s", metadata_url)
    response = requests.get(metadata_url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    LOGGER.info("Saved metadata CSV (%d bytes).", destination.stat().st_size)
    return destination


def _write_manifest(
    manifest_path: Path,
    record_id: str,
    artifacts: Sequence[ArtifactInfo],
    metadata_path: Path | None,
) -> None:
    manifest = {
        "record_id": record_id,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [artifact.__dict__ for artifact in artifacts],
        "metadata_path": str(metadata_path) if metadata_path else None,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote manifest to %s", manifest_path)


def download_pijama(
    output_dir: Path,
    record_id: str = DEFAULT_RECORD_ID,
    metadata_url: str = DEFAULT_METADATA_URL,
    file_glob: Sequence[str] = DEFAULT_FILE_GLOB,
    force: bool = False,
    max_retries: int = 5,
) -> Path:
    """Public API entrypoint used by the Makefile target."""
    output_dir = output_dir.resolve()
    raw_archives_dir = output_dir / f"zenodo_{record_id}"
    archives = _download_archives(record_id, raw_archives_dir, file_glob, force, max_retries)
    artifacts: list[ArtifactInfo] = []
    for archive in archives:
        artifacts.append(
            ArtifactInfo(
                filename=str(archive.relative_to(output_dir)),
                bytes=archive.stat().st_size,
                md5=md5sum(archive),
                variant=archive.stem,
            )
        )
    extracted_dirs = _extract_archives(archives, output_dir, force)
    metadata_path: Path | None = None
    if metadata_url:
        metadata_path = _download_metadata(metadata_url, output_dir / "metadata", force)
    _write_manifest(output_dir / "pijama_download_manifest.json", record_id, artifacts, metadata_path)
    LOGGER.info(
        "Finished downloading PiJAMA: %d archives, %d extracted folders.",
        len(artifacts),
        len(extracted_dirs),
    )
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the PiJAMA dataset via Zenodo.")
    parser.add_argument("--record", default=DEFAULT_RECORD_ID, help="Zenodo record ID to fetch.")
    parser.add_argument(
        "--output-dir",
        default="data/raw/pijama",
        help="Root folder where archives + metadata will be stored.",
    )
    parser.add_argument(
        "--metadata-url",
        default=DEFAULT_METADATA_URL,
        help="URL to the PiJAMA metadata CSV (set empty to skip)",
    )
    parser.add_argument(
        "--file-glob",
        nargs="+",
        default=list(DEFAULT_FILE_GLOB),
        help="Glob patterns selecting which files from the record to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives and re-extract even if they exist.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="How many HTTP retries zenodo_get should attempt per file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit debug-level logs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    try:
        download_pijama(
            output_dir=Path(args.output_dir),
            record_id=str(args.record),
            metadata_url=args.metadata_url,
            file_glob=args.file_glob,
            force=args.force,
            max_retries=args.max_retries,
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        LOGGER.error("PiJAMA download failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
