"""Compute OA / KLD / FMD on a directory of Aria-decoded MIDI generations.

Walks ``$gen_dir/T<t>_k<k>_p<p>/prompt_<i>/var_<j>.mid`` (the layout
produced by ``src.sample_aria_sweep``), extracts the same handcrafted
distributional features used in the v1 LSTM evaluation (``project3/src/metrics``),
and compares each generation against the TEST-set's empirical features.

Outputs two CSVs:

* ``--out`` (per-clip): one row per (T, top_k, min_p, prompt_id, var_id)
  with the OA / forward-KLD / reverse-KLD averaged across features, plus
  the per-feature columns ``oa_<feature>`` / ``kld_<feature>``. FMD lives
  on a per-setting summary because it's a population-level statistic.
* ``${out%.csv}_summary.csv``: one row per (T, top_k, min_p) with mean
  OA, mean KLDs and FMD for that population vs the TEST set.

OA / KLD design choices (port from project3/src/metrics):

* Use a 13-dim handcrafted feature vector per clip (pitch-class hist,
  pitch range, IOI hist, note-density, velocity stats, …) — see
  ``_FEATURE_SPECS`` in ``ariautils_features.compute_clip_features``.
* For OA / KLD: take the *Euclidean distance distribution* between the
  TEST set (intra-set pairwise distances) and the (gen, TEST) cross-set
  distances, KDE-smooth both, integrate the min for OA and the
  log-ratio for KLD. This is the same protocol the LSTM rows use, so
  the headline table is apples-to-apples.

FMD design choice:

* If the ``frechet-music-distance`` (CLaMP-2) library is importable we
  use it — it's the same backbone the v1 paper used. Else fall back to
  the AriaEmbedder route (Aria-medium-embedding's last-token hidden
  state). If neither is available we emit NaN and warn — the user
  prefers a smaller correct table over a larger one with bogus
  numbers.
* The TEST-set embedding pool is computed ONCE and cached at
  ``$gen_dir/../test_emb_cache.npy`` so the next variant reuses it.

The implementation is intentionally model-agnostic — it consumes
.mid/.midi files and never touches Aria tokens. That keeps it usable
against the LSTM, NMT, SMDIM, etc. generations too (we just need to
dump their outputs to MIDI).

Usage:
    python -m src.eval_aria_metrics \
        --gen-dir path/to/runs/v1/stageC/sweep \
        --test-midi-dir path/to/data/pijama_raw_hawthorne/test \
        --out path/to/runs/v1/stageC/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Don't import pretty_midi at module load — it's slow and the import is
# only needed inside main()/helper. Same for torch / aria_embedder.


# ---------------------------------------------------------------------------
# Handcrafted features (ported from project3/src/metrics/features.py).
# The v1 implementation operated on PerformanceTokenizer token sequences;
# this one operates on a list of (pitch, start, end, velocity) tuples, so
# the same code can run on any MIDI without a tokenizer round-trip.
# ---------------------------------------------------------------------------

# Note-length bins in beats (4=whole, 0.25=16th, plus dotted+triplet).
_NOTE_LENGTH_BEATS = np.array(
    [4.0, 2.0, 1.0, 0.5, 0.25, 3.0, 1.5, 0.75, 0.375,
     4.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0],
    dtype=np.float32,
)


_FEATURE_SPECS: Dict[str, int] = {
    "pitch_count":            1,
    "pitch_range":            1,
    "pitch_class_hist":      12,
    "pitch_class_transition":144,
    "note_density":           1,
    "note_count":             1,
    "avg_pitch_interval":     1,
    "pitch_interval_hist":   12,
    "avg_ioi":                1,
    "ioi_hist":              12,
    "note_length_hist":      12,
    "note_length_transition":144,
    "velocity_stats":         2,
}


def _normalize(vec: np.ndarray) -> np.ndarray:
    total = vec.sum()
    return (vec / total if total > 0 else vec).astype(np.float32)


def _zero_features() -> Dict[str, np.ndarray]:
    return {n: np.zeros(d, dtype=np.float32) for n, d in _FEATURE_SPECS.items()}


def load_midi_notes(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, float]:
    """Return (pitch, start_s, end_s, velocity, duration_s) numpy arrays.

    Pulls all instrument tracks together — Aria outputs single-piano
    MIDIs but we don't enforce that. ``duration_s`` is max(end) across
    all notes (or 1e-3 if empty).
    """
    import pretty_midi
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as e:
        raise RuntimeError(f"failed to load {path}: {e}") from e

    pitches: List[int] = []
    starts: List[float] = []
    ends: List[float] = []
    vels: List[int] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            pitches.append(int(note.pitch))
            starts.append(float(note.start))
            ends.append(float(note.end))
            vels.append(int(note.velocity))
    if not pitches:
        return (np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                1e-3)
    duration = max(ends) if ends else 1e-3
    return (np.array(pitches, dtype=np.int32),
            np.array(starts, dtype=np.float32),
            np.array(ends, dtype=np.float32),
            np.array(vels, dtype=np.int32),
            max(duration, 1e-3))


def compute_clip_features(path: Path) -> Dict[str, np.ndarray]:
    """Compute the 13-key feature dict for one MIDI file."""
    pitches, starts, ends, vels, dur = load_midi_notes(path)
    if pitches.size == 0:
        return _zero_features()

    sorted_idx = np.argsort(starts)
    starts = starts[sorted_idx]
    pitches_sorted = pitches[sorted_idx]
    durations = np.maximum(ends[sorted_idx] - starts, 1e-3)
    velocities = vels[sorted_idx].astype(np.float32)

    pitch_classes = np.mod(pitches.astype(np.int32), 12)
    pitch_class_hist = _normalize(
        np.histogram(pitch_classes, bins=12, range=(0, 12))[0].astype(np.float32)
    )
    pcs_sorted = np.mod(pitches_sorted.astype(np.int32), 12)
    pct_matrix = np.zeros((12, 12), dtype=np.float32)
    for a, b in zip(pcs_sorted[:-1], pcs_sorted[1:]):
        pct_matrix[int(a), int(b)] += 1.0
    pitch_class_transition = _normalize(pct_matrix.reshape(-1))

    intervals = np.abs(np.diff(pitches_sorted.astype(np.float32)))
    pitch_interval_hist = _normalize(
        np.histogram(intervals, bins=12, range=(0, 12))[0].astype(np.float32)
    )
    avg_interval = float(intervals.mean()) if intervals.size else 0.0

    iois = np.diff(starts)
    ioi_hist = _normalize(
        np.histogram(iois, bins=12, range=(0.0, 1.2))[0].astype(np.float32)
    )
    avg_ioi = float(iois.mean()) if iois.size else 0.0

    positives = iois[iois > 1e-4]
    if positives.size:
        beat_duration = float(np.median(positives))
    else:
        dur_pos = durations[durations > 1e-4]
        beat_duration = float(np.median(dur_pos)) if dur_pos.size else 0.5
    beat_duration = max(beat_duration, 1e-3)

    nl_buckets = np.zeros(len(_NOTE_LENGTH_BEATS), dtype=np.float32)
    nl_indices: List[int] = []
    for d in durations:
        beats = d / beat_duration
        idx = int(np.argmin(np.abs(_NOTE_LENGTH_BEATS - beats)))
        nl_buckets[idx] += 1.0
        nl_indices.append(idx)
    note_length_hist = _normalize(nl_buckets)
    nlt_matrix = np.zeros((12, 12), dtype=np.float32)
    for a, b in zip(nl_indices[:-1], nl_indices[1:]):
        nlt_matrix[a, b] += 1.0
    note_length_transition = _normalize(nlt_matrix.reshape(-1))

    feat = {
        "pitch_count":      np.array([len(np.unique(pitches))], dtype=np.float32),
        "pitch_range":      np.array([pitches.max() - pitches.min()], dtype=np.float32),
        "pitch_class_hist": pitch_class_hist,
        "pitch_class_transition": pitch_class_transition,
        "note_density":     np.array([len(pitches) / max(dur, 1e-3)], dtype=np.float32),
        "note_count":       np.array([len(pitches)], dtype=np.float32),
        "avg_pitch_interval": np.array([avg_interval], dtype=np.float32),
        "pitch_interval_hist": pitch_interval_hist,
        "avg_ioi":          np.array([avg_ioi], dtype=np.float32),
        "ioi_hist":         ioi_hist,
        "note_length_hist": note_length_hist,
        "note_length_transition": note_length_transition,
        "velocity_stats":   np.array([float(velocities.mean()),
                                      float(velocities.std())], dtype=np.float32),
    }
    return feat


# ---------------------------------------------------------------------------
# OA / KLD via pairwise-distance KDE (Yang & Lerch 2018, used in v1 LSTM).
# ---------------------------------------------------------------------------

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None  # type: ignore


def _pairwise_euclidean(vecs: Sequence[np.ndarray]) -> np.ndarray:
    if len(vecs) < 2:
        return np.zeros(1, dtype=np.float32)
    A = np.stack(vecs).astype(np.float32)
    # (N,d) pairwise distances, upper triangular only.
    diff = A[:, None, :] - A[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(A.shape[0], k=1)
    return d[iu].astype(np.float32)


def _pairwise_cross(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    if not a or not b:
        return np.zeros(1, dtype=np.float32)
    A = np.stack(a).astype(np.float32)
    B = np.stack(b).astype(np.float32)
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return d.reshape(-1).astype(np.float32)


def _kde_pdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(grid)
    if values.size == 1 or np.allclose(values, values[0]) or gaussian_kde is None:
        std = max(1e-3, float(np.std(values)) or 1e-3)
        coef = 1.0 / (std * np.sqrt(2.0 * np.pi))
        pdf = coef * np.exp(-0.5 * ((grid - float(values.mean())) / std) ** 2)
    else:
        try:
            kde = gaussian_kde(values)
            pdf = kde(grid)
        except np.linalg.LinAlgError:
            std = max(1e-3, float(np.std(values)) or 1e-3)
            coef = 1.0 / (std * np.sqrt(2.0 * np.pi))
            pdf = coef * np.exp(-0.5 * ((grid - float(values.mean())) / std) ** 2)
    area = np.trapezoid(pdf, grid)
    if area > 0:
        pdf = pdf / area
    return pdf.astype(np.float32)


def _compare_distributions(intra: np.ndarray, inter: np.ndarray,
                           samples: int = 256) -> Dict[str, float]:
    values = np.concatenate([intra, inter])
    if values.size == 0:
        return {"oa": 0.0, "kld_fwd": 0.0, "kld_rev": 0.0}
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        hi = lo + 1e-3
    grid = np.linspace(lo, hi, samples)
    p = _kde_pdf(intra, grid)  # TEST (reference)
    q = _kde_pdf(inter, grid)  # GEN-vs-TEST distances
    oa = float(np.trapezoid(np.minimum(p, q), grid))
    eps = 1e-10
    kld_fwd = float(np.trapezoid(p * np.log((p + eps) / (q + eps)), grid))  # KL(p||q)
    kld_rev = float(np.trapezoid(q * np.log((q + eps) / (p + eps)), grid))  # KL(q||p)
    return {"oa": oa, "kld_fwd": kld_fwd, "kld_rev": kld_rev}


# ---------------------------------------------------------------------------
# FMD — try frechet_music_distance (CLaMP-2). If unavailable, try
# AriaEmbedder (last-token hidden state). Else mark as NaN.
# ---------------------------------------------------------------------------

def _try_compute_fmd(gen_paths: List[Path], test_paths: List[Path],
                     test_emb_cache: Path | None = None,
                     backbone: str = "auto") -> Tuple[float, str]:
    """Return ``(fmd, backend_str)``. ``fmd`` is NaN on failure."""
    if backbone in ("auto", "clamp2"):
        try:
            return _fmd_via_clamp2(gen_paths, test_paths)
        except Exception as e:
            print(f"[fmd] clamp2 backend unavailable: {e}; trying aria emb")
    if backbone in ("auto", "aria"):
        try:
            return _fmd_via_aria(gen_paths, test_paths, test_emb_cache)
        except Exception as e:
            print(f"[fmd] aria backend unavailable: {e}")
    return float("nan"), "none"


def _fmd_via_clamp2(gen_paths: List[Path],
                    test_paths: List[Path]) -> Tuple[float, str]:
    # SERIAL GPU PATH. The frechet_music_distance lib loads CLaMP-2 onto the GPU
    # correctly, but its DatasetLoader.load_dataset_async() spawns a
    # multiprocessing.Pool sized to os.cpu_count() (host cores, e.g. 255) while
    # the container cgroup quota may be ~27 CPUs -> oversubscription + (with CUDA
    # already initialised in the parent) fork deadlock => the ~50min "hang".
    # We bypass that loader: load each MIDI serially and run the GPU model
    # file-by-file. Reused by Track B.
    import numpy as np
    from frechet_music_distance.models.clamp2.clamp2_extractor import CLaMP2Extractor  # type: ignore
    if not gen_paths or not test_paths:
        return float("nan"), "clamp2 (empty)"
    ext = CLaMP2Extractor(verbose=False)

    def _feats(paths):
        out = []
        for pth in paths:
            try:
                mtf = ext._midi_dataset_loader.load_file(str(pth))  # serial, no Pool
                out.append(ext._extract_feature(mtf))               # GPU forward
            except Exception as e:
                print(f"[fmd]   skip {pth.name}: {e}")
        return np.vstack(out) if out else None

    test_E = _feats(test_paths)
    gen_E = _feats(gen_paths)
    if test_E is None or gen_E is None or test_E.shape[0] < 2 or gen_E.shape[0] < 2:
        return float("nan"), "clamp2 (insufficient)"
    score = _frechet_distance(test_E, gen_E)
    return float(score), "clamp2"


def _fmd_via_aria(gen_paths: List[Path], test_paths: List[Path],
                  test_emb_cache: Path | None) -> Tuple[float, str]:
    from src.metrics.aria_embedder import AriaEmbedder
    import torch
    emb = AriaEmbedder(device="auto")
    if test_emb_cache is not None and test_emb_cache.exists():
        test_E = np.load(test_emb_cache)
        print(f"[fmd] test_emb_cache hit ({test_E.shape}) <- {test_emb_cache}")
    else:
        with torch.no_grad():
            test_E_t = emb.encode_midi_files([str(p) for p in test_paths])
        test_E = test_E_t.numpy()
        if test_emb_cache is not None:
            test_emb_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(test_emb_cache, test_E)
    with torch.no_grad():
        gen_E_t = emb.encode_midi_files([str(p) for p in gen_paths])
    gen_E = gen_E_t.numpy()
    if test_E.shape[0] < 2 or gen_E.shape[0] < 2:
        return float("nan"), "aria (insufficient samples)"
    score = _frechet_distance(test_E, gen_E)
    return float(score), "aria"


def _frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Standard Fréchet distance between two Gaussians fitted to X, Y."""
    from scipy.linalg import sqrtm
    mu_x, mu_y = X.mean(0), Y.mean(0)
    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)
    diff = mu_x - mu_y
    # scipy>=1.13 removed the `disp` kwarg and returns just the matrix (older
    # scipy with disp=False returned a (matrix, errest) tuple). Handle both so
    # FMD works regardless of scipy version (pod has a newer scipy).
    covmean = sqrtm(cov_x @ cov_y)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(cov_x + cov_y - 2.0 * covmean))
    return fid


# ---------------------------------------------------------------------------
# Sweep dir walking + driver.
# ---------------------------------------------------------------------------

# Optional ``_s<N>`` suffix lets a sweep record a reverse-step count.
# Backward-compatible: AR/VAE sweeps without the suffix still match (steps=None).
_SETTING_RE = re.compile(
    r"^T(?P<t>[0-9.eE+-]+)_k(?P<k>\d+)_p(?P<p>[0-9.eE+-]+)(?:_s(?P<s>\d+))?$"
)


@dataclass(frozen=True)
class SweepKey:
    T: float
    top_k: int
    min_p: float
    steps: int | None = None

    @property
    def dir_name(self) -> str:
        base = f"T{self.T:g}_k{self.top_k:d}_p{self.min_p:g}"
        return base if self.steps is None else f"{base}_s{self.steps:d}"


def _parse_setting(name: str) -> SweepKey | None:
    m = _SETTING_RE.match(name)
    if not m:
        return None
    s = m.group("s")
    return SweepKey(T=float(m.group("t")), top_k=int(m.group("k")),
                    min_p=float(m.group("p")),
                    steps=int(s) if s is not None else None)


def _list_test_midis(test_midi_dir: Path, limit: int) -> List[Path]:
    paths = sorted(test_midi_dir.glob("*.mid")) + sorted(test_midi_dir.glob("*.midi"))
    return paths[:limit]


def _features_or_none(path: Path) -> Dict[str, np.ndarray] | None:
    try:
        return compute_clip_features(path)
    except Exception as e:
        print(f"[eval] WARN: skipping {path.name}: {e}")
        return None


def _bundle_features(feat_dicts: List[Dict[str, np.ndarray]]
                     ) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {n: [] for n in _FEATURE_SPECS}
    for d in feat_dicts:
        for n in _FEATURE_SPECS:
            out[n].append(d[n])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-dir", required=True,
                   help="Root of the sweep (the dir holding T*_k*_p* subdirs).")
    p.add_argument("--test-midi-dir", required=True,
                   help="TEST-split raw MIDI dir.")
    p.add_argument("--out", required=True,
                   help="Per-clip CSV. Summary CSV goes next to it as "
                        "<out_stem>_summary.csv.")
    p.add_argument("--max-test-clips", type=int, default=200)
    p.add_argument("--bins", type=int, default=128,
                   help="KDE grid resolution.")
    p.add_argument("--variant", default="?",
                   help="Tag stored in the CSV (e.g. v1).")
    p.add_argument("--fmd-backbone", default="auto",
                   choices=["auto", "clamp2", "aria", "none"],
                   help="auto -> try clamp2 then aria; none -> skip FMD.")
    p.add_argument("--test-emb-cache", default=None,
                   help="Path to test-set embedding cache for aria-FMD. "
                        "Defaults to <gen_dir>/../test_emb_cache.npy.")
    p.add_argument("--max-gen-per-setting", type=int, default=0,
                   help="If >0, cap the number of generated MIDIs per "
                        "setting (debug knob).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gen_root = Path(args.gen_dir)
    if not gen_root.exists():
        raise SystemExit(f"gen-dir does not exist: {gen_root}")
    test_paths = _list_test_midis(Path(args.test_midi_dir), args.max_test_clips)
    if not test_paths:
        raise SystemExit(f"No TEST MIDIs in {args.test_midi_dir}")
    print(f"[eval] TEST clips: {len(test_paths)} from {args.test_midi_dir}")

    # --- 1) Build TEST feature pool + intra-set distance pool ---
    test_feats: List[Dict[str, np.ndarray]] = []
    for p in test_paths:
        f = _features_or_none(p)
        if f is not None:
            test_feats.append(f)
    print(f"[eval] TEST features: {len(test_feats)} clips usable")
    test_bundle = _bundle_features(test_feats)
    # Intra-set pairwise distances per feature.
    intra_dists: Dict[str, np.ndarray] = {
        n: _pairwise_euclidean(v) for n, v in test_bundle.items()
    }

    # --- 2) Iterate over every (T, top_k, min_p) subdir ---
    settings: List[Tuple[SweepKey, Path]] = []
    for child in sorted(gen_root.iterdir()):
        if not child.is_dir():
            continue
        key = _parse_setting(child.name)
        if key is None:
            continue
        settings.append((key, child))
    if not settings:
        raise SystemExit(
            f"No T*_k*_p* subdirs under {gen_root}; did sample_aria_sweep "
            "run? "
        )
    print(f"[eval] settings: {len(settings)}")

    per_clip_rows: List[Dict[str, object]] = []
    per_setting_rows: List[Dict[str, object]] = []

    test_emb_cache = Path(args.test_emb_cache) if args.test_emb_cache \
        else gen_root.parent / "test_emb_cache.npy"

    for key, setting_dir in settings:
        print(f"[eval] {key.dir_name}")
        # Collect all (prompt_i, var_j, path) under this setting.
        gen_entries: List[Tuple[int, int, Path]] = []
        for prompt_dir in sorted(setting_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            m = re.match(r"^prompt_(\d+)$", prompt_dir.name)
            if not m:
                continue
            prompt_id = int(m.group(1))
            for mid in sorted(prompt_dir.glob("var_*.mid")):
                vm = re.match(r"^var_(\d+)\.mid$", mid.name)
                if not vm:
                    continue
                var_id = int(vm.group(1))
                gen_entries.append((prompt_id, var_id, mid))
        if args.max_gen_per_setting > 0:
            gen_entries = gen_entries[:args.max_gen_per_setting]
        if not gen_entries:
            print(f"[eval]   no .mid files under {setting_dir}; skipping")
            continue

        gen_feats: List[Dict[str, np.ndarray]] = []
        gen_paths_clean: List[Path] = []
        per_clip_local: List[Tuple[int, int, Path, Dict[str, np.ndarray]]] = []
        for prompt_id, var_id, path in gen_entries:
            f = _features_or_none(path)
            if f is None:
                continue
            gen_feats.append(f)
            gen_paths_clean.append(path)
            per_clip_local.append((prompt_id, var_id, path, f))
        if not gen_feats:
            print(f"[eval]   no usable generations under {setting_dir}; skip")
            continue

        gen_bundle = _bundle_features(gen_feats)

        # --- Per-setting feature-level OA / KLD vs TEST ---
        per_feature: Dict[str, Dict[str, float]] = {}
        for name in _FEATURE_SPECS:
            intra = intra_dists[name]
            inter = _pairwise_cross(gen_bundle[name], test_bundle[name])
            per_feature[name] = _compare_distributions(intra, inter,
                                                      samples=args.bins)
        oa_values = [v["oa"] for v in per_feature.values()]
        kld_fwd_values = [v["kld_fwd"] for v in per_feature.values()]
        kld_rev_values = [v["kld_rev"] for v in per_feature.values()]

        # --- Per-clip OA: distance of clip to TEST mean / pairwise-cross.
        # We re-use the same intra-set pool but compute the inter set as
        # just THIS clip vs TEST; for KLD same protocol.
        for prompt_id, var_id, clip_path, clip_feat in per_clip_local:
            row: Dict[str, object] = {
                "variant": args.variant,
                "T": key.T,
                "top_k": key.top_k,
                "min_p": key.min_p,
                "steps": key.steps,
                "prompt_id": prompt_id,
                "var_id": var_id,
                "midi_path": str(clip_path),
            }
            clip_oas: List[float] = []
            clip_klds_fwd: List[float] = []
            clip_klds_rev: List[float] = []
            for name in _FEATURE_SPECS:
                intra = intra_dists[name]
                inter = _pairwise_cross([clip_feat[name]], test_bundle[name])
                cmp_ = _compare_distributions(intra, inter, samples=args.bins)
                row[f"oa_{name}"] = cmp_["oa"]
                row[f"kld_fwd_{name}"] = cmp_["kld_fwd"]
                row[f"kld_rev_{name}"] = cmp_["kld_rev"]
                clip_oas.append(cmp_["oa"])
                clip_klds_fwd.append(cmp_["kld_fwd"])
                clip_klds_rev.append(cmp_["kld_rev"])
            row["OA"] = float(np.mean(clip_oas))
            row["KLD_forward"] = float(np.mean(clip_klds_fwd))
            row["KLD_reverse"] = float(np.mean(clip_klds_rev))
            row["FMD"] = float("nan")  # FMD is population-level, not per-clip
            per_clip_rows.append(row)

        # --- FMD per setting ---
        if args.fmd_backbone == "none":
            fmd_score = float("nan")
            fmd_backend = "none"
        else:
            fmd_score, fmd_backend = _try_compute_fmd(
                gen_paths_clean, test_paths,
                test_emb_cache=test_emb_cache,
                backbone=args.fmd_backbone,
            )

        summary = {
            "variant": args.variant,
            "T": key.T,
            "top_k": key.top_k,
            "min_p": key.min_p,
            "steps": key.steps,
            "n_gen": len(gen_feats),
            "n_test": len(test_feats),
            "OA_mean": float(np.mean(oa_values)),
            "KLD_forward_mean": float(np.mean(kld_fwd_values)),
            "KLD_reverse_mean": float(np.mean(kld_rev_values)),
            "FMD": fmd_score,
            "FMD_backend": fmd_backend,
        }
        for name in _FEATURE_SPECS:
            summary[f"oa_{name}"] = per_feature[name]["oa"]
        per_setting_rows.append(summary)
        print(f"[eval]   OA={summary['OA_mean']:.4f}  "
              f"KLDf={summary['KLD_forward_mean']:.4f}  "
              f"KLDr={summary['KLD_reverse_mean']:.4f}  "
              f"FMD={fmd_score:.4f} ({fmd_backend})")

    # --- 3) Dump CSVs ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_path, per_clip_rows)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    _write_csv(summary_path, per_setting_rows)
    print(f"[eval] wrote {out_path}  ({len(per_clip_rows)} clip rows)")
    print(f"[eval] wrote {summary_path}  ({len(per_setting_rows)} setting rows)")

    # Best-OA setting summary (for quick eyeballing).
    if per_setting_rows:
        best = max(per_setting_rows, key=lambda r: r["OA_mean"])
        print(f"[eval] best OA: T={best['T']} k={best['top_k']} "
              f"p={best['min_p']}  OA={best['OA_mean']:.4f}  "
              f"FMD={best['FMD']}")
    return 0


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    # Union of all keys, preserving order of appearance.
    seen: Dict[str, None] = {}
    for r in rows:
        for k in r:
            seen.setdefault(k, None)
    cols = list(seen.keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


if __name__ == "__main__":
    raise SystemExit(main())
