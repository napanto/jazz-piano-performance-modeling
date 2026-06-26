"""Compute the PiJAMA 'GT' OA ceiling — i.e., the OA between two random
halves of the test set, exactly analogous to Huang24's Table 2 GT row.

This tells us the maximum-possible mean OA that any generative model could
ever achieve on this dataset under this evaluation, before the inherent
sampling-noise floor of the intra-set distribution kicks in.

Run: python -m src.metrics.gt_ceiling --n-repeats 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.data.tokenizer import PerformanceTokenizer
from src.data.utils import list_token_paths, load_split_ids
from src.metrics.features import FeatureExtractor
from src.metrics.oa_kld import (
    compare_distributions,
    pairwise_euclidean,
    pairwise_euclidean_cross,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", default="data/processed/pijama")
    p.add_argument("--splits-manifest", default="data/splits/pijama_album_splits.json")
    p.add_argument("--split", default="test")
    p.add_argument("--max-clips", type=int, default=200)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="reports/gt_ceiling.json")
    return p.parse_args()


def features_per_clip(paths, extractor: FeatureExtractor) -> dict[str, list[np.ndarray]]:
    features: dict[str, list[np.ndarray]] = {}
    for path in paths:
        sample = extractor.from_json(path)
        for name, value in sample.features.items():
            features.setdefault(name, []).append(value.astype(np.float32))
    return features


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    tokenizer = PerformanceTokenizer()
    extractor = FeatureExtractor(tokenizer)
    allowed_ids = load_split_ids(Path(args.splits_manifest), args.split)
    all_paths = list_token_paths(Path(args.processed_root), args.split, allowed_ids)[: args.max_clips]
    print(f"Total test clips: {len(all_paths)}")
    if len(all_paths) < 20:
        raise SystemExit("not enough clips for a meaningful ceiling")
    print(f"Will split {len(all_paths)} clips into halves and average over {args.n_repeats} random splits")
    feat_cache = features_per_clip(all_paths, extractor)

    per_repeat: list[dict[str, dict[str, float]]] = []
    for rep in range(args.n_repeats):
        shuffled = list(range(len(all_paths)))
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        A = shuffled[:half]
        B = shuffled[half: 2 * half]
        result = {}
        for feat_name, vecs in feat_cache.items():
            a = [vecs[i] for i in A]
            b = [vecs[i] for i in B]
            intra = pairwise_euclidean(a)
            inter = pairwise_euclidean_cross(b, a)
            cmp = compare_distributions(intra, inter, samples=args.bins)
            result[feat_name] = cmp
        per_repeat.append(result)
        print(f"  repeat {rep+1}/{args.n_repeats} done")

    # Aggregate over repeats.
    aggregated: dict[str, dict[str, float]] = {}
    for feat_name in per_repeat[0].keys():
        oas = [per_repeat[r][feat_name]["oa"] for r in range(args.n_repeats)]
        klds = [per_repeat[r][feat_name]["kld"] for r in range(args.n_repeats)]
        aggregated[feat_name] = {
            "oa_mean": float(np.mean(oas)),
            "oa_std":  float(np.std(oas)),
            "kld_mean": float(np.mean(klds)),
            "kld_std": float(np.std(klds)),
        }
    mean_oa_per_repeat = [
        float(np.mean([per_repeat[r][k]["oa"] for k in per_repeat[r]]))
        for r in range(args.n_repeats)
    ]
    aggregated["_aggregate"] = {
        "mean_oa_mean": float(np.mean(mean_oa_per_repeat)),
        "mean_oa_std":  float(np.std(mean_oa_per_repeat)),
        "n_repeats": args.n_repeats,
        "n_clips_per_half": len(all_paths) // 2,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(aggregated, indent=2))

    print()
    print(f"PiJAMA GT ceiling (test split, half vs half):")
    print(f"  mean OA = {aggregated['_aggregate']['mean_oa_mean']:.4f}  ± {aggregated['_aggregate']['mean_oa_std']:.4f}")
    print()
    for k, v in aggregated.items():
        if k == "_aggregate":
            continue
        print(f"  {k:30s} OA = {v['oa_mean']:.4f} ± {v['oa_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
