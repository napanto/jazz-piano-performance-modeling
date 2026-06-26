"""Utilities for computing Overlapping Area and KLD between distributions."""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def pairwise_euclidean(vectors: list[np.ndarray]) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros(1, dtype=np.float32)
    dists: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            diff = vectors[i] - vectors[j]
            dists.append(float(np.linalg.norm(diff)))
    return np.array(dists, dtype=np.float32)


def pairwise_euclidean_cross(a_vectors: list[np.ndarray], b_vectors: list[np.ndarray]) -> np.ndarray:
    if not a_vectors or not b_vectors:
        return np.zeros(1, dtype=np.float32)
    dists: list[float] = []
    for vec_a in a_vectors:
        for vec_b in b_vectors:
            dists.append(float(np.linalg.norm(vec_a - vec_b)))
    return np.array(dists, dtype=np.float32)


def _kde_pdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(grid)
    if values.size == 1 or np.allclose(values, values[0]):
        # Manual Gaussian with a heuristic bandwidth
        std = max(1e-3, float(np.std(values)) or 1e-3)
        coef = 1.0 / (std * np.sqrt(2.0 * np.pi))
        pdf = coef * np.exp(-0.5 * ((grid - values[0]) / std) ** 2)
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


def overlap_area(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> float:
    minima = np.minimum(p, q)
    return float(np.trapezoid(minima, grid))


def kl_divergence(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> float:
    eps = 1e-10
    ratio = (p + eps) / (q + eps)
    return float(np.trapezoid(p * np.log(ratio), grid))


def compare_distributions(intra: np.ndarray, inter: np.ndarray, samples: int = 256) -> dict[str, float]:
    values = np.concatenate([intra, inter])
    if values.size == 0:
        return {"oa": 0.0, "kld": 0.0}
    min_val = float(values.min())
    max_val = float(values.max())
    if np.isclose(min_val, max_val):
        max_val = min_val + 1e-3
    grid = np.linspace(min_val, max_val, samples)
    p = _kde_pdf(intra, grid)
    q = _kde_pdf(inter, grid)
    return {
        "oa": overlap_area(p, q, grid),
        "kld": kl_divergence(p, q, grid),
    }


__all__ = [
    "pairwise_euclidean",
    "pairwise_euclidean_cross",
    "overlap_area",
    "kl_divergence",
    "compare_distributions",
]
