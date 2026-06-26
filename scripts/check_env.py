"""Environment smoke test for jazz-piano.

Run this before any training run on a fresh machine. It verifies:

  - Python version
  - torch + accelerator visibility (CUDA / ROCm-HIP / CPU)
  - VRAM
  - That the PerformanceRNN baseline can be instantiated and run a tiny
    forward pass on the chosen device
  - That the PiJAMA tokenized cache is reachable from the configured
    ``data.processed_root``
  - That a config loads, when one is passed with ``--config``

Exit code 0 on success, non-zero (with a list of failures) otherwise.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --skip-model-forward    # quicker
    python scripts/check_env.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SKIP = "[SKIP]"


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"{PASS} {msg}")

    def fail(self, msg: str, detail: str | None = None) -> None:
        print(f"{FAIL} {msg}")
        if detail:
            for line in detail.splitlines():
                print(f"       {line}")
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        print(f"{WARN} {msg}")
        self.warnings.append(msg)

    def skip(self, msg: str) -> None:
        print(f"{SKIP} {msg}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_python(report: Report) -> None:
    v = sys.version_info
    if v < (3, 10):
        report.fail(f"Python {v.major}.{v.minor} is too old (need >= 3.10)")
    else:
        report.ok(f"Python {v.major}.{v.minor}.{v.micro}")


def check_torch(report: Report) -> tuple[object | None, str]:
    """Return (torch_module, device_str) so downstream checks can reuse."""
    try:
        import torch
    except Exception as e:
        report.fail("torch not importable", str(e))
        return None, "cpu"
    report.ok(f"torch {torch.__version__}")
    hip = getattr(torch.version, "hip", None)
    cuda = torch.version.cuda
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        backend = "ROCm/HIP" if hip else "CUDA"
        report.ok(f"{backend} visible: {n} device(s), GPU0='{name}', VRAM={vram:.1f} GB")
        if hip:
            report.ok(f"  HIP version = {hip}")
        else:
            report.ok(f"  CUDA version = {cuda}")
        return torch, "cuda"
    report.warn("No GPU visible — falling back to CPU. Training will be slow.")
    return torch, "cpu"


def check_dtype_amp(torch_mod, device: str, report: Report) -> None:
    if torch_mod is None or device != "cuda":
        report.skip("AMP dtype check (no CUDA)")
        return
    try:
        x = torch_mod.randn(8, 8, device=device, dtype=torch_mod.bfloat16) @ \
            torch_mod.randn(8, 8, device=device, dtype=torch_mod.bfloat16)
        x.sum().item()
        report.ok("bfloat16 matmul OK (preferred on H100 / Ampere)")
    except Exception as e:
        report.warn(f"bfloat16 matmul failed: {e}")
    try:
        x = torch_mod.randn(8, 8, device=device, dtype=torch_mod.float16) @ \
            torch_mod.randn(8, 8, device=device, dtype=torch_mod.float16)
        x.sum().item()
        report.ok("float16 matmul OK")
    except Exception as e:
        report.warn(f"float16 matmul failed: {e}")


def check_models_instantiate(report: Report, torch_mod, device: str) -> None:
    """Instantiate the PerformanceRNN baseline and run a small forward pass."""
    if torch_mod is None:
        report.skip("model instantiation (no torch)")
        return
    try:
        from src.model.performance_rnn import PerformanceRNN, PerformanceRNNConfig
        cfg = PerformanceRNNConfig(vocab_size=413, hidden_size=64, num_layers=2, dropout=0.0)
        mdl = PerformanceRNN(cfg).to(device).eval()
        n_params = sum(p.numel() for p in mdl.parameters())
        x = torch_mod.zeros(1, 64, dtype=torch_mod.long, device=device)
        with torch_mod.no_grad():
            _ = mdl(x)
        report.ok(f"PerformanceRNN OK ({n_params/1e6:.2f} M params, dummy forward)")
    except Exception:
        report.fail("PerformanceRNN instantiation/forward failed",
                    traceback.format_exc(limit=2))


def check_dataset(report: Report) -> None:
    cache = _REPO_ROOT / "data" / "processed" / "pijama"
    if not cache.exists():
        report.warn(f"{cache} not present — training will fail until you sync it. "
                    "See `scripts/sync_data.sh` (or rsync/S3).")
        return
    splits = [d for d in cache.iterdir() if d.is_dir()]
    if not splits:
        report.warn(f"{cache} exists but contains no split subdirectories")
        return
    report.ok(f"PiJAMA cache at {cache} with splits: "
              f"{', '.join(sorted(d.name for d in splits))}")


def check_config(report: Report, config_path: str | None) -> None:
    if config_path is None:
        report.skip("config parse check (pass --config path/to/config.yaml to enable)")
        return
    p = Path(config_path)
    if not p.exists():
        report.fail(f"config not found: {p}")
        return
    try:
        import yaml
        cfg = yaml.safe_load(p.read_text())
        keys = list(cfg.keys()) if isinstance(cfg, dict) else []
        report.ok(f"config {p.name} loads ({len(keys)} top-level keys)")
    except Exception as e:
        report.fail(f"config {p.name} failed to parse", str(e))


def check_requirements(report: Report) -> None:
    for pkg in ("numpy", "yaml", "pretty_midi", "mido", "muspy", "einops",
                "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            importlib.import_module(pkg)
            report.ok(f"import {pkg}")
        except Exception as e:
            report.fail(f"import {pkg} failed", str(e))


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="A config to test parsing.")
    p.add_argument("--skip-model-forward", action="store_true",
                   help="Skip the dummy model instantiation step.")
    args = p.parse_args()

    print(f"== jazz-piano environment check ==")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Working dir: {os.getcwd()}")
    print(f"Repo root:   {_REPO_ROOT}\n")
    report = Report()

    check_python(report)
    check_requirements(report)
    torch_mod, device = check_torch(report)
    check_dtype_amp(torch_mod, device, report)
    check_dataset(report)
    check_config(report, args.config)
    if not args.skip_model_forward:
        check_models_instantiate(report, torch_mod, device)
    else:
        report.skip("model instantiation (--skip-model-forward)")

    print()
    if report.failures:
        print(f"!! {len(report.failures)} failure(s):")
        for f in report.failures:
            print(f"  - {f}")
        return 1
    if report.warnings:
        print(f"-- {len(report.warnings)} warning(s) (non-fatal):")
        for w in report.warnings:
            print(f"  - {w}")
    print("\nAll critical checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
