"""Lightweight wrapper around Aria-medium-embedding (1B LLaMA-3.2,
EleutherAI, Apache-2.0). Loads the model once, exposes batched MIDI ->
embedding inference, used by the re-ranker and as a general-purpose
"musical quality" embedding for FMD-style distances.

Usage:
    embedder = AriaEmbedder(device="cuda", dtype=torch.float16)
    emb = embedder.encode_midi_files(["a.mid", "b.mid"])  # (B, 1024)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch

# Default lookup order:
#   1. $ARIA_MODEL_PATH if set
#   2. <repo_root>/models/aria-medium-embedding (where repo_root is the
#      directory two levels above this file: src/metrics/ -> src/ -> repo)
#   3. "aria-medium-embedding" (caller must `cd` into a dir that contains it,
#      or pass `model_path=` explicitly)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPO = os.environ.get(
    "ARIA_MODEL_PATH",
    str(_REPO_ROOT / "models" / "aria-medium-embedding"),
)
_MAX_SEQ_LEN = 2048


class AriaEmbedder:
    def __init__(
        self,
        model_path: str | Path = _DEFAULT_REPO,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device).eval()
        self.eos_id = self.tokenizer._convert_token_to_id(self.tokenizer.eos_token)

    @torch.no_grad()
    def encode_midi_file(self, path: str | Path) -> torch.Tensor:
        """Return a (d,) embedding tensor for one MIDI file."""
        prompt = self.tokenizer.encode_from_file(str(path), return_tensors="pt")
        ids = prompt.input_ids[:, :_MAX_SEQ_LEN]
        ids[:, -1] = self.eos_id
        out = self.model.forward(input_ids=ids.to(self.device))
        return out[0].squeeze(0).detach().float().cpu()

    @torch.no_grad()
    def encode_midi_files(self, paths: Sequence[str | Path]) -> torch.Tensor:
        """Sequential encode (the Aria README's recipe doesn't show a batched
        path; tokens are variable-length, so batching needs custom padding).
        Returns (N, d) on CPU.
        """
        embs: List[torch.Tensor] = []
        for p in paths:
            try:
                embs.append(self.encode_midi_file(p))
            except Exception as e:
                print(f"  [aria] skipping {p}: {e}")
                continue
        return torch.stack(embs) if embs else torch.empty(0)


__all__ = ["AriaEmbedder"]
