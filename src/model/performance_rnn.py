"""PerformanceRNN PyTorch implementation inspired by Oore et al. (2018)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from src.data.tokenizer import VOCAB_SIZE


@dataclass
class PerformanceRNNConfig:
    vocab_size: int = VOCAB_SIZE
    embedding_dim: int = 512
    hidden_size: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    tie_weights: bool = False


class PerformanceRNN(nn.Module):
    """Stacked LSTM for autoregressive performance modeling."""

    def __init__(self, config: PerformanceRNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or PerformanceRNNConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.rnn = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.decoder = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        if self.config.tie_weights:
            if self.config.hidden_size != self.config.embedding_dim:
                raise ValueError("To tie weights, embedding_dim must equal hidden_size")
            self.decoder.weight = self.embedding.weight
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the network forward.

        Args:
            input_ids: LongTensor of shape (batch, seq_len).
            hidden: Optional tuple of hidden/cell states.
        Returns:
            logits: FloatTensor of shape (batch, seq_len, vocab_size)
            hidden: Updated hidden state tuple.
        """
        emb = self.embedding(input_ids)
        rnn_out, hidden = self.rnn(emb, hidden)
        logits = self.decoder(self.dropout(rnn_out))
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        primer: torch.Tensor,
        steps: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens starting from primer."""
        self.eval()
        sequence = primer.clone()
        hidden = None

        if sequence.numel() == 0:
            raise ValueError("Primer tensor must contain at least one token")

        # Warm-up hidden state with all but the last primer token so the
        # continuation depends on the entire primer sequence.
        if sequence.size(1) > 1:
            _, hidden = self.forward(sequence[:, :-1], hidden)
        last_token = sequence[:, -1:]

        for _ in range(steps):
            logits, hidden = self.forward(last_token, hidden)
            next_token = _sample_from_logits(logits[:, -1, :], temperature, top_k)
            sequence = torch.cat([sequence, next_token.unsqueeze(1)], dim=1)
            last_token = next_token.unsqueeze(1)
        return sequence


def _sample_from_logits(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(scaled, k=min(top_k, scaled.size(-1)))
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, choice).squeeze(-1)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


__all__ = ["PerformanceRNN", "PerformanceRNNConfig"]
