"""Reference policies.

These are **not** the submitted model. They serve two purposes:

1. calibrated lower bounds so we can quantify what the neural model actually
   buys us, and
2. a bootstrap policy for DAgger round 0, before any network exists.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import LETTER_TO_ID, MASK_TOKEN, MAX_WORD_LEN, N_LETTERS, PAD_TOKEN


class LengthConditionedFrequencyPolicy:
    """``P(letter occurs | word length)`` estimated on the training corpus.

    A static ordering per length: no board information is used beyond the word
    length. This is the classic frequency baseline and the weakest thing that
    is still sensible.
    """

    def __init__(self, words: list[str], device: torch.device | str = "cuda",
                 smoothing: float = 5.0) -> None:
        table = np.full((MAX_WORD_LEN + 1, N_LETTERS), smoothing, dtype=np.float64)
        totals = np.full(MAX_WORD_LEN + 1, 2.0 * smoothing, dtype=np.float64)
        for word in words:
            length = min(len(word), MAX_WORD_LEN)
            totals[length] += 1.0
            for char in set(word):
                if char in LETTER_TO_ID:
                    table[length, LETTER_TO_ID[char]] += 1.0
        probs = table / totals[:, None]

        # Lengths that are rare in the corpus fall back to the global profile.
        global_profile = probs.mean(axis=0)
        for length in range(MAX_WORD_LEN + 1):
            if totals[length] < 50.0:
                probs[length] = global_profile

        self.log_probs = torch.tensor(
            np.log(probs), dtype=torch.float32, device=device
        )

    def __call__(self, board: torch.Tensor, guessed: torch.Tensor) -> torch.Tensor:
        lengths = (board != PAD_TOKEN).sum(dim=1).clamp(max=MAX_WORD_LEN)
        return self.log_probs.index_select(0, lengths)


class PositionalNGramPolicy:
    """Board-aware statistical baseline.

    Scores a letter by how often it completes the observed pattern, using
    length-bucketed positional character statistics plus the set of letters
    already ruled out. Purely statistical -- included to measure the headroom
    the neural model has to beat.
    """

    def __init__(self, words: list[str], device: torch.device | str = "cuda",
                 smoothing: float = 1.0) -> None:
        # counts[length, position, letter]
        counts = np.full(
            (MAX_WORD_LEN + 1, MAX_WORD_LEN, N_LETTERS), smoothing, dtype=np.float32
        )
        for word in words:
            length = min(len(word), MAX_WORD_LEN)
            for pos, char in enumerate(word[:MAX_WORD_LEN]):
                if char in LETTER_TO_ID:
                    counts[length, pos, LETTER_TO_ID[char]] += 1.0
        counts /= counts.sum(axis=2, keepdims=True)
        self.pos_probs = torch.tensor(counts, dtype=torch.float32, device=device)

    def __call__(self, board: torch.Tensor, guessed: torch.Tensor) -> torch.Tensor:
        lengths = (board != PAD_TOKEN).sum(dim=1).clamp(max=MAX_WORD_LEN)
        # (B, L, 26) positional distributions for each game's word length
        probs = self.pos_probs.index_select(0, lengths)
        hidden = (board == MASK_TOKEN).unsqueeze(-1)
        # Probability that a hidden slot is NOT the letter, per slot.
        not_letter = torch.where(hidden, 1.0 - probs, torch.ones_like(probs))
        # Noisy-OR across hidden slots: P(letter appears somewhere hidden).
        present = 1.0 - not_letter.prod(dim=1)
        return torch.log(present.clamp_min(1e-9))


class UniformRandomPolicy:
    """Pure noise -- the absolute floor, used in unit tests."""

    def __call__(self, board: torch.Tensor, guessed: torch.Tensor) -> torch.Tensor:
        return torch.rand((board.shape[0], N_LETTERS), device=board.device)
