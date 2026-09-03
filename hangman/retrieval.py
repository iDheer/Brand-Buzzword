"""Retrieval prior over the training lexicon.

Idea
----
The board plus the ruled-out letters define a hard constraint on what the word
can be. Every training word of the same length either satisfies it or does not:

* a revealed slot pins one specific letter;
* a still-hidden slot can hold *any letter that has not been guessed*, because
  a correct guess reveals all of its occurrences, so a hidden slot is never a
  guessed letter.

The surviving words are a sample from the posterior over the answer, and the
letter frequencies within them are a strong prior for the next guess. Test
words are disjoint from the training lexicon, so this never degenerates into a
lookup: what transfers is shared spelling structure, not the words themselves.

Implementation
--------------
Done naively this is a regex scan per state and is hopelessly slow at 2M+
states. Both halves instead reduce to a dense matrix product.

Encode every length-``L`` training word as a flat one-hot row of width
``L * 26``. Encode a query board as an *allowed-symbol* vector of the same
width (1 where a slot may hold that letter). Then

    consistency = onehot @ allowed.T          # (n_words, n_queries)

counts matched slots, and a word is a candidate exactly when it matches all
``L`` of them. Projecting the candidate mask back through the same one-hot
matrix,

    letter_counts = mask.T @ onehot           # (n_queries, L * 26)

yields, for every query and every slot, how many candidates put each letter
there. Masking to the hidden slots and summing gives the prior. Two GEMMs per
length bucket, which the GPU eats.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import MASK_TOKEN, MAX_WORD_LEN, N_LETTERS, PAD_TOKEN


class LexiconRetriever:
    """Length-bucketed constraint-satisfaction prior over a word list."""

    def __init__(
        self,
        words: list[str],
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.buckets: dict[int, torch.Tensor] = {}

        by_length: dict[int, list[str]] = {}
        for word in words:
            length = len(word)
            # The flat one-hot encoding below is a-z byte arithmetic.
            if 1 <= length <= MAX_WORD_LEN and word.isalpha() and word.islower():
                by_length.setdefault(length, []).append(word)

        for length, group in by_length.items():
            codes = np.frombuffer("".join(group).encode("ascii"), dtype=np.uint8)
            codes = codes.reshape(len(group), length).astype(np.int64) - ord("a")
            flat = np.arange(length)[None, :] * N_LETTERS + codes
            onehot = np.zeros((len(group), length * N_LETTERS), dtype=np.float32)
            np.put_along_axis(onehot, flat, 1.0, axis=1)
            self.buckets[length] = torch.from_numpy(onehot).to(self.device, dtype)

        self.bucket_sizes = {k: v.shape[0] for k, v in self.buckets.items()}

    def memory_mb(self) -> float:
        return sum(t.numel() * t.element_size() for t in self.buckets.values()) / 1e6

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def letter_prior(
        self, board: torch.Tensor, guessed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prior, n_candidates)``.

        ``prior[b, c]`` is the expected number of hidden slots holding letter
        ``c`` among consistent training words, normalised by the candidate
        count, i.e. an estimate of ``P(c is still hidden in the answer)``.
        ``n_candidates`` is returned so callers can tell a confident prior
        (thousands of candidates) from a vacuous one (none at all).
        """
        batch = board.shape[0]
        prior = torch.zeros((batch, N_LETTERS), device=board.device, dtype=torch.float32)
        n_candidates = torch.zeros(batch, device=board.device, dtype=torch.float32)

        lengths = (board != PAD_TOKEN).sum(dim=1)
        for length in torch.unique(lengths).tolist():
            bucket = self.buckets.get(int(length))
            if bucket is None:
                continue
            rows = (lengths == length).nonzero(as_tuple=True)[0]
            sub_board = board.index_select(0, rows)[:, :length]
            sub_guessed = guessed.index_select(0, rows)

            n_queries = rows.numel()
            allowed = torch.zeros(
                (n_queries, length, N_LETTERS), device=board.device, dtype=self.dtype
            )
            hidden = sub_board == MASK_TOKEN
            # Hidden slot: any letter not yet guessed is admissible.
            allowed[hidden] = (~sub_guessed).unsqueeze(1).expand(-1, length, -1)[hidden].to(self.dtype)
            # Revealed slot: exactly the letter shown.
            revealed_rows, revealed_cols = (~hidden).nonzero(as_tuple=True)
            allowed[revealed_rows, revealed_cols, sub_board[~hidden]] = 1

            flat_allowed = allowed.reshape(n_queries, length * N_LETTERS)

            matched = bucket @ flat_allowed.transpose(0, 1)        # (n_words, n_queries)
            candidate = (matched.float() >= length - 0.5)
            counts = candidate.to(self.dtype).transpose(0, 1) @ bucket   # (n_queries, L*26)
            counts = counts.float().reshape(n_queries, length, N_LETTERS)
            counts = counts * hidden.unsqueeze(-1).float()
            letter_counts = counts.sum(dim=1)

            total = candidate.sum(dim=0).float()
            prior[rows] = letter_counts / total.clamp_min(1.0).unsqueeze(1)
            n_candidates[rows] = total

        return prior, n_candidates


class RetrievalPolicy:
    """The retrieval prior used on its own, as a reference baseline."""

    def __init__(self, retriever: LexiconRetriever, fallback=None) -> None:
        self.retriever = retriever
        self.fallback = fallback

    @torch.no_grad()
    def __call__(self, board: torch.Tensor, guessed: torch.Tensor) -> torch.Tensor:
        prior, n_candidates = self.retriever.letter_prior(board, guessed)
        logits = torch.log(prior.clamp(1e-6, 1.0))
        if self.fallback is not None:
            # With no consistent training word left the prior says nothing.
            empty = (n_candidates < 0.5).unsqueeze(1)
            logits = torch.where(empty, self.fallback(board, guessed), logits)
        return logits


class HybridPolicy:
    """Neural policy blended with the retrieval prior in log-odds space.

    The blend weight is annealed by how much evidence the lexicon actually
    provides: with thousands of consistent words the prior is worth listening
    to, with a handful it is noise, and with none it is silent. ``alpha``
    scales the maximum influence the prior is ever allowed to have.
    """

    def __init__(
        self,
        neural,
        retriever: LexiconRetriever,
        alpha: float = 0.5,
        evidence_scale: float = 20.0,
    ) -> None:
        self.neural = neural
        self.retriever = retriever
        self.alpha = alpha
        self.evidence_scale = evidence_scale

    @torch.no_grad()
    def __call__(self, board: torch.Tensor, guessed: torch.Tensor) -> torch.Tensor:
        neural_logits = self.neural(board, guessed)
        prior, n_candidates = self.retriever.letter_prior(board, guessed)

        p = prior.clamp(1e-4, 1.0 - 1e-4)
        prior_logits = torch.log(p) - torch.log1p(-p)

        # 0 with no candidates, saturating towards 1 as evidence accumulates.
        confidence = (n_candidates / (n_candidates + self.evidence_scale)).unsqueeze(1)
        weight = self.alpha * confidence
        return (1.0 - weight) * neural_logits + weight * prior_logits
