"""Verify the GEMM-based retriever against a brute-force reference."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hangman.config import ALPHABET, LETTER_TO_ID, MASK_TOKEN, MAX_WORD_LEN, N_LETTERS, PAD_TOKEN
from hangman.data import load_words
from hangman.retrieval import LexiconRetriever

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)


def brute_force(lexicon: list[str], board: str, guessed: set[str]) -> tuple[np.ndarray, int]:
    """Reference implementation: scan the lexicon, count letters in hidden slots."""
    counts = np.zeros(N_LETTERS)
    n_candidates = 0
    for word in lexicon:
        if len(word) != len(board):
            continue
        ok = True
        for shown, actual in zip(board, word):
            if shown == "_":
                if actual in guessed:
                    ok = False
                    break
            elif shown != actual:
                ok = False
                break
        if not ok:
            continue
        n_candidates += 1
        for shown, actual in zip(board, word):
            if shown == "_":
                counts[LETTER_TO_ID[actual]] += 1
    return counts / max(n_candidates, 1), n_candidates


def encode_state(board: str, guessed: set[str]) -> tuple[torch.Tensor, torch.Tensor]:
    row = np.full(MAX_WORD_LEN, PAD_TOKEN, dtype=np.int64)
    for i, ch in enumerate(board):
        row[i] = MASK_TOKEN if ch == "_" else LETTER_TO_ID[ch]
    g = np.zeros(N_LETTERS, dtype=bool)
    for ch in guessed:
        g[LETTER_TO_ID[ch]] = True
    return (
        torch.from_numpy(row).unsqueeze(0).to(DEVICE),
        torch.from_numpy(g).unsqueeze(0).to(DEVICE),
    )


def main() -> None:
    lexicon = load_words("train.txt")[:40_000]
    retriever = LexiconRetriever(lexicon, DEVICE)
    print(f"index: {sum(retriever.bucket_sizes.values()):,} words, "
          f"{retriever.memory_mb():.1f} MB")

    failures = 0
    for trial in range(60):
        word = lexicon[rng.integers(len(lexicon))]
        distinct = sorted(set(word))
        n_reveal = rng.integers(0, len(distinct))
        revealed = set(rng.choice(distinct, size=n_reveal, replace=False)) if n_reveal else set()
        wrong_pool = [c for c in ALPHABET if c not in word]
        n_wrong = int(rng.integers(0, 4))
        wrong = set(rng.choice(wrong_pool, size=n_wrong, replace=False)) if n_wrong else set()
        guessed = revealed | wrong
        board = "".join(c if c in revealed else "_" for c in word)
        if "_" not in board:
            continue

        want_prior, want_n = brute_force(lexicon, board, guessed)
        b, g = encode_state(board, guessed)
        got_prior, got_n = retriever.letter_prior(b, g)
        got_prior = got_prior[0].cpu().numpy()
        got_n = int(got_n[0].item())

        if got_n != want_n or not np.allclose(got_prior, want_prior, atol=2e-2):
            failures += 1
            print(f"  MISMATCH trial {trial}: board={board!r} guessed={sorted(guessed)}")
            print(f"    n_candidates got={got_n} want={want_n}")
            print(f"    max abs diff {np.abs(got_prior - want_prior).max():.4f}")

    print(f"\n{'PASS' if failures == 0 else f'{failures} FAILURES'} "
          f"(60 randomised states vs brute force)")


if __name__ == "__main__":
    main()
