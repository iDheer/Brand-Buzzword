"""Corpus loading, deterministic splitting and dense word encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import LETTER_TO_ID, MAX_WORD_LEN, N_LETTERS, PAD_TOKEN


def load_words(path: str | Path) -> list[str]:
    """Read a competition word list (one lowercase token per line)."""
    with open(path, "r", encoding="utf-8") as handle:
        words = [line.strip().lower() for line in handle if line.strip()]
    return words


def split_train_val(
    words: list[str], n_val: int, seed: int
) -> tuple[list[str], list[str]]:
    """Carve a held-out validation set out of the training corpus.

    The validation words are never used to fit the model, so the win rate we
    measure on them is an honest estimate of performance on unseen vocabulary,
    which is exactly what the private leaderboard measures.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(words))
    val_idx = set(order[:n_val].tolist())
    train = [w for i, w in enumerate(words) if i not in val_idx]
    val = [words[i] for i in order[:n_val]]
    return train, val


def encode_words(words: list[str], max_len: int = MAX_WORD_LEN) -> np.ndarray:
    """Encode words into a dense ``(n_words, max_len)`` uint8 matrix.

    Positions past a word's length hold :data:`PAD_TOKEN`.

    **Non-letter characters also map to** :data:`PAD_TOKEN`. The rules state
    that spaces, digits and punctuation "are shown to you from the start; you
    only ever guess a-z", so such a slot is never hidden, never guessable, and
    must not block the win condition. Encoding it as padding gives exactly that
    behaviour: the engine never masks it, and the encoder ignores it. The
    provided corpora are pure a-z, but the hidden evaluation set is described as
    brand names, which routinely contain spaces, so this path must not crash.
    """
    out = np.full((len(words), max_len), PAD_TOKEN, dtype=np.uint8)
    for row, word in enumerate(words):
        for col, char in enumerate(word[:max_len]):
            out[row, col] = LETTER_TO_ID.get(char, PAD_TOKEN)
    return out


def word_lengths(words: list[str], max_len: int = MAX_WORD_LEN) -> np.ndarray:
    return np.asarray([min(len(w), max_len) for w in words], dtype=np.int16)


def letter_bitmasks(words: list[str]) -> np.ndarray:
    """26-bit set membership mask per word (bit *i* set iff letter *i* occurs)."""
    out = np.zeros(len(words), dtype=np.int32)
    for row, word in enumerate(words):
        acc = 0
        for char in set(word):
            if char in LETTER_TO_ID:
                acc |= 1 << LETTER_TO_ID[char]
        out[row] = acc
    return out


def corpus_letter_frequency(words: list[str]) -> np.ndarray:
    """Document frequency of each letter: P(letter occurs in a random word).

    Used only to seed the round-0 bootstrap policy, never as a final predictor.
    """
    counts = np.zeros(N_LETTERS, dtype=np.float64)
    for word in words:
        for char in set(word):
            if char in LETTER_TO_ID:
                counts[LETTER_TO_ID[char]] += 1.0
    return counts / max(len(words), 1)
