"""Exact, vectorised Hangman engine.

A single implementation of the rules drives three different jobs:

* **state generation**: every decision point encountered during simulated
  play becomes a supervised training example (DAgger);
* **evaluation**: win rate / efficiency on held-out words;
* **submission**: the chronological guess string required by the CSV schema.

Using one engine for all three removes any chance of train/serve skew.

Rules implemented (per the competition Evaluation section)
---------------------------------------------------------
* A correct guess reveals *every* occurrence of that letter at once.
* Any guess that fails to reveal a new position costs one wrong guess.
* The game terminates the instant the word is complete (win) or the
  :data:`~hangman.config.MAX_WRONG_GUESSES`-th wrong guess is made (loss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .config import (
    ID_TO_LETTER,
    LETTER_TO_ID,
    MASK_TOKEN,
    MAX_WORD_LEN,
    MAX_WRONG_GUESSES,
    N_LETTERS,
    PAD_TOKEN,
)
from .data import encode_words

#: A policy maps a batch of boards + guessed-letter sets to 26 letter scores.
#: Higher score == more likely to be a productive guess.
Policy = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class StateBuffer:
    """Compact record of visited game states.

    ``board`` is the literal encoder input; ``guessed`` and ``contains`` are
    26-bit masks from which the training target is derived on the fly as
    ``contains & ~guessed``, the letters that are still hidden and still
    legal to guess.
    """

    board: np.ndarray      # (n_states, MAX_WORD_LEN) uint8
    guessed: np.ndarray    # (n_states,) int32 bitmask
    contains: np.ndarray   # (n_states,) int32 bitmask
    word_id: np.ndarray    # (n_states,) int32 index into the source corpus

    def __len__(self) -> int:
        return int(len(self.guessed))

    @staticmethod
    def empty() -> "StateBuffer":
        return StateBuffer(
            np.zeros((0, MAX_WORD_LEN), np.uint8),
            np.zeros(0, np.int32),
            np.zeros(0, np.int32),
            np.zeros(0, np.int32),
        )

    @staticmethod
    def concat(buffers: list["StateBuffer"]) -> "StateBuffer":
        buffers = [b for b in buffers if len(b)]
        if not buffers:
            return StateBuffer.empty()
        return StateBuffer(
            np.concatenate([b.board for b in buffers]),
            np.concatenate([b.guessed for b in buffers]),
            np.concatenate([b.contains for b in buffers]),
            np.concatenate([b.word_id for b in buffers]),
        )

    def subsample(self, n: int, rng: np.random.Generator) -> "StateBuffer":
        if n >= len(self):
            return self
        idx = rng.choice(len(self), size=n, replace=False)
        return StateBuffer(
            self.board[idx], self.guessed[idx], self.contains[idx], self.word_id[idx]
        )


@dataclass
class PlayResult:
    """Outcome of one batch of games."""

    won: np.ndarray            # (n_games,) bool
    wrong: np.ndarray          # (n_games,) int32, wrong guesses at termination
    guess_strings: list[str]   # chronological guesses, ready for the CSV
    states: StateBuffer | None

    @property
    def win_rate(self) -> float:
        return float(self.won.mean()) * 100.0

    @property
    def total_wrong(self) -> int:
        return int(self.wrong.sum())

    def summary(self) -> str:
        return (
            f"win_rate={self.win_rate:.3f}%  "
            f"avg_wrong={self.wrong.mean():.3f}  "
            f"total_wrong={self.total_wrong}"
        )


def _to_bitmask(flags: torch.Tensor) -> torch.Tensor:
    """``(B, 26)`` bool -> ``(B,)`` int32 bitmask."""
    weights = (2 ** torch.arange(N_LETTERS, device=flags.device)).to(torch.int32)
    return (flags.to(torch.int32) * weights).sum(dim=1)


def _contains_matrix(truth: torch.Tensor) -> torch.Tensor:
    """``(B, L)`` letter ids -> ``(B, 26)`` bool set-membership matrix."""
    batch, _ = truth.shape
    out = torch.zeros((batch, N_LETTERS + 2), dtype=torch.bool, device=truth.device)
    out.scatter_(1, truth, True)
    return out[:, :N_LETTERS].contiguous()


@torch.no_grad()
def play_games(
    words: list[str],
    policy: Policy,
    *,
    device: torch.device | str = "cuda",
    max_wrong: int = MAX_WRONG_GUESSES,
    record_states: bool = False,
    explore_eps: float = 0.0,
    explore_top_k: int = 4,
    collect_guess_strings: bool = True,
    word_ids: np.ndarray | None = None,
) -> PlayResult:
    """Play one Hangman game per word, all games advancing in lockstep.

    Parameters
    ----------
    policy:
        Callable ``(board, guessed) -> logits``. ``board`` is
        ``(B, MAX_WORD_LEN)`` int64 holding :data:`MASK_TOKEN` at hidden
        positions and :data:`PAD_TOKEN` past the word length; ``guessed`` is
        ``(B, 26)`` bool. Already-guessed letters are masked out by the engine,
        so a policy never has to defend against repeat guesses itself.
    explore_eps:
        Probability of replacing the greedy action with a sample from the
        top-``explore_top_k`` candidates. Used *only* when generating training
        states, so that the buffer also covers states an imperfect policy
        reaches. Always zero during evaluation and submission.
    """
    device = torch.device(device)
    n_games = len(words)
    if n_games == 0:
        return PlayResult(np.zeros(0, bool), np.zeros(0, np.int32), [], StateBuffer.empty())

    if word_ids is None:
        word_ids = np.arange(n_games, dtype=np.int32)
    word_id_tensor = torch.from_numpy(np.ascontiguousarray(word_ids, dtype=np.int32)).to(device)

    truth = torch.from_numpy(encode_words(words)).to(device).long()
    is_real = truth != PAD_TOKEN
    board = torch.where(is_real, torch.full_like(truth, MASK_TOKEN), truth)

    contains = _contains_matrix(truth)
    contains_mask = _to_bitmask(contains)

    guessed = torch.zeros((n_games, N_LETTERS), dtype=torch.bool, device=device)
    wrong = torch.zeros(n_games, dtype=torch.int32, device=device)
    won = torch.zeros(n_games, dtype=torch.bool, device=device)
    active = torch.ones(n_games, dtype=torch.bool, device=device)

    # A word with no maskable characters would already be complete.
    solved_at_start = ~(board == MASK_TOKEN).any(dim=1)
    won |= solved_at_start
    active &= ~solved_at_start

    guess_log = torch.full((n_games, N_LETTERS), -1, dtype=torch.int8, device=device)
    guess_count = torch.zeros(n_games, dtype=torch.long, device=device)

    rec_board: list[np.ndarray] = []
    rec_guessed: list[np.ndarray] = []
    rec_contains: list[np.ndarray] = []
    rec_word_id: list[np.ndarray] = []

    for _ in range(N_LETTERS):
        idx = active.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break

        sub_board = board.index_select(0, idx)
        sub_guessed = guessed.index_select(0, idx)

        if record_states:
            rec_board.append(sub_board.to(torch.uint8).cpu().numpy())
            rec_guessed.append(_to_bitmask(sub_guessed).cpu().numpy())
            rec_contains.append(contains_mask.index_select(0, idx).cpu().numpy())
            rec_word_id.append(word_id_tensor.index_select(0, idx).cpu().numpy())

        logits = policy(sub_board, sub_guessed).float()
        logits = logits.masked_fill(sub_guessed, float("-inf"))
        action = logits.argmax(dim=1)

        if explore_eps > 0.0:
            k = min(explore_top_k, N_LETTERS)
            top_val, top_idx = logits.topk(k, dim=1)
            sampled = torch.multinomial(torch.softmax(top_val, dim=1), 1).squeeze(1)
            explored = top_idx.gather(1, sampled.unsqueeze(1)).squeeze(1)
            use_explore = torch.rand(idx.numel(), device=device) < explore_eps
            action = torch.where(use_explore, explored, action)

        if collect_guess_strings:
            guess_log[idx, guess_count.index_select(0, idx)] = action.to(torch.int8)
            guess_count[idx] += 1

        # --- apply the guess -------------------------------------------------
        hit = contains[idx, action]
        guessed[idx, action] = True

        sub_truth = truth.index_select(0, idx)
        board[idx] = torch.where(sub_truth == action.unsqueeze(1), sub_truth, sub_board)
        wrong[idx] += (~hit).to(torch.int32)

        just_won = ~(board.index_select(0, idx) == MASK_TOKEN).any(dim=1)
        just_lost = wrong.index_select(0, idx) >= max_wrong

        won[idx] |= just_won
        active[idx] = ~(just_won | just_lost)

    guess_strings: list[str] = []
    if collect_guess_strings:
        log_np = guess_log.cpu().numpy()
        count_np = guess_count.cpu().numpy()
        for row in range(n_games):
            guess_strings.append(
                "".join(ID_TO_LETTER[int(c)] for c in log_np[row, : count_np[row]])
            )

    states = None
    if record_states:
        states = (
            StateBuffer(
                np.concatenate(rec_board),
                np.concatenate(rec_guessed),
                np.concatenate(rec_contains),
                np.concatenate(rec_word_id),
            )
            if rec_board
            else StateBuffer.empty()
        )

    return PlayResult(
        won=won.cpu().numpy(),
        wrong=wrong.cpu().numpy(),
        guess_strings=guess_strings,
        states=states,
    )


def score_guess_strings(
    words: list[str],
    guess_strings: list[str],
    max_wrong: int = MAX_WRONG_GUESSES,
) -> tuple[float, int]:
    """Re-score a finished submission with an independent scalar implementation.

    Deliberately a plain Python loop: it exists to cross-check the vectorised
    engine and to mirror the grader's literal reading of the rules (a repeated
    or unproductive character costs one wrong guess).
    """
    wins = 0
    total_wrong = 0
    for word, guesses in zip(words, guess_strings):
        # Non-letter slots are revealed from the start and are never guessed.
        letters = {c for c in word if c in LETTER_TO_ID}
        revealed: set[str] = set()
        wrong = 0
        solved = not letters
        for char in guesses:
            if solved:
                break
            if char in letters and char not in revealed:
                revealed.add(char)
                if revealed == letters:
                    solved = True
            else:
                wrong += 1
                if wrong >= max_wrong:
                    break
        wins += int(solved)
        total_wrong += wrong
    return wins / max(len(words), 1) * 100.0, total_wrong
