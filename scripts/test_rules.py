"""Rule-level tests for the game engine.

The whole submission is only as correct as this engine, so the rules are
pinned down explicitly rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hangman.config import ID_TO_LETTER, LETTER_TO_ID, MASK_TOKEN, N_LETTERS, PAD_TOKEN
from hangman.simulator import play_games, score_guess_strings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def fixed_order_policy(order: str):
    """A policy that always prefers ``order``, left to right."""
    scores = torch.full((N_LETTERS,), -1.0, device=DEVICE)
    for rank, char in enumerate(order):
        scores[LETTER_TO_ID[char]] = float(len(order) - rank)

    def policy(board: torch.Tensor, guessed: torch.Tensor) -> torch.Tensor:
        return scores.unsqueeze(0).expand(board.shape[0], -1).clone()

    return policy


def main() -> None:
    print("engine rule tests")

    # 1. A correct guess reveals every occurrence at once, so one guess per
    #    distinct letter is enough to win with zero wrong guesses.
    word = "banana"
    res = play_games([word], fixed_order_policy("ban"), device=DEVICE)
    check("all occurrences revealed by one guess",
          bool(res.won[0]) and res.wrong[0] == 0 and res.guess_strings[0] == "ban",
          f"won={res.won[0]} wrong={res.wrong[0]} guesses={res.guess_strings[0]!r}")

    # 2. Six wrong guesses ends the game as a loss.
    res = play_games(["zzz"], fixed_order_policy("abcdefz"), device=DEVICE)
    check("6th wrong guess terminates as a loss",
          not res.won[0] and res.wrong[0] == 6 and res.guess_strings[0] == "abcdef",
          f"won={res.won[0]} wrong={res.wrong[0]} guesses={res.guess_strings[0]!r}")

    # 3. Five wrong guesses are survivable: the game continues.
    res = play_games(["zz"], fixed_order_policy("abcdez"), device=DEVICE)
    check("5 wrong guesses still allow a win",
          bool(res.won[0]) and res.wrong[0] == 5 and res.guess_strings[0] == "abcdez",
          f"won={res.won[0]} wrong={res.wrong[0]} guesses={res.guess_strings[0]!r}")

    # 4. The engine never emits a repeated guess (repeats would cost a life).
    words = ["abcdefghij", "zyxwvu", "aaa", "qwertyuiop"]
    res = play_games(words, fixed_order_policy("etaoinshrdlucmfwypvbgkjqxz"), device=DEVICE)
    check("no repeated guesses emitted",
          all(len(set(g)) == len(g) for g in res.guess_strings))

    # 5. Play stops the moment the word is complete: nothing trailing.
    res = play_games(["aa"], fixed_order_policy("abcdef"), device=DEVICE)
    check("stops immediately on completion",
          res.guess_strings[0] == "a" and res.wrong[0] == 0,
          f"guesses={res.guess_strings[0]!r}")

    # 6. Vectorised engine and the independent scalar scorer must agree.
    rng = np.random.default_rng(7)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    random_words = [
        "".join(rng.choice(list(alphabet), size=int(rng.integers(2, 16))))
        for _ in range(4000)
    ]

    def noisy(board: torch.Tensor, guessed: torch.Tensor) -> torch.Tensor:
        return torch.rand((board.shape[0], N_LETTERS), device=board.device)

    torch.manual_seed(3)
    res = play_games(random_words, noisy, device=DEVICE)
    scalar_win, scalar_wrong = score_guess_strings(random_words, res.guess_strings)
    check("vectorised engine == scalar scorer",
          abs(scalar_win - res.win_rate) < 1e-9 and scalar_wrong == res.total_wrong,
          f"{scalar_win:.4f}/{scalar_wrong} vs {res.win_rate:.4f}/{res.total_wrong}")

    # 7. Recorded states are consistent: the board never shows a letter that is
    #    not marked as guessed, and never hides one that is.
    res = play_games(random_words[:500], noisy, device=DEVICE, record_states=True)
    states = res.states
    bits = np.arange(N_LETTERS)
    guessed_bits = ((states.guessed[:, None] >> bits) & 1).astype(bool)
    board = states.board
    visible = np.zeros_like(guessed_bits)
    for letter in range(N_LETTERS):
        visible[:, letter] = (board == letter).any(axis=1)
    check("every visible letter was guessed", bool((visible & ~guessed_bits).sum() == 0))

    contains_bits = ((states.contains[:, None] >> bits) & 1).astype(bool)
    check("visible letters == guessed AND contained",
          bool((visible != (guessed_bits & contains_bits)).sum() == 0))

    # 8. Padding is never mistaken for a hidden slot.
    res = play_games(["ab", "abcdefghijklm"], fixed_order_policy("abcdefghijklm"), device=DEVICE)
    check("short words are not padded into the board",
          res.guess_strings[0] == "ab" and bool(res.won[0]))

    print("\n" + ("ALL RULE TESTS PASS" if not FAILURES else f"FAILURES: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
