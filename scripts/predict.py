"""Generate submission.csv from one or more trained checkpoints.

Usage
-----
    python scripts/predict.py --checkpoints artifacts/v1_d256.pt
    python scripts/predict.py --checkpoints artifacts/a.pt artifacts/b.pt --out submission.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hangman.data import load_words, split_train_val
from hangman.predict import (
    EnsemblePolicy,
    build_guess_strings,
    validate_submission,
    write_submission,
)
from hangman.simulator import score_guess_strings
from hangman.train import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--test-file", default="test.txt")
    p.add_argument("--train-file", default="train.txt")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--batch-size", type=int, default=16_384)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-val", type=int, default=12_000)
    p.add_argument("--skip-holdout", action="store_true")
    p.add_argument("--play-max-wrong", type=int, default=6,
                   help="how far to keep recording guesses; scoring is always "
                        "at 6. Use 26 to append the full fallback ordering.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = [load_model(path, device) for path in args.checkpoints]
    for path, model in zip(args.checkpoints, models):
        print(f"loaded {path}  ({model.n_parameters():,} params)")
    policy = EnsemblePolicy(models, args.weights)

    # Honest generalisation estimate: words held out of training entirely.
    if not args.skip_holdout:
        corpus = load_words(args.train_file)
        _, val_words = split_train_val(corpus, args.n_val, args.seed)
        _, val_win, val_wrong = build_guess_strings(
            val_words, policy, device=device, batch_size=args.batch_size, verbose=False
        )
        print(f"held-out (unseen train words): win_rate={val_win:.3f}%  wrong={val_wrong:,}")

    test_words = load_words(args.test_file)
    print(f"\nplaying {len(test_words):,} public test words...")
    t0 = time.time()
    guess_strings, win_rate, wrong = build_guess_strings(
        test_words, policy, device=device, batch_size=args.batch_size,
        play_max_wrong=args.play_max_wrong,
    )
    print(f"public test: win_rate={win_rate:.4f}%  total_wrong={wrong:,}  "
          f"({time.time() - t0:.0f}s)")
    print(f"projected leaderboard score = {win_rate:.4f} - {wrong}/1e8 = "
          f"{win_rate - wrong / 1e8:.6f}")

    # Independent scalar re-score of the exact strings we are about to submit.
    check_win, check_wrong = score_guess_strings(test_words, guess_strings)
    assert abs(check_win - win_rate) < 1e-9 and check_wrong == wrong, "scorer mismatch"
    print("independent re-score of the written strings: PASS")

    path = write_submission(guess_strings, args.out)
    validate_submission(path, expected_rows=len(test_words))


if __name__ == "__main__":
    main()
