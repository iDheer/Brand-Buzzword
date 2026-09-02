"""Compare policies on held-out words and tune the retrieval blend weight.

Everything here is measured on words carved out of ``train.txt`` and never used
for fitting, so the numbers are honest generalisation estimates. ``test.txt`` is
not touched.

Usage
-----
    python scripts/evaluate.py --checkpoints artifacts/v1_d256.pt
    python scripts/evaluate.py --checkpoints artifacts/a.pt artifacts/b.pt --alphas 0 0.3 0.5 0.7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hangman.baselines import LengthConditionedFrequencyPolicy, PositionalNGramPolicy
from hangman.data import load_words, split_train_val
from hangman.predict import EnsemblePolicy
from hangman.retrieval import HybridPolicy, LexiconRetriever, RetrievalPolicy
from hangman.simulator import play_games
from hangman.train import load_model


def run(words: list[str], policy, device: str, batch_size: int = 8192) -> tuple[float, int, float]:
    t0 = time.time()
    wins, wrong = 0, 0
    for start in range(0, len(words), batch_size):
        chunk = words[start : start + batch_size]
        result = play_games(chunk, policy, device=device, collect_guess_strings=False)
        wins += int(result.won.sum())
        wrong += result.total_wrong
    return wins / len(words) * 100.0, wrong, time.time() - t0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--train-file", default="train.txt")
    p.add_argument("--n-val", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--alphas", nargs="*", type=float,
                   default=[0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--skip-baselines", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus = load_words(args.train_file)
    train_words, val_words = split_train_val(corpus, args.n_val, args.seed)
    print(f"fit corpus {len(train_words):,}   held-out {len(val_words):,}   device {device}\n")

    rows: list[tuple[str, float, int, float]] = []

    if not args.skip_baselines:
        for name, policy in [
            ("length-frequency", LengthConditionedFrequencyPolicy(train_words, device)),
            ("positional-ngram", PositionalNGramPolicy(train_words, device)),
        ]:
            rows.append((name, *run(val_words, policy, device, args.batch_size)))

    models = [load_model(path, device) for path in args.checkpoints]
    for path, model in zip(args.checkpoints, models):
        name = f"neural: {Path(path).stem}"
        rows.append((name, *run(val_words, model.as_policy(), device, args.batch_size)))

    ensemble = EnsemblePolicy(models)
    if len(models) > 1:
        rows.append(("neural ensemble", *run(val_words, ensemble, device, args.batch_size)))

    # The retriever indexes only the words the model was fitted on.
    retriever = LexiconRetriever(train_words, device)
    print(f"retrieval index: {sum(retriever.bucket_sizes.values()):,} words, "
          f"{retriever.memory_mb():.0f} MB\n")

    rows.append((
        "retrieval only",
        *run(val_words, RetrievalPolicy(retriever,
                                        fallback=PositionalNGramPolicy(train_words, device)),
             device, args.batch_size),
    ))

    best = (None, -1.0)
    for alpha in args.alphas:
        if alpha == 0.0:
            continue
        policy = HybridPolicy(ensemble, retriever, alpha=alpha)
        win, wrong, secs = run(val_words, policy, device, args.batch_size)
        rows.append((f"hybrid alpha={alpha:g}", win, wrong, secs))
        if win > best[1]:
            best = (alpha, win)

    width = max(len(r[0]) for r in rows) + 2
    print(f"{'policy':<{width}}{'win rate':>10}{'wrong':>12}{'seconds':>10}")
    print("-" * (width + 32))
    for name, win, wrong, secs in rows:
        print(f"{name:<{width}}{win:>9.3f}%{wrong:>12,}{secs:>10.1f}")

    if best[0] is not None:
        print(f"\nbest retrieval alpha = {best[0]:g}  ({best[1]:.3f}% on held-out words)")


if __name__ == "__main__":
    main()
