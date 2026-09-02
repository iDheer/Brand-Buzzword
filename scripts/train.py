"""Train the Hangman Transformer.

Usage
-----
    python scripts/train.py --name v1 --rounds 4 --words-per-round 200000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hangman.config import ExperimentConfig, ModelConfig, Paths, TrainConfig
from hangman.data import load_words, split_train_val
from hangman.train import Trainer, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="transformer_dagger")
    p.add_argument("--seed", type=int, default=1234,
                   help="init / shuffling seed; vary this for ensemble diversity")
    p.add_argument("--split-seed", type=int, default=1234,
                   help="train/val split seed; keep FIXED across all models")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--words-per-round", type=int, default=200_000)
    p.add_argument("--epochs-per-round", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--explore-eps", type=float, default=0.15)
    p.add_argument("--replay-fraction", type=float, default=0.35)
    p.add_argument("--n-val", type=int, default=12_000)
    p.add_argument("--train-file", default="train.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ExperimentConfig(
        name=args.name,
        paths=Paths(train_words=Path(args.train_file)).ensure(),
        model=ModelConfig(
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ),
        train=TrainConfig(
            seed=args.seed,
            split_seed=args.split_seed,
            device=device,
            n_val_words=args.n_val,
            batch_size=args.batch_size,
            lr=args.lr,
            n_rounds=args.rounds,
            words_per_round=args.words_per_round,
            epochs_per_round=args.epochs_per_round,
            explore_eps=args.explore_eps,
            replay_fraction=args.replay_fraction,
        ),
    )

    words = load_words(cfg.paths.train_words)
    train_words, val_words = split_train_val(
        words, cfg.train.n_val_words, cfg.train.split_seed
    )
    print(
        f"corpus={len(words):,}  train={len(train_words):,}  val={len(val_words):,}  "
        f"device={device}  run={cfg.name}"
    )

    t0 = time.time()
    trainer = Trainer(cfg, train_words, val_words)
    trainer.fit()

    win_rate, wrong = evaluate(trainer.model, val_words, device=device)
    print(f"\nfinal (last-epoch weights) val win_rate={win_rate:.3f}%  wrong={wrong:,}")
    print(f"checkpoint: {trainer.checkpoint_path}")
    print(f"elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
