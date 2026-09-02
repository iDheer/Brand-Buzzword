"""DAgger training loop.

Why DAgger and not random masking
---------------------------------
The obvious way to build a training set is to take a word, reveal a random
subset of its letters, and ask the model to name a hidden one. That is easy and
badly wrong: the states it produces are not the states a *playing* model
encounters. Real boards are reached by a policy that guesses common letters
first, and they always come with a set of letters that have been ruled out by
failed guesses -- information random masking cannot express at all.

So instead we generate states by *playing*. Round 0 bootstraps with a
statistical policy; every later round replays the corpus with the current
network (plus a little exploration) and aggregates the new states into the
buffer. This is Dataset Aggregation (Ross et al., 2011): it drives the training
distribution towards the model's own state distribution, which is precisely the
distribution the leaderboard measures.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, ModelConfig
from .data import encode_words
from .dataset import GpuStateStore
from .baselines import PositionalNGramPolicy
from .model import HangmanTransformer, hangman_loss
from .simulator import Policy, StateBuffer, play_games


# --------------------------------------------------------------------------- #
# State generation
# --------------------------------------------------------------------------- #

def generate_states(
    words: list[str],
    policy: Policy,
    *,
    device: str = "cuda",
    explore_eps: float = 0.15,
    explore_top_k: int = 4,
    chunk_size: int = 4_096,
) -> tuple[StateBuffer, float]:
    """Play the corpus and collect every decision point encountered.

    Returns the aggregated buffer and the greedy-equivalent win rate observed
    during generation (informative, though depressed by the exploration noise).
    """
    buffers: list[StateBuffer] = []
    wins = 0
    for start in range(0, len(words), chunk_size):
        chunk = words[start : start + chunk_size]
        ids = np.arange(start, start + len(chunk), dtype=np.int32)
        result = play_games(
            chunk,
            policy,
            device=device,
            record_states=True,
            explore_eps=explore_eps,
            explore_top_k=explore_top_k,
            collect_guess_strings=False,
            word_ids=ids,
        )
        buffers.append(result.states)
        wins += int(result.won.sum())
    return StateBuffer.concat(buffers), wins / max(len(words), 1) * 100.0


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    model: HangmanTransformer,
    words: list[str],
    *,
    device: str = "cuda",
    batch_size: int = 16_384,
) -> tuple[float, int]:
    """Greedy win rate and total wrong guesses on a held-out word list."""
    model.eval()
    policy = model.as_policy()
    wins, wrong = 0, 0
    for start in range(0, len(words), batch_size):
        chunk = words[start : start + batch_size]
        result = play_games(chunk, policy, device=device, collect_guess_strings=False)
        wins += int(result.won.sum())
        wrong += result.total_wrong
    return wins / max(len(words), 1) * 100.0, wrong


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class Trainer:
    def __init__(self, cfg: ExperimentConfig, train_words: list[str], val_words: list[str]):
        self.cfg = cfg
        self.train_words = train_words
        self.val_words = val_words
        self.device = torch.device(cfg.train.device)

        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        self.rng = np.random.default_rng(cfg.train.seed)

        self.model = HangmanTransformer(cfg.model).to(self.device)
        self.encoded_corpus = encode_words(train_words)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.98),
        )
        self.amp_dtype = getattr(torch, cfg.train.amp_dtype)
        self.global_step = 0
        self.total_steps = 1  # refined once the first buffer size is known
        self.history: list[dict] = []
        self.best_win_rate = -1.0

        cfg.paths.ensure()
        self.checkpoint_path = cfg.paths.artifacts / f"{cfg.name}.pt"
        self.history_path = cfg.paths.artifacts / f"{cfg.name}_history.json"

    # -- schedule ---------------------------------------------------------- #

    def _lr_at(self, step: int) -> float:
        warmup = self.cfg.train.warmup_steps
        base = self.cfg.train.lr
        if step < warmup:
            return base * (step + 1) / warmup
        progress = (step - warmup) / max(self.total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.05 * base + 0.95 * base * 0.5 * (1.0 + math.cos(math.pi * progress))

    # -- one optimisation epoch -------------------------------------------- #

    def _train_epoch(self, store: GpuStateStore) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in store.epoch_batches(self.cfg.train.batch_size):
            lr = self._lr_at(self.global_step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
                outputs = self.model(batch["board"], batch["guessed"])
                loss, stats = hangman_loss(
                    outputs, batch["target"], batch["guessed"], batch["truth"]
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.optimizer.step()

            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1
            self.global_step += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    # -- DAgger loop -------------------------------------------------------- #

    def fit(self) -> HangmanTransformer:
        cfg = self.cfg.train
        print(f"model parameters: {self.model.n_parameters():,}")

        bootstrap = PositionalNGramPolicy(self.train_words, self.device)
        aggregated: StateBuffer | None = None

        for round_idx in range(cfg.n_rounds):
            t0 = time.time()

            if round_idx == 0:
                policy: Policy = bootstrap
                eps, top_k = 0.35, 6
                label = "bootstrap(positional-ngram)"
            else:
                policy = self.model.as_policy()
                eps, top_k = cfg.explore_eps, 4
                label = "self-play"

            sample_n = min(cfg.words_per_round, len(self.train_words))
            sample_idx = self.rng.choice(len(self.train_words), size=sample_n, replace=False)
            sample_idx.sort()
            sample_words = [self.train_words[i] for i in sample_idx]

            fresh, gen_win = generate_states(
                sample_words, policy, device=str(self.device),
                explore_eps=eps, explore_top_k=top_k,
                chunk_size=cfg.play_chunk_size,
            )
            # word_id from generate_states indexes the *sample*; remap to corpus.
            fresh.word_id = sample_idx[fresh.word_id].astype(np.int32)

            if aggregated is None or cfg.replay_fraction <= 0.0:
                buffer = fresh
            else:
                keep = int(len(aggregated) * cfg.replay_fraction)
                buffer = StateBuffer.concat(
                    [fresh, aggregated.subsample(keep, self.rng)]
                )
            aggregated = buffer

            store = GpuStateStore(buffer, self.encoded_corpus, self.device)
            steps_this_round = math.ceil(len(store) / cfg.batch_size) * cfg.epochs_per_round
            remaining_rounds = cfg.n_rounds - round_idx
            self.total_steps = self.global_step + steps_this_round * remaining_rounds

            gen_time = time.time() - t0
            print(
                f"\n[round {round_idx}] policy={label}  states={len(store):,}  "
                f"gen_win_rate={gen_win:.2f}%  ({gen_time:.0f}s)",
                flush=True,
            )

            for epoch in range(cfg.epochs_per_round):
                stats = self._train_epoch(store)
                win_rate, wrong = evaluate(
                    self.model, self.val_words, device=str(self.device),
                    batch_size=cfg.eval_batch_size,
                )
                record = {
                    "round": round_idx, "epoch": epoch, "step": self.global_step,
                    "val_win_rate": win_rate, "val_total_wrong": wrong, **stats,
                }
                self.history.append(record)
                print(
                    f"  round {round_idx} epoch {epoch}  loss={stats['loss']:.4f} "
                    f"(fused {stats['fused']:.4f} / pos {stats['position']:.4f})  "
                    f"VAL win_rate={win_rate:.3f}%  wrong={wrong:,}",
                    flush=True,
                )
                improved = win_rate > self.best_win_rate
                if improved:
                    self.best_win_rate = win_rate
                self.save(checkpoint=improved)

            del store
            torch.cuda.empty_cache()

        print(f"\nbest validation win rate: {self.best_win_rate:.3f}%")
        return self.model

    # -- persistence -------------------------------------------------------- #

    def save(self, checkpoint: bool = True) -> None:
        """Persist history always; the checkpoint only when it improved."""
        if checkpoint:
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "model_config": asdict(self.cfg.model),
                    "val_win_rate": self.best_win_rate,
                    "step": self.global_step,
                },
                self.checkpoint_path,
            )
        self.history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")


def load_model(path: str | Path, device: str = "cuda") -> HangmanTransformer:
    """Rebuild a trained model from a checkpoint."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HangmanTransformer(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
