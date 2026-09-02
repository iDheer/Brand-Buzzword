"""Central configuration for the Hangman solver.

Every magic number used by the training / inference stack lives here so that a
run is fully described by a single, serialisable object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Game constants (fixed by the competition rules)
# --------------------------------------------------------------------------- #

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
N_LETTERS = len(ALPHABET)

#: Token ids used by the character encoder.
MASK_TOKEN = N_LETTERS          # 26 -> an unrevealed position ("_")
PAD_TOKEN = N_LETTERS + 1       # 27 -> padding beyond the word length
VOCAB_SIZE = N_LETTERS + 2      # 28

#: A game is lost the moment the 6th wrong guess is made.
MAX_WRONG_GUESSES = 6

#: Longest word present in the competition corpus (train and test both peak at 29).
MAX_WORD_LEN = 29

LETTER_TO_ID = {c: i for i, c in enumerate(ALPHABET)}
ID_TO_LETTER = {i: c for i, c in enumerate(ALPHABET)}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

@dataclass
class Paths:
    """Filesystem layout for *training*. Overridden inside the Kaggle notebook.

    Deliberately has no field for the test word list: nothing on the training
    path may read it, and the cleanest way to guarantee that is to give the
    training configuration no way to name it. The test list is opened only by
    the inference entry points (``hangman.predict`` / ``scripts/predict.py``).
    """

    root: Path = Path(".")
    train_words: Path = Path("train.txt")
    artifacts: Path = Path("artifacts")

    def ensure(self) -> "Paths":
        self.artifacts.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- #
# Model / training hyper-parameters
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    """Architecture of the masked-word Transformer encoder."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    #: Number of scalar game-state features appended to the global context.
    n_scalar_features: int = 6
    #: Blend the set-level head with the noisy-OR of the per-position head.
    use_position_head: bool = True


@dataclass
class TrainConfig:
    """Optimisation and DAgger schedule."""

    #: Controls weight initialisation, batch shuffling and self-play sampling.
    seed: int = 1234
    #: Controls the train/validation split ONLY. Must be identical across every
    #: model that will be ensembled or compared: varying it would let one model
    #: train on another's validation words and inflate every number downstream.
    split_seed: int = 1234
    device: str = "cuda"

    # Held-out words used to estimate the win rate. Disjoint from training words.
    n_val_words: int = 12_000

    # --- optimisation -----------------------------------------------------
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 500
    amp_dtype: str = "bfloat16"

    # --- DAgger rounds ----------------------------------------------------
    #: Round 0 bootstraps from a stochastic frequency policy; later rounds
    #: replay the game with the model itself (dataset aggregation).
    n_rounds: int = 4
    #: Training words simulated per round to build the state buffer.
    words_per_round: int = 200_000
    #: Optimisation epochs over the buffer within each round.
    epochs_per_round: int = 2
    #: Probability of taking an exploratory (non-greedy) action during self-play.
    explore_eps: float = 0.15
    #: Fraction of the buffer retained from previous rounds (DAgger aggregation).
    replay_fraction: float = 0.35
    #: Sampling weight applied to words of length 4-9 when building each round's
    #: self-play corpus. Those lengths carry ~84% of all losses (long words are
    #: already solved 95%+ of the time), so a uniform sample spends most of its
    #: budget on states whose outcome is not in doubt. 1.0 reproduces uniform
    #: sampling exactly; >1 concentrates the round on the region that decides
    #: the win rate. Sampling switches to with-replacement when this is not 1.0.
    short_boost: float = 1.0

    # --- inference --------------------------------------------------------
    #: Games advanced in lockstep per forward pass. Keep this modest: an
    #: oversized batch pushes activations past VRAM and the driver silently
    #: spills to host memory, which looks like 100% GPU utilisation at a
    #: fraction of the power draw and is ~10x slower.
    play_chunk_size: int = 4096
    eval_batch_size: int = 8192

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = "transformer_dagger"
