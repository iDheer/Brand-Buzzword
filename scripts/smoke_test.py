"""Validate the vectorised engine against the independent scalar scorer."""
import sys, time
sys.path.insert(0, ".")
import torch
from hangman.data import load_words, split_train_val
from hangman.simulator import play_games, score_guess_strings
from hangman.baselines import (
    LengthConditionedFrequencyPolicy,
    PositionalNGramPolicy,
    UniformRandomPolicy,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

words = load_words("train.txt")
train, val = split_train_val(words, n_val=12000, seed=1234)
print(f"train={len(train)}  val={len(val)}  device={DEV}")

for name, policy in [
    ("uniform-random     ", UniformRandomPolicy()),
    ("length-frequency   ", LengthConditionedFrequencyPolicy(train, DEV)),
    ("positional-ngram   ", PositionalNGramPolicy(train, DEV)),
]:
    t0 = time.time()
    res = play_games(val, policy, device=DEV)
    wr, tw = score_guess_strings(val, res.guess_strings)
    agree = abs(wr - res.win_rate) < 1e-9 and tw == res.total_wrong
    print(f"{name} {res.summary()}  scalar_check={'PASS' if agree else f'MISMATCH {wr:.3f}/{tw}'}  ({time.time()-t0:.1f}s)")
