"""Independent leakage audit.

Answers one question with evidence rather than assertion: could any part of
this submission have seen the answers it is being scored on?

The decisive checks are behavioural. A model that leaked the test words would
play near-perfectly -- it would rarely miss, would almost never lose, and its
first guess would be correct essentially always. A model that genuinely infers
letters from spelling structure misses constantly. We measure that.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hangman.config import MAX_WRONG_GUESSES
from hangman.data import load_words, split_train_val
from hangman.simulator import score_guess_strings

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> None:
    train_words = load_words(ROOT / "train.txt")
    test_words = load_words(ROOT / "test.txt")
    fitted, held_out = split_train_val(train_words, 12_000, 1234)

    print("\n1. CORPUS SEPARATION")
    train_set, test_set = set(train_words), set(test_words)
    check("test.txt shares no word with train.txt",
          len(train_set & test_set) == 0,
          f"overlap = {len(train_set & test_set)}")
    check("held-out words excluded from the fitted corpus",
          len(set(held_out) & set(fitted)) == 0,
          f"overlap = {len(set(held_out) & set(fitted))}")
    check("fitted + held-out reconstruct the corpus exactly",
          len(fitted) + len(held_out) == len(train_words)
          and set(fitted) | set(held_out) == train_set)

    print("\n2. SOURCE AUDIT -- is test.txt referenced anywhere in the training path?")
    training_modules = ["config", "data", "simulator", "baselines", "model",
                        "dataset", "retrieval", "train"]
    offenders = []
    for name in training_modules:
        text = (ROOT / "hangman" / f"{name}.py").read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if re.search(r"test\.txt|test_words|TEST_FILE", line) and not line.strip().startswith("#"):
                offenders.append(f"{name}.py:{lineno}: {line.strip()}")
    check("no training module reads test.txt", not offenders,
          "; ".join(offenders) if offenders else "")

    predict_text = (ROOT / "hangman" / "predict.py").read_text(encoding="utf-8")
    check("inference module never fits anything",
          not re.search(r"\.backward\(|optimizer|\.train\(\)|Trainer", predict_text))

    print("\n3. BEHAVIOURAL PROOF -- does the submission actually play the game?")
    submission = ROOT / "submission.csv"
    if not submission.exists():
        print("  submission.csv missing; skipping")
        return

    with open(submission, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        guesses = [row[1] for row in reader]

    check("row count matches the test set",
          len(guesses) == len(test_words), f"{len(guesses):,} vs {len(test_words):,}")

    win_rate, total_wrong = score_guess_strings(test_words, guesses)

    # A leaked solution guesses only the letters that are in the word.
    first_correct = sum(1 for w, g in zip(test_words, guesses) if g and g[0] in w)
    perfect_games = sum(
        1 for w, g in zip(test_words, guesses)
        if g and all(c in w for c in g)
    )
    wrong_per_word = total_wrong / len(test_words)

    print(f"      win rate            {win_rate:.4f}%")
    print(f"      total wrong guesses {total_wrong:,}  ({wrong_per_word:.2f} per word)")
    print(f"      first guess correct {first_correct / len(guesses) * 100:.1f}%")
    print(f"      games with zero misses {perfect_games / len(guesses) * 100:.1f}%")

    check("the model misses often (a lookup would not)",
          wrong_per_word > 1.0, f"{wrong_per_word:.2f} wrong guesses per word")
    check("first guess is frequently wrong (a lookup would always be right)",
          first_correct / len(guesses) < 0.95,
          f"{first_correct / len(guesses) * 100:.1f}% correct")
    check("a substantial share of games are lost",
          win_rate < 95.0, f"{100 - win_rate:.1f}% of words unsolved")

    print("\n4. STRATEGY IS WORD-INDEPENDENT AT THE START")
    # Without leakage the opening guess can only depend on word length, so a
    # handful of distinct openings must cover the entire test set.
    openers = Counter(g[0] for g in guesses if g)
    by_length: dict[int, Counter] = {}
    for word, g in zip(test_words, guesses):
        if g:
            by_length.setdefault(len(word), Counter())[g[0]] += 1
    ambiguous = [ln for ln, c in by_length.items() if len(c) > 1]
    print(f"      distinct opening letters overall: {len(openers)}  {dict(openers.most_common(6))}")
    check("one fixed opening letter per word length",
          not ambiguous,
          f"lengths with more than one opener: {ambiguous}" if ambiguous else
          f"{len(by_length)} lengths, each with a single opener")

    print("\n5. RULE CONFORMANCE OF THE WRITTEN STRINGS")
    check("no repeated guess in any row",
          all(len(set(g)) == len(g) for g in guesses))
    check("all characters are lowercase a-z",
          all(g.isalpha() and g.islower() for g in guesses if g))
    # The scored prefix is what the grader reads: everything up to the guess
    # that completes the word or lands the 6th miss. Trailing characters are
    # explicitly "locked out and ignored", so they are checked separately.
    def scored_prefix(word: str, g: str) -> str:
        letters = {c for c in word if c.isalpha()}
        seen: set[str] = set()
        wrong = 0
        for i, c in enumerate(g):
            if c in letters and c not in seen:
                seen.add(c)
                if seen == letters:
                    return g[: i + 1]
            else:
                wrong += 1
                if wrong >= MAX_WRONG_GUESSES:
                    return g[: i + 1]
        return g

    prefixes = [scored_prefix(w, g) for w, g in zip(test_words, guesses)]
    over = [i for i, (w, g) in enumerate(zip(test_words, prefixes))
            if sum(1 for c in g if c not in w) > MAX_WRONG_GUESSES]
    check("no scored prefix spends more than 6 wrong guesses", not over,
          f"{len(over)} offending rows" if over else "")
    tail = sum(len(g) - len(p) for g, p in zip(guesses, prefixes))
    print(f"      trailing (ignored) characters: {tail:,} "
          f"= {tail / sum(map(len, guesses)) * 100:.1f}% of all characters")

    print("\n" + ("AUDIT PASSED -- no evidence of leakage"
                  if not FAILURES else f"AUDIT FAILURES: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
