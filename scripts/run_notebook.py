"""Execute the generated Kaggle notebook end to end, locally.

Catches the failure mode that matters most: a notebook that looks right but
raises on Kaggle after you have spent a daily submission on it. The notebook's
code cells are concatenated and executed in one namespace, exactly as
*Save & Run All* would.

Usage
-----
    python scripts/run_notebook.py                      # full 250k test words
    python scripts/run_notebook.py --limit-test 20000   # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notebook", default="notebooks/hangman_submission.ipynb")
    p.add_argument("--weights-dir", default="artifacts")
    p.add_argument("--limit-test", type=int, default=0,
                   help="truncate the test list for a fast smoke run (0 = full)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    notebook = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    source = "\n\n".join("".join(c["source"]) for c in cells)

    # Kaggle-only affordances that do not exist in a plain interpreter.
    namespace: dict = {"display": lambda *a, **k: None, "__name__": "__main__"}

    # Point the notebook at this repository instead of /kaggle/input.
    source = source.replace(
        'WEIGHTS_DIR = Path("/kaggle/input/hangman-weights")',
        f'WEIGHTS_DIR = Path({args.weights_dir!r})',
    )
    if args.limit_test:
        source = source.replace(
            "test_words = load_words(TEST_FILE)",
            f"test_words = load_words(TEST_FILE)[:{args.limit_test}]",
        )
        source = source.replace(
            "validate_submission(submission_path, expected_rows=len(test_words))",
            "validate_submission(submission_path, expected_rows=len(test_words))",
        )

    print(f"executing {len(cells)} code cells from {args.notebook}")
    print("-" * 70)
    exec(compile(source, args.notebook, "exec"), namespace)
    print("-" * 70)
    print("NOTEBOOK RAN CLEAN")


if __name__ == "__main__":
    main()
