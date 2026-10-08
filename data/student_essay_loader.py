"""Load student essays from ``data/raw/student_essays/`` for V2 pipelines.

Layout::

    student_essays/
        <author_id>/          # one folder per student
            essay1.txt        # one file per essay
            topic_2.txt
            ...

Each subfolder name becomes ``author_id``. Topic strings for ``TextPair`` are
inferred from the filename stem (same rule as ``load_student_essays`` in
``preprocessor.py``): ``history_1.txt`` → topic ``history``.

Pair generation uses ``create_pairs_from_author_dict``: all within-author
pairs (same-author) and one random cross-author pair per student pair
(different-author), then 50/50 balancing by under-sampling the majority class.
"""

from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from data.preprocessor import TextPair, clean_text, create_pairs_from_author_dict

DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "raw" / "student_essays"
SOURCE_DATASET = "StudentEssays"

_TEXT_EXTENSIONS = {".txt", ".md"}


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_EXTENSIONS and path.is_file()


def _topic_from_filename(stem: str) -> str:
    """Match ``preprocessor.load_student_essays`` topic inference."""
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def load_student_essay_pairs(
    base_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
    *,
    print_summary: bool = True,
) -> List[TextPair]:
    """Load all essays and return balanced ``TextPair`` objects (Reuters-compatible).

    Parameters
    ----------
    base_dir
        Root directory containing one subfolder per author. Defaults to
        ``data/raw/student_essays`` next to this package.
    seed
        Random seed for different-author sampling and balancing shuffle.
    print_summary
        If True, print author/essay counts and raw vs balanced pair counts.

    Returns
    -------
    List[TextPair]
        Same contract as ``load_reuters()`` output.

    Raises
    ------
    FileNotFoundError
        If ``base_dir`` does not exist.
    ValueError
        If no essays were loaded, or fewer than two authors (no pairs possible).
    """
    root = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Student essay corpus not found: {root}\n"
            "Create the directory and add one subfolder per author with .txt/.md essays."
        )

    texts_by_author: Dict[str, List[str]] = {}
    topics_by_author: Dict[str, List[str]] = {}

    for author in sorted(os.listdir(root)):
        author_dir = root / author
        if not author_dir.is_dir() or author.startswith("."):
            continue
        for fname in sorted(os.listdir(author_dir)):
            fpath = author_dir / fname
            if not _is_text_file(fpath):
                continue
            stem = fpath.stem
            topic = _topic_from_filename(stem)
            raw = fpath.read_text(encoding="utf-8", errors="replace")
            texts_by_author.setdefault(author, []).append(clean_text(raw))
            topics_by_author.setdefault(author, []).append(topic)

    n_authors = len(texts_by_author)
    if n_authors == 0:
        raise ValueError(
            f"No author folders with readable .txt/.md essays under {root}."
        )

    counts = {a: len(texts_by_author[a]) for a in texts_by_author}
    total_essays = sum(counts.values())
    if total_essays == 0:
        raise ValueError(f"No essay files found under {root}.")

    # Raw pair counts before 50/50 balancing (same logic as create_pairs_from_author_dict)
    n_same_raw = sum(
        math.comb(n, 2) if n >= 2 else 0 for n in counts.values()
    )
    n_diff_raw = math.comb(n_authors, 2) if n_authors >= 2 else 0

    if print_summary:
        print("─" * 60)
        print(f"STUDENT ESSAYS  ({SOURCE_DATASET})")
        print(f"  Root          : {root.resolve()}")
        print(f"  Authors       : {n_authors}")
        print(f"  Total essays  : {total_essays}")
        print("  Essays / author:")
        for aid in sorted(counts.keys()):
            print(f"    {aid}: {counts[aid]}")
        print(f"  Raw same-author pairs      : {n_same_raw}  (all within-author pairs)")
        print(
            f"  Raw different-author pairs : {n_diff_raw}  "
            f"(one sampled pair per author-pair)"
        )
        if n_same_raw == 0 and n_authors >= 2:
            print(
                "  NOTE: No same-author pairs — each author needs at least 2 essays."
            )
        print("─" * 60)

    if n_authors < 2:
        raise ValueError(
            "Need at least two author folders to build different-author pairs."
        )

    pairs = create_pairs_from_author_dict(
        texts_by_author,
        SOURCE_DATASET,
        topics_by_author=topics_by_author,
        seed=seed,
    )

    if print_summary:
        n_same_bal = sum(p.label for p in pairs)
        n_diff_bal = len(pairs) - n_same_bal
        print(
            f"  Balanced dataset           : {len(pairs)} pairs "
            f"({n_same_bal} same-author, {n_diff_bal} different-author)"
        )
        print("─" * 60)

    if not pairs:
        raise ValueError(
            "Balanced pair list is empty. Usually this means n_same_raw == 0 "
            "while different-author pairs exist — add 2+ essays per student."
        )

    return pairs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Load student essay pairs and print summary.")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override corpus root (default: data/raw/student_essays)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    load_student_essay_pairs(base_dir=args.base_dir, seed=args.seed, print_summary=True)
