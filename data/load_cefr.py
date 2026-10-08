"""CEFR vocabulary wordlist loader with module-level caching.

Canonical file location
-----------------------
    data/raw/ENGLISH_CEFR_WORDS.csv          (configured via config.CEFR_WORDLIST_PATH)

The CSV must contain at minimum two columns:

    headword   — the vocabulary item in its dictionary (head) form
    CEFR       — the CEFR level: one of A1, A2, B1, B2, C1, C2

Compatible sources
------------------
- English Profile Word Lists  (https://www.englishprofile.org/wordlists)
- The Oxford 5000 CEFR list
- Any custom wordlist in the same CSV format

Usage
-----
    # Typical pipeline usage (config-aware, cached):
    from data.load_cefr import get_cefr_dict
    cefr = get_cefr_dict()
    features = extract_all(text, cefr_dict=cefr)

    # Test / explicit-path usage:
    from data.load_cefr import load_cefr_dict
    cefr = load_cefr_dict("path/to/my_wordlist.csv")
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd


# Module-level cache — populated on first call to get_cefr_dict()
_cached_dict: Optional[Dict[str, str]] = None


def load_cefr_dict(path: str) -> Dict[str, str]:
    """Load a CEFR vocabulary CSV at the given path.

    Parameters
    ----------
    path : Absolute or relative path to the CSV file.

    Returns
    -------
    ``{word_lower -> level}`` where level is one of A1, A2, B1, B2, C1, C2.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the CSV is missing the required ``headword`` or ``CEFR`` columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CEFR wordlist not found: {path}\n"
            "Place your CEFR CSV at that path or update CEFR_WORDLIST_PATH in .env.\n"
            "The CSV must contain columns: headword, CEFR"
        )

    df = pd.read_csv(path)

    # Tolerate minor column name variations (e.g. 'Word' instead of 'headword')
    col_map = {c.lower(): c for c in df.columns}
    headword_col = col_map.get("headword", col_map.get("word", col_map.get("headword")))
    cefr_col = col_map.get("cefr", col_map.get("level"))

    if headword_col is None or cefr_col is None:
        raise KeyError(
            f"Could not find required columns in {path}.\n"
            f"Found columns: {list(df.columns)}\n"
            "Expected: 'headword' (or 'word') and 'CEFR' (or 'level')."
        )

    result: Dict[str, str] = {}
    for word, level in zip(df[headword_col], df[cefr_col]):
        word_lower = str(word).strip().lower()
        level_str = str(level).strip()
        result[word_lower] = level_str

    return result


def get_cefr_dict() -> Dict[str, str]:
    """Return the CEFR dict for the configured wordlist path.

    Loads and caches on first call; subsequent calls return the cached copy.
    Uses ``config.CEFR_WORDLIST_PATH``.

    Raises
    ------
    FileNotFoundError
        If the configured wordlist file is absent.  See ``config.py`` and
        ``.env.example`` for how to set the path.
    """
    global _cached_dict
    if _cached_dict is None:
        from config import CEFR_WORDLIST_PATH
        _cached_dict = load_cefr_dict(CEFR_WORDLIST_PATH)
        print(f"CEFR dict loaded: {len(_cached_dict)} entries from {CEFR_WORDLIST_PATH}")
    return _cached_dict


def clear_cache() -> None:
    """Reset the module-level cache.  Useful in tests that swap wordlist files."""
    global _cached_dict
    _cached_dict = None
