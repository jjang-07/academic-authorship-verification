"""FAISS-backed stylometric vector store.

Every indexed entry represents a *verified-author* writing sample.  Retrieval is
driven exclusively by stylometric feature vectors — not by raw text or topic
embeddings — so that the top-k results are stylistically similar rather than
topically similar.

Module layout
-------------
TextEntry          — the unit stored in the index (author_id, raw_text, features, …)
IndexBundle        — wraps a live FAISS index + aligned metadata + feature key order
feature_dict_to_vector  — Dict[str, float] → np.ndarray with a canonical key order
build_index        — extract vectors, build and optionally persist an IndexBundle
entries_from_pairs — convenience: turn List[TextPair] into List[TextEntry]
load_index         — load a previously saved IndexBundle from disk

Persistence layout (paths come from config)
-------------------------------------------
    data/vector_store/faiss.index          ← binary FAISS index
    data/vector_store/metadata.jsonl       ← one JSON object per line, TextEntry fields
    data/vector_store/metadata.keys.json   ← ordered list of feature key names

Distance metric
---------------
Vectors are L2-normalised before insertion, so FAISS IndexFlatL2 distances are
equivalent to (1 − cosine_similarity) × 2.  The retriever converts back to
cosine similarity for display.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config


# ──────────────────────────────── Data Structures ─────────────────────────────

@dataclass
class TextEntry:
    """One verified-author document stored in the vector index.

    Fields
    ------
    author_id      : Canonical author identifier (from the source corpus).
    raw_text       : Full text of the document.
    feature_vector : Output of ``features.handcrafted.extract_all()``.
                     Stored as a plain dict so it serialises cleanly to JSON.
    source_dataset : Which corpus this entry came from.
    """

    author_id: str
    raw_text: str
    feature_vector: Dict[str, float]
    source_dataset: str

    @property
    def excerpt(self) -> str:
        """First 200 characters, with newlines collapsed to spaces."""
        return " ".join(self.raw_text[:200].split())


@dataclass
class IndexBundle:
    """Live FAISS index plus the aligned metadata needed to interpret results.

    Fields
    ------
    index        : The FAISS index (IndexFlatL2, L2-normalised entries).
    entries      : TextEntry list aligned 1-to-1 with the FAISS row numbers.
    feature_keys : Canonical sorted key order used when building vectors.
                   Queries must use the same key order.
    """

    index: object          # faiss.IndexFlatL2 — typed as object to keep import optional
    entries: List[TextEntry]
    feature_keys: List[str]

    # ── Derived paths ────────────────────────────────────────────────────────

    @staticmethod
    def _keys_path(metadata_path: str) -> str:
        """Derive the feature-keys JSON path from the metadata JSONL path."""
        return str(Path(metadata_path).with_suffix(".keys.json"))

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        """Persist the FAISS index, entries, and feature keys to disk.

        Parameters default to ``config.FAISS_INDEX_PATH`` and
        ``config.FAISS_METADATA_PATH``.
        """
        import faiss  # type: ignore[import]

        idx_path  = index_path    or config.FAISS_INDEX_PATH
        meta_path = metadata_path or config.FAISS_METADATA_PATH
        keys_path = self._keys_path(meta_path)

        # Ensure parent directories exist
        Path(idx_path).parent.mkdir(parents=True, exist_ok=True)
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, idx_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                d = {
                    "author_id":      entry.author_id,
                    "source_dataset": entry.source_dataset,
                    "raw_text":       entry.raw_text,
                    "feature_vector": entry.feature_vector,
                }
                f.write(json.dumps(d) + "\n")

        with open(keys_path, "w", encoding="utf-8") as f:
            json.dump(self.feature_keys, f)

        print(
            f"[indexer] Saved {self.index.ntotal} vectors to {idx_path} | "
            f"metadata → {meta_path} | keys → {keys_path}"
        )

    @classmethod
    def load(
        cls,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> "IndexBundle":
        """Load a previously saved IndexBundle from disk.

        Raises
        ------
        FileNotFoundError
            If any of the three expected files is missing.
        """
        import faiss  # type: ignore[import]

        idx_path  = index_path    or config.FAISS_INDEX_PATH
        meta_path = metadata_path or config.FAISS_METADATA_PATH
        keys_path = cls._keys_path(meta_path)

        for p in (idx_path, meta_path, keys_path):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"[indexer] Index file missing: {p}\n"
                    "Build the index first:\n"
                    "    from retrieval.indexer import entries_from_pairs, build_index\n"
                    "    from data.preprocessor import load_reuters\n"
                    "    build_index(entries_from_pairs(load_reuters(config.REUTERS_DIR)))"
                )

        faiss_index = faiss.read_index(idx_path)

        entries: List[TextEntry] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                entries.append(TextEntry(
                    author_id=d["author_id"],
                    source_dataset=d["source_dataset"],
                    raw_text=d["raw_text"],
                    feature_vector=d["feature_vector"],
                ))

        with open(keys_path, "r", encoding="utf-8") as f:
            feature_keys = json.load(f)

        print(
            f"[indexer] Loaded {faiss_index.ntotal} vectors from {idx_path} "
            f"({len(entries)} entries, {len(feature_keys)} feature dimensions)"
        )
        return cls(index=faiss_index, entries=entries, feature_keys=feature_keys)


# ──────────────────────────────── Vector Utilities ────────────────────────────

def feature_dict_to_vector(
    feat_dict: Dict[str, float],
    feature_keys: List[str],
) -> np.ndarray:
    """Convert a feature dict to a dense numpy vector using a canonical key order.

    Missing keys are filled with ``0.0``.  The resulting vector is **not**
    normalised here — call ``_l2_normalize`` separately when needed.
    """
    return np.array(
        [feat_dict.get(k, 0.0) for k in feature_keys],
        dtype=np.float32,
    )


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of ``vec``.  Returns ``vec`` unchanged if norm ≈ 0."""
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else vec


# ──────────────────────────────── Index Builder ───────────────────────────────

def build_index(
    entries: List[TextEntry],
    save: bool = True,
    index_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> IndexBundle:
    """Build a FAISS IndexFlatL2 from the feature vectors in ``entries``.

    Parameters
    ----------
    entries       : List of TextEntry objects.  Each must have a non-empty
                    ``feature_vector`` dict (populate via ``entries_from_pairs``).
    save          : If True, persist the bundle to disk after building.
    index_path    : Override for ``config.FAISS_INDEX_PATH``.
    metadata_path : Override for ``config.FAISS_METADATA_PATH``.

    Returns
    -------
    IndexBundle ready for querying.

    Raises
    ------
    ValueError   : If ``entries`` is empty or all feature vectors are empty.
    ImportError  : If ``faiss`` is not installed.
    """
    import faiss  # type: ignore[import]

    if not entries:
        raise ValueError("[indexer] build_index received an empty entries list.")

    # ── 1. Determine canonical feature key space (union across all entries) ──
    all_keys: set = set()
    for e in entries:
        all_keys.update(e.feature_vector.keys())
    if not all_keys:
        raise ValueError(
            "[indexer] All entries have empty feature_vector dicts.  "
            "Run entries_from_pairs() to populate features before calling build_index()."
        )
    feature_keys: List[str] = sorted(all_keys)
    dim = len(feature_keys)

    # ── 2. Build normalised matrix ────────────────────────────────────────────
    matrix = np.stack(
        [_l2_normalize(feature_dict_to_vector(e.feature_vector, feature_keys))
         for e in entries]
    ).astype(np.float32)

    # ── 3. Build FAISS index ──────────────────────────────────────────────────
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(matrix)  # type: ignore[arg-type]

    bundle = IndexBundle(
        index=faiss_index,
        entries=entries,
        feature_keys=feature_keys,
    )

    print(
        f"[indexer] Built index: {faiss_index.ntotal} entries, "
        f"{dim} feature dimensions."
    )

    if save:
        bundle.save(index_path=index_path, metadata_path=metadata_path)

    return bundle


# ──────────────────────────────── Corpus Helper ───────────────────────────────

def entries_from_pairs(
    pairs,  # List[TextPair] — avoid importing here to prevent circular deps
    cefr_dict: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> List[TextEntry]:
    """Extract features for every text in ``pairs`` and return TextEntry objects.

    Each ``TextPair`` contributes *two* TextEntry objects — one for ``text_a``
    and one for ``text_b`` — because both are verified-author samples in the
    Reuters-50-50 corpus.

    Parameters
    ----------
    pairs     : Output of any ``data.preprocessor`` corpus loader.
    cefr_dict : Optional CEFR vocabulary dict.  Pass ``None`` if the wordlist
                is unavailable; CEFR features will be omitted.
    verbose   : Print progress every 100 entries.
    """
    from features.handcrafted import extract_all  # local import avoids circular

    entries: List[TextEntry] = []
    seen: set = set()  # deduplicate by (author_id, text[:80]) to avoid adding same text twice

    for pair in pairs:
        for text, label_side in ((pair.text_a, "a"), (pair.text_b, "b")):
            dedup_key = (pair.author_id, text[:80])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            feat = extract_all(text, cefr_dict=cefr_dict)
            entries.append(TextEntry(
                author_id=pair.author_id,
                raw_text=text,
                feature_vector=feat,
                source_dataset=pair.source_dataset,
            ))

        if verbose and len(entries) % 100 == 0 and len(entries) > 0:
            print(f"  [entries_from_pairs] {len(entries)} entries so far…")

    print(f"[indexer] entries_from_pairs: {len(entries)} unique entries from {len(pairs)} pairs.")
    return entries
