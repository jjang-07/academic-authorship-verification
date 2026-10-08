"""Stylometric retriever: query the FAISS index and format results for LLM agents.

Public API
----------
    retriever = Retriever()                                  # auto-loads index
    results_a = retriever.retrieve_from_features(feats_a)   # List[RetrievalResult]
    results_b = retriever.retrieve_from_features(feats_b)
    context   = Retriever.format_context(results_a, results_b)  # str for agent prompt

The formatted context is ready to be passed as ``retrieved_context`` to
``AnalystAgent.analyze(pair, evidence_packet, retrieved_context=context)``.

Distance → similarity conversion
---------------------------------
Vectors are L2-normalised before insertion into the index, so FAISS returns
squared Euclidean distances d² where:

    cosine_similarity = 1 − d² / 2

Values are clamped to [0.0, 1.0] to handle tiny floating-point overshoots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from retrieval.indexer import IndexBundle, TextEntry, feature_dict_to_vector, _l2_normalize


# ──────────────────────────────── Result Type ─────────────────────────────────

@dataclass
class RetrievalResult:
    """One retrieved writing sample together with its similarity score.

    Fields
    ------
    entry      : The matched TextEntry (author_id, raw_text, feature_vector, …).
    similarity : Cosine similarity to the query vector, in [0.0, 1.0].
    rank       : 1-based position in the ranked list (1 = closest match).
    """

    entry: TextEntry
    similarity: float
    rank: int


# ──────────────────────────────── Retriever ───────────────────────────────────

class Retriever:
    """Query the FAISS stylometric index and format results for agent consumption.

    Parameters
    ----------
    bundle : Pre-loaded ``IndexBundle``.  If ``None``, the index is loaded
             lazily from the paths configured in ``config.py`` on the first
             call to ``retrieve_from_features``.

    Usage
    -----
        # Typical pipeline usage:
        retriever = Retriever()
        results = retriever.retrieve_from_features(features_dict, k=3)
        context_str = Retriever.format_context(results_a, results_b)

        # Offline index build (run once before the pipeline):
        from retrieval.indexer import entries_from_pairs, build_index
        from data.preprocessor import load_reuters
        pairs = load_reuters(config.REUTERS_DIR)
        build_index(entries_from_pairs(pairs, cefr_dict=get_cefr_dict()))
    """

    def __init__(self, bundle: Optional[IndexBundle] = None) -> None:
        self._bundle: Optional[IndexBundle] = bundle

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def _get_bundle(self) -> IndexBundle:
        if self._bundle is None:
            self._bundle = IndexBundle.load()
        return self._bundle

    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    # ── Core retrieval ───────────────────────────────────────────────────────

    def retrieve_from_features(
        self,
        query_features: Dict[str, float],
        k: int = 3,
    ) -> List[RetrievalResult]:
        """Return the top-k indexed entries most stylistically similar to ``query_features``.

        Parameters
        ----------
        query_features : Feature dict from ``features.handcrafted.extract_all()``.
        k              : Number of results to return.

        Returns
        -------
        List of ``RetrievalResult`` objects sorted by similarity (highest first).
        Returns an empty list if the index is unavailable or the query vector is
        all zeros (degenerate text).

        Raises
        ------
        FileNotFoundError
            If the index has never been built (propagated from ``IndexBundle.load``).
        """
        bundle = self._get_bundle()

        vec = _l2_normalize(
            feature_dict_to_vector(query_features, bundle.feature_keys)
        ).reshape(1, -1).astype(np.float32)

        # Guard: degenerate vector (all-zero text or no overlapping features)
        if float(np.linalg.norm(vec)) < 1e-8:
            return []

        n_results = min(k, bundle.index.ntotal)
        distances, indices = bundle.index.search(vec, n_results)  # type: ignore[attr-defined]

        results: List[RetrievalResult] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
            # Convert squared L2 distance on normalised vectors to cosine similarity
            cosine_sim = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))
            results.append(RetrievalResult(
                entry=bundle.entries[idx],
                similarity=cosine_sim,
                rank=rank,
            ))

        return results

    # ── Formatting ───────────────────────────────────────────────────────────

    @staticmethod
    def _format_one(result: RetrievalResult) -> str:
        """Render a single RetrievalResult as a compact multi-line string."""
        e   = result.entry
        fv  = e.feature_vector

        def _f(key: str, decimals: int = 3) -> str:
            v = fv.get(key)
            return f"{v:.{decimals}f}" if v is not None else "n/a"

        lines = [
            f"  {result.rank}. Author: {e.author_id} | "
            f"Source: {e.source_dataset} | "
            f"Style-similarity: {result.similarity:.3f}",
            f"     Vocabulary  : TTR={_f('vocab_ttr')}, "
            f"lexical_density={_f('vocab_lexical_density')}, "
            f"avg_word_len={_f('vocab_avg_word_length', 1)}",
            f"     Sentences   : mean={_f('sent_mean', 1)} tokens, "
            f"std={_f('sent_std', 1)}, "
            f"structure_var={_f('sent_structure_variation')}",
            f"     Syntax      : noun_ratio={_f('pos_NOUN')}, "
            f"passive_voice={_f('passive_voice_freq')}, "
            f"adv_sent_initial={_f('adv_sentence_initial')}",
            f"     Style       : commas/sent={_f('punct_comma_per_sentence', 2)}, "
            f"FK_grade={_f('readability_flesch_kincaid_grade', 1)}, "
            f"discourse_total={_f('discourse_total', 2)}",
            f"     Excerpt     : \"{e.excerpt}\"",
        ]
        return "\n".join(lines)

    @staticmethod
    def format_context(
        results_a: List[RetrievalResult],
        results_b: List[RetrievalResult],
    ) -> str:
        """Format two ranked result lists into a string for LLM agent prompts.

        The returned string is passed as ``retrieved_context`` to
        ``AnalystAgent.analyze()``.  It describes each retrieved sample's key
        stylometric characteristics so the agent can reason about which author
        Text A or B most resembles stylistically.

        Parameters
        ----------
        results_a : Results for Text A (query = features_a).
        results_b : Results for Text B (query = features_b).

        Returns
        -------
        A formatted string ready for direct injection into the agent prompt.
        Returns an empty string if both result lists are empty.
        """
        if not results_a and not results_b:
            return ""

        sections: List[str] = [
            "=== RETRIEVED STYLISTICALLY SIMILAR SAMPLES ===",
            "(Retrieval is based on stylometric feature similarity — not topic or content.)",
        ]

        if results_a:
            sections.append(
                f"\n[Text A Anchors — top {len(results_a)} most stylistically similar"
                " verified writings]"
            )
            for r in results_a:
                sections.append(Retriever._format_one(r))
        else:
            sections.append("\n[Text A Anchors — no index results available]")

        if results_b:
            sections.append(
                f"\n[Text B Anchors — top {len(results_b)} most stylistically similar"
                " verified writings]"
            )
            for r in results_b:
                sections.append(Retriever._format_one(r))
        else:
            sections.append("\n[Text B Anchors — no index results available]")

        return "\n".join(sections)
