"""Bridge between validated feature vectors and LLM agent prompts.

Takes two per-document feature dicts produced by ``features.handcrafted.extract_all``
and returns a single formatted string ready for direct injection into an agent prompt.

Design constraints
------------------
- No LLM calls, no model loading, no network I/O — pure formatting logic.
- All feature keys match the exact names produced by ``extract_all()`` so the
  caller never needs to rename or re-map anything.
- Missing features (CEFR when wordlist is absent, perplexity when GPT-2 is
  not loaded) produce graceful "not available" lines rather than crashes.
- The output format exactly matches the example in AV_V2_Project_Spec.md.

Public API
----------
    format_evidence_packet(features_a, features_b) -> str

Output structure
----------------
    === STYLOMETRIC EVIDENCE PACKET ===

    VOCABULARY COMPLEXITY:
    - Text A ...
    - Text B ...
    - Interpretation: ...

    SENTENCE STRUCTURE:          (same pattern)
    SYNTACTIC PATTERNS:          (same pattern)
    DISCOURSE & READABILITY:     (same pattern)
    PUNCTUATION & STYLE:         (same pattern)

    SIMILARITY DELTAS (|A - B| per feature):
    - feature_name_delta: value (HIGH/MODERATE/LOW)
    ...
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# ──────────────────────────── Severity Thresholds ─────────────────────────────
#
# Each entry: feature_key_prefix -> (low_threshold, high_threshold)
# Severity:  LOW      if |delta| <  low_threshold
#            MODERATE if low_threshold  <= |delta| < high_threshold
#            HIGH     if |delta| >= high_threshold
#
# Prefixes ending in "_" match any feature key that starts with that prefix.
# Exact-name entries (no trailing "_") match only that exact key.

_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    # CEFR level ratios (0–1)
    "cefr_":                              (0.04, 0.10),
    # Vocabulary richness ratios (0–1)
    "vocab_ttr":                          (0.05, 0.12),
    "vocab_cttr":                         (0.05, 0.15),
    "vocab_avg_word_length":              (0.30, 0.80),
    "vocab_hapax":                        (0.10, 0.30),
    "vocab_lexical":                      (0.04, 0.10),
    # Sentence length — token counts
    "sent_mean":                          (3.0,  7.0),
    "sent_std":                           (2.0,  5.0),
    "sent_entropy":                       (0.30, 0.80),
    "sent_bin_":                          (0.05, 0.12),
    # Sentence structural ratios (0–1)
    "sent_type_":                         (0.05, 0.15),
    "sent_structure":                     (0.10, 0.25),
    "sent_subordinate":                   (0.10, 0.25),
    # POS / syntactic ratios (0–1)
    "pos_":                               (0.03, 0.08),
    "bigram_":                            (0.01, 0.03),
    "adv_":                               (0.05, 0.15),
    "passive_voice":                      (0.05, 0.15),
    # Readability — grade-level scale (roughly 1–16)
    "readability_flesch_kincaid_grade":   (1.5,  3.5),
    "readability_gunning_fog":            (1.5,  3.5),
    "readability_coleman_liau":           (1.5,  3.5),
    # Perplexity — language-model score
    "perplexity":                         (20.0, 60.0),
    # Discourse connectives — rate per 100 words
    "discourse_total":                    (0.50, 1.50),
    "discourse_":                         (0.30, 0.80),
    # Punctuation / style
    "punct_comma":                        (0.30, 0.80),
    "punct_semicolon":                    (0.50, 1.50),
    "punct_exclamation":                  (0.02, 0.06),
    "punct_question":                     (0.03, 0.08),
    "punct_avg_para":                     (20.0, 50.0),
    # Topic-independent stylistic
    "sent_opening_":                      (0.05, 0.15),
    "coord_subord_ratio":                 (2.0,  5.0),   # ratio scale
    "coord_subord_":                      (0.20, 0.60),   # per-sent rates
    "hedging_":                          (0.50, 2.00),
    "contraction_per_100":                (1.00, 4.00),
    "pronoun_":                          (0.05, 0.20),
    "dep_depth_":                        (0.30, 0.80),
}

_DEFAULT_THRESHOLDS: Tuple[float, float] = (0.05, 0.15)


def _get_thresholds(key: str) -> Tuple[float, float]:
    """Return (low, high) thresholds for ``key``, using longest matching prefix."""
    best: Optional[Tuple[float, float]] = None
    best_len = -1
    for prefix, thresholds in _THRESHOLDS.items():
        bare = prefix.rstrip("_")
        if key == bare or key.startswith(prefix):
            if len(prefix) > best_len:
                best = thresholds
                best_len = len(prefix)
    return best if best is not None else _DEFAULT_THRESHOLDS


def _severity(key: str, abs_delta: float) -> str:
    """Return 'LOW', 'MODERATE', or 'HIGH'."""
    low, high = _get_thresholds(key)
    if abs_delta < low:
        return "LOW"
    if abs_delta < high:
        return "MODERATE"
    return "HIGH"
 


# ──────────────────────────── Formatting Micro-Helpers ───────────────────────

def _g(d: dict, key: str, default=None):
    """Safe dict getter."""
    return d.get(key, default)


def _pct(v: Optional[float], decimals: int = 1) -> str:
    """Format a 0–1 ratio as a percentage string, or 'n/a'."""
    if v is None:
        return "n/a"
    return f"{v * 100:.{decimals}f}%"


def _f(v: Optional[float], decimals: int = 2) -> str:
    """Format a float with the given number of decimals, or 'n/a'."""
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}"


def _higher(val_a: Optional[float], val_b: Optional[float]) -> str:
    """Return 'Text A' or 'Text B' — whichever has the higher value."""
    if val_a is None or val_b is None:
        return "one text"
    return "Text B" if (val_b >= val_a) else "Text A"


def _lower(val_a: Optional[float], val_b: Optional[float]) -> str:
    h = _higher(val_a, val_b)
    return "Text A" if h == "Text B" else "Text B"


def _maxv(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return max(a, b)


def _minv(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return min(a, b)


# ──────────────────────────── Section: VOCABULARY ─────────────────────────────

def _build_vocab(fa: dict, fb: dict) -> str:
    lines: List[str] = ["VOCABULARY COMPLEXITY:"]

    # ── CEFR distribution ──────────────────────────────────────────────────
    CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
    fa_cefr = {k: _g(fa, f"cefr_{k}") for k in CEFR_LEVELS}
    fb_cefr = {k: _g(fb, f"cefr_{k}") for k in CEFR_LEVELS}
    has_cefr = any(v is not None for v in fa_cefr.values())

    if has_cefr:
        a_str = ", ".join(f"{k}: {_pct(fa_cefr[k])}" for k in CEFR_LEVELS)
        b_str = ", ".join(f"{k}: {_pct(fb_cefr[k])}" for k in CEFR_LEVELS)
        lines.append(f"- Text A CEFR distribution: {a_str}")
        lines.append(f"- Text B CEFR distribution: {b_str}")
    else:
        lines.append("- CEFR distribution: not available (wordlist not loaded)")

    # ── Vocabulary richness ────────────────────────────────────────────────
    ttr_a, ttr_b = _g(fa, "vocab_ttr"), _g(fb, "vocab_ttr")
    cttr_a, cttr_b = _g(fa, "vocab_cttr"), _g(fb, "vocab_cttr")
    if ttr_a is not None:
        lines.append(
            f"- Text A vocabulary richness: TTR {_f(ttr_a)}, CTTR {_f(cttr_a)}"
        )
        lines.append(
            f"- Text B vocabulary richness: TTR {_f(ttr_b)}, CTTR {_f(cttr_b)}"
        )

    # ── Word length & lexical density ──────────────────────────────────────
    awl_a, awl_b = _g(fa, "vocab_avg_word_length"), _g(fb, "vocab_avg_word_length")
    ld_a, ld_b = _g(fa, "vocab_lexical_density"), _g(fb, "vocab_lexical_density")
    hap_a, hap_b = _g(fa, "vocab_hapax_legomena_ratio"), _g(fb, "vocab_hapax_legomena_ratio")
    if awl_a is not None:
        lines.append(
            f"- Text A: avg word length {_f(awl_a)} chars, "
            f"lexical density {_pct(ld_a)}, hapax ratio {_f(hap_a)}"
        )
        lines.append(
            f"- Text B: avg word length {_f(awl_b)} chars, "
            f"lexical density {_pct(ld_b)}, hapax ratio {_f(hap_b)}"
        )

    # ── Interpretation ─────────────────────────────────────────────────────
    # Primary signal: C1+C2 advanced-vocabulary gap (most diagnostically powerful)
    c1c2_a = (_g(fa, "cefr_C1") or 0.0) + (_g(fa, "cefr_C2") or 0.0)
    c1c2_b = (_g(fb, "cefr_C1") or 0.0) + (_g(fb, "cefr_C2") or 0.0)
    cefr_d = abs(c1c2_b - c1c2_a)
    cefr_sev = _severity("cefr_", cefr_d) if has_cefr else "LOW"

    ttr_d = abs((ttr_b or 0.0) - (ttr_a or 0.0)) if ttr_a is not None else 0.0
    ttr_sev = _severity("vocab_ttr", ttr_d) if ttr_a is not None else "LOW"

    ld_d = abs((ld_b or 0.0) - (ld_a or 0.0)) if ld_a is not None else 0.0
    ld_sev = _severity("vocab_lexical", ld_d) if ld_a is not None else "LOW"

    if cefr_sev == "HIGH":
        h = _higher(c1c2_a, c1c2_b)
        lines.append(
            f"- Interpretation: {h} uses substantially more advanced vocabulary "
            f"(C1+C2: {_pct(_maxv(c1c2_a, c1c2_b))} vs {_pct(_minv(c1c2_a, c1c2_b))}); "
            f"a strong stylometric divergence signal."
        )
    elif cefr_sev == "MODERATE":
        h = _higher(c1c2_a, c1c2_b)
        lines.append(
            f"- Interpretation: {h} uses notably more advanced vocabulary "
            f"(C1+C2: {_pct(_maxv(c1c2_a, c1c2_b))} vs {_pct(_minv(c1c2_a, c1c2_b))})."
        )
    elif ttr_sev != "LOW" and ttr_a is not None:
        h = _higher(ttr_a, ttr_b)
        lines.append(
            f"- Interpretation: {h} shows greater lexical variety "
            f"(TTR: {_f(_maxv(ttr_a, ttr_b))} vs {_f(_minv(ttr_a, ttr_b))}); "
            f"vocabulary choice patterns differ."
        )
    elif ld_sev != "LOW" and ld_a is not None:
        h = _higher(ld_a, ld_b)
        lines.append(
            f"- Interpretation: {h} uses proportionally more content words "
            f"(lexical density: {_pct(_maxv(ld_a, ld_b))} vs {_pct(_minv(ld_a, ld_b))})."
        )
    else:
        lines.append(
            "- Interpretation: Vocabulary complexity is broadly consistent across both texts."
        )

    return "\n".join(lines)


# ──────────────────────────── Section: SENTENCE STRUCTURE ────────────────────

def _build_sentence(fa: dict, fb: dict) -> str:
    lines: List[str] = ["SENTENCE STRUCTURE:"]

    # ── Length statistics ──────────────────────────────────────────────────
    mean_a, mean_b = _g(fa, "sent_mean"), _g(fb, "sent_mean")
    std_a,  std_b  = _g(fa, "sent_std"),  _g(fb, "sent_std")
    if mean_a is not None:
        lines.append(
            f"- Text A mean sentence length: {_f(mean_a, 1)} tokens (std: {_f(std_a, 1)})"
        )
        lines.append(
            f"- Text B mean sentence length: {_f(mean_b, 1)} tokens (std: {_f(std_b, 1)})"
        )

    # ── Length distribution histogram ──────────────────────────────────────
    bvs_a, bvs_b = _g(fa, "sent_bin_very_short"), _g(fb, "sent_bin_very_short")
    bs_a,  bs_b  = _g(fa, "sent_bin_short"),       _g(fb, "sent_bin_short")
    bm_a,  bm_b  = _g(fa, "sent_bin_medium"),      _g(fb, "sent_bin_medium")
    bl_a,  bl_b  = _g(fa, "sent_bin_long"),         _g(fb, "sent_bin_long")
    if bvs_a is not None:
        lines.append(
            f"- Text A sentence lengths: <8 tokens {_pct(bvs_a)}, "
            f"8–15 {_pct(bs_a)}, 15–25 {_pct(bm_a)}, >25 {_pct(bl_a)}"
        )
        lines.append(
            f"- Text B sentence lengths: <8 tokens {_pct(bvs_b)}, "
            f"8–15 {_pct(bs_b)}, 15–25 {_pct(bm_b)}, >25 {_pct(bl_b)}"
        )

    # ── Sentence type distribution (top 3 by Text A share) ────────────────
    TYPE_KEYS = ("Simple", "Compound", "Complex", "Compound-Complex", "Fragment")
    type_a = {k: _g(fa, f"sent_type_{k}") for k in TYPE_KEYS}
    type_b = {k: _g(fb, f"sent_type_{k}") for k in TYPE_KEYS}
    has_types = any(v is not None for v in type_a.values())
    if has_types:
        # Show the three most common types in Text A
        top3 = sorted(
            [(k, type_a[k]) for k in TYPE_KEYS if type_a[k] is not None],
            key=lambda x: x[1], reverse=True
        )[:3]
        a_type = ", ".join(f"{k}: {_pct(v)}" for k, v in top3)
        b_type = ", ".join(f"{k}: {_pct(type_b[k])}" for k, _ in top3)
        lines.append(f"- Text A sentence types (top 3): {a_type}")
        lines.append(f"- Text B sentence types (top 3): {b_type}")

    # ── Structural variation & subordinate clauses ─────────────────────────
    sv_a, sv_b   = _g(fa, "sent_structure_variation"), _g(fb, "sent_structure_variation")
    sub_a, sub_b = _g(fa, "sent_subordinate_freq"),    _g(fb, "sent_subordinate_freq")
    if sv_a is not None:
        lines.append(
            f"- Text A: structure variation {_f(sv_a)}, "
            f"subordinate clause freq {_f(sub_a)} per sentence"
        )
        lines.append(
            f"- Text B: structure variation {_f(sv_b)}, "
            f"subordinate clause freq {_f(sub_b)} per sentence"
        )

    # ── Sentence opening patterns ─────────────────────────────────────────
    OPEN_KEYS = ("subordinate_conj", "adverb", "pronoun", "article")
    open_a = {k: _g(fa, f"sent_opening_{k}") for k in OPEN_KEYS}
    open_b = {k: _g(fb, f"sent_opening_{k}") for k in OPEN_KEYS}
    if any(v is not None for v in open_a.values()):
        a_open = ", ".join(f"{k}: {_pct(open_a[k])}" for k in OPEN_KEYS)
        b_open = ", ".join(f"{k}: {_pct(open_b[k])}" for k in OPEN_KEYS)
        lines.append(f"- Text A sentence openings: {a_open}")
        lines.append(f"- Text B sentence openings: {b_open}")

    # ── Interpretation ─────────────────────────────────────────────────────
    mean_d = abs((mean_b or 0.0) - (mean_a or 0.0)) if mean_a is not None else 0.0
    std_d  = abs((std_b  or 0.0) - (std_a  or 0.0)) if std_a  is not None else 0.0
    sub_d  = abs((sub_b  or 0.0) - (sub_a  or 0.0)) if sub_a  is not None else 0.0

    mean_sev = _severity("sent_mean",       mean_d) if mean_a is not None else "LOW"
    std_sev  = _severity("sent_std",        std_d)  if std_a  is not None else "LOW"
    sub_sev  = _severity("sent_subordinate", sub_d) if sub_a  is not None else "LOW"

    if mean_sev == "HIGH":
        h = _higher(mean_a, mean_b)
        var_clause = (
            " with considerably higher sentence-to-sentence variability"
            if std_sev != "LOW" else ""
        )
        lines.append(
            f"- Interpretation: {h} uses significantly longer sentences "
            f"({_f(_maxv(mean_a, mean_b), 1)} vs {_f(_minv(mean_a, mean_b), 1)} tokens){var_clause}; "
            f"sentence rhythm is a strong divergence signal here."
        )
    elif mean_sev == "MODERATE":
        h = _higher(mean_a, mean_b)
        lines.append(
            f"- Interpretation: {h} tends toward longer sentences "
            f"({_f(_maxv(mean_a, mean_b), 1)} vs {_f(_minv(mean_a, mean_b), 1)} tokens)."
        )
    elif std_sev != "LOW" and std_a is not None:
        h = _higher(std_a, std_b)
        lines.append(
            f"- Interpretation: Despite comparable averages, {h} has "
            f"substantially more variable sentence rhythm "
            f"(std: {_f(_maxv(std_a, std_b), 1)} vs {_f(_minv(std_a, std_b), 1)}), "
            f"suggesting less controlled pacing."
        )
    elif sub_sev != "LOW" and sub_a is not None:
        h = _higher(sub_a, sub_b)
        lines.append(
            f"- Interpretation: {h} makes greater use of subordinate clauses "
            f"({_f(_maxv(sub_a, sub_b), 2)} vs {_f(_minv(sub_a, sub_b), 2)} per sentence), "
            f"indicating more syntactically complex sentence construction."
        )
    else:
        lines.append(
            "- Interpretation: Sentence structure patterns are broadly consistent across both texts."
        )

    return "\n".join(lines)


# ──────────────────────────── Section: SYNTACTIC PATTERNS ────────────────────

def _build_syntactic(fa: dict, fb: dict) -> str:
    lines: List[str] = ["SYNTACTIC PATTERNS:"]

    # ── POS distribution ───────────────────────────────────────────────────
    POS_CATS = ("NOUN", "VERB", "ADJ", "ADV", "AUX", "CCONJ", "SCONJ")
    pos_a = {c: _g(fa, f"pos_{c}") for c in POS_CATS}
    pos_b = {c: _g(fb, f"pos_{c}") for c in POS_CATS}
    has_pos = any(v is not None for v in pos_a.values())
    if has_pos:
        shown = [c for c in POS_CATS if pos_a[c] is not None][:5]
        a_str = ", ".join(f"{c}: {_pct(pos_a[c])}" for c in shown)
        b_str = ", ".join(f"{c}: {_pct(pos_b[c])}" for c in shown)
        lines.append(f"- Text A POS distribution: {a_str}")
        lines.append(f"- Text B POS distribution: {b_str}")

    # ── Adverbial placement ────────────────────────────────────────────────
    ADV_CATS = ("sentence_initial", "preverbal", "postverbal", "sentence_final")
    adv_a = {k: _g(fa, f"adv_{k}") for k in ADV_CATS}
    adv_b = {k: _g(fb, f"adv_{k}") for k in ADV_CATS}
    has_adv = any(v is not None for v in adv_a.values())
    if has_adv:
        a_adv = ", ".join(
            f"{k.replace('_', '-')}: {_pct(adv_a[k])}"
            for k in ADV_CATS if adv_a[k] is not None
        )
        b_adv = ", ".join(
            f"{k.replace('_', '-')}: {_pct(adv_b[k])}"
            for k in ADV_CATS if adv_b[k] is not None
        )
        lines.append(f"- Text A adverbial placement: {a_adv}")
        lines.append(f"- Text B adverbial placement: {b_adv}")

    # ── Passive voice ──────────────────────────────────────────────────────
    pv_a, pv_b = _g(fa, "passive_voice_freq"), _g(fb, "passive_voice_freq")
    if pv_a is not None:
        lines.append(
            f"- Text A passive voice: {_f(pv_a, 3)} per sentence  |  "
            f"Text B: {_f(pv_b, 3)}"
        )

    # ── POS bigrams (most diagnostic) ─────────────────────────────────────
    BG_KEYS = ("ADJ_NOUN", "NOUN_VERB", "VERB_ADV", "ADV_ADJ")
    bg_a = {k: _g(fa, f"bigram_{k}") for k in BG_KEYS}
    bg_b = {k: _g(fb, f"bigram_{k}") for k in BG_KEYS}
    if any(v is not None for v in bg_a.values()):
        a_bg = ", ".join(f"{k}: {_f(bg_a[k], 3)}" for k in BG_KEYS if bg_a[k] is not None)
        b_bg = ", ".join(f"{k}: {_f(bg_b[k], 3)}" for k in BG_KEYS if bg_b[k] is not None)
        lines.append(f"- Text A POS bigrams: {a_bg}")
        lines.append(f"- Text B POS bigrams: {b_bg}")

    # ── Coordination vs subordination ──────────────────────────────────────
    coord_a, coord_b = _g(fa, "coord_subord_coord_per_sent"), _g(fb, "coord_subord_coord_per_sent")
    subord_a, subord_b = _g(fa, "coord_subord_subord_per_sent"), _g(fb, "coord_subord_subord_per_sent")
    ratio_a, ratio_b = _g(fa, "coord_subord_ratio"), _g(fb, "coord_subord_ratio")
    if coord_a is not None:
        lines.append(
            f"- Text A coord/subord: {_f(coord_a, 2)} coord/sent, "
            f"{_f(subord_a, 2)} subord/sent, ratio {_f(ratio_a, 2)}"
        )
        lines.append(
            f"- Text B coord/subord: {_f(coord_b, 2)} coord/sent, "
            f"{_f(subord_b, 2)} subord/sent, ratio {_f(ratio_b, 2)}"
        )

    # ── Pronoun distribution (first-person singular) ────────────────────────
    pron_a, pron_b = _g(fa, "pronoun_first_person_singular_ratio"), _g(fb, "pronoun_first_person_singular_ratio")
    if pron_a is not None:
        lines.append(
            f"- Text A first-person singular pronoun ratio: {_pct(pron_a)}  |  "
            f"Text B: {_pct(pron_b)}"
        )

    # ── Dependency depth ──────────────────────────────────────────────────
    dep_a, dep_b = _g(fa, "dep_depth_mean_depth"), _g(fb, "dep_depth_mean_depth")
    if dep_a is not None:
        lines.append(
            f"- Text A avg dependency depth: {_f(dep_a, 2)}  |  Text B: {_f(dep_b, 2)}"
        )

    # ── Interpretation ─────────────────────────────────────────────────────
    pv_d = abs((pv_b or 0.0) - (pv_a or 0.0)) if pv_a is not None else 0.0
    pv_sev = _severity("passive_voice", pv_d) if pv_a is not None else "LOW"

    adv_si_a = _g(fa, "adv_sentence_initial")
    adv_si_b = _g(fb, "adv_sentence_initial")
    adv_si_d = abs((adv_si_b or 0.0) - (adv_si_a or 0.0)) if adv_si_a is not None else 0.0
    adv_si_sev = _severity("adv_", adv_si_d) if adv_si_a is not None else "LOW"

    # Best POS divergence across the five main categories
    best_pos_sev, best_pos_cat, best_pos_d = "LOW", "", 0.0
    SEV_RANK = {"HIGH": 2, "MODERATE": 1, "LOW": 0}
    for cat in POS_CATS:
        v_a, v_b = pos_a.get(cat), pos_b.get(cat)
        if v_a is not None and v_b is not None:
            d = abs(v_b - v_a)
            s = _severity("pos_", d)
            if SEV_RANK[s] > SEV_RANK[best_pos_sev]:
                best_pos_sev, best_pos_cat, best_pos_d = s, cat, d

    if pv_sev == "HIGH":
        h = _higher(pv_a, pv_b)
        lines.append(
            f"- Interpretation: {h} relies substantially more on passive constructions "
            f"({_f(_maxv(pv_a, pv_b), 3)} vs {_f(_minv(pv_a, pv_b), 3)} per sentence); "
            f"a notable register and style marker."
        )
    elif adv_si_sev != "LOW" and adv_si_a is not None:
        h = _higher(adv_si_a, adv_si_b)
        lines.append(
            f"- Interpretation: {h} more frequently opens sentences with adverbs "
            f"({_pct(_maxv(adv_si_a, adv_si_b))} vs {_pct(_minv(adv_si_a, adv_si_b))}); "
            f"a distinguishing sentence-initial style habit."
        )
    elif best_pos_sev != "LOW" and best_pos_cat:
        va = pos_a.get(best_pos_cat, 0.0) or 0.0
        vb = pos_b.get(best_pos_cat, 0.0) or 0.0
        h = "Text B" if vb > va else "Text A"
        lines.append(
            f"- Interpretation: {h} uses {best_pos_cat} at a notably different rate "
            f"({_pct(max(va, vb))} vs {_pct(min(va, vb))}); "
            f"the overall POS profile diverges at this category."
        )
    else:
        lines.append(
            "- Interpretation: Syntactic patterns show limited divergence between texts."
        )

    return "\n".join(lines)


# ──────────────────────────── Section: DISCOURSE & READABILITY ───────────────

def _build_discourse(fa: dict, fb: dict) -> str:
    lines: List[str] = ["DISCOURSE & READABILITY:"]

    # ── Readability metrics ────────────────────────────────────────────────
    fk_a  = _g(fa, "readability_flesch_kincaid_grade")
    fk_b  = _g(fb, "readability_flesch_kincaid_grade")
    gf_a  = _g(fa, "readability_gunning_fog")
    gf_b  = _g(fb, "readability_gunning_fog")
    cl_a  = _g(fa, "readability_coleman_liau")
    cl_b  = _g(fb, "readability_coleman_liau")
    if fk_a is not None:
        lines.append(
            f"- Text A readability: FK grade {_f(fk_a, 1)}, "
            f"Gunning Fog {_f(gf_a, 1)}, Coleman-Liau {_f(cl_a, 1)}"
        )
        lines.append(
            f"- Text B readability: FK grade {_f(fk_b, 1)}, "
            f"Gunning Fog {_f(gf_b, 1)}, Coleman-Liau {_f(cl_b, 1)}"
        )

    # ── Perplexity ─────────────────────────────────────────────────────────
    pxl_a, pxl_b = _g(fa, "perplexity"), _g(fb, "perplexity")
    if pxl_a is not None:
        lines.append(
            f"- Text A perplexity (GPT-2): {_f(pxl_a, 1)}  |  Text B: {_f(pxl_b, 1)}"
        )

    # ── Hedging language ───────────────────────────────────────────────────
    hedge_a, hedge_b = _g(fa, "hedging_per_100"), _g(fb, "hedging_per_100")
    if hedge_a is not None:
        lines.append(
            f"- Text A hedging expressions /100 words: {_f(hedge_a, 2)}  |  "
            f"Text B: {_f(hedge_b, 2)}"
        )

    # ── Discourse connectives ──────────────────────────────────────────────
    dc_tot_a   = _g(fa, "discourse_total_per_100")
    dc_tot_b   = _g(fb, "discourse_total_per_100")
    dc_ctr_a   = _g(fa, "discourse_contrastive_per_100")
    dc_ctr_b   = _g(fb, "discourse_contrastive_per_100")
    dc_add_a   = _g(fa, "discourse_additive_per_100")
    dc_add_b   = _g(fb, "discourse_additive_per_100")
    dc_cau_a   = _g(fa, "discourse_causal_per_100")
    dc_cau_b   = _g(fb, "discourse_causal_per_100")
    if dc_tot_a is not None:
        lines.append(
            f"- Text A discourse connectives /100 words: total {_f(dc_tot_a, 2)}, "
            f"contrastive {_f(dc_ctr_a, 2)}, additive {_f(dc_add_a, 2)}, "
            f"causal {_f(dc_cau_a, 2)}"
        )
        lines.append(
            f"- Text B discourse connectives /100 words: total {_f(dc_tot_b, 2)}, "
            f"contrastive {_f(dc_ctr_b, 2)}, additive {_f(dc_add_b, 2)}, "
            f"causal {_f(dc_cau_b, 2)}"
        )

    # ── Interpretation ─────────────────────────────────────────────────────
    fk_d   = abs((fk_b or 0.0) - (fk_a or 0.0)) if fk_a is not None else 0.0
    pxl_d  = abs((pxl_b or 0.0) - (pxl_a or 0.0)) if pxl_a is not None else 0.0
    dc_d   = abs((dc_tot_b or 0.0) - (dc_tot_a or 0.0)) if dc_tot_a is not None else 0.0

    fk_sev  = _severity("readability_flesch_kincaid_grade", fk_d)  if fk_a  is not None else "LOW"
    pxl_sev = _severity("perplexity",  pxl_d) if pxl_a is not None else "LOW"
    dc_sev  = _severity("discourse_total", dc_d) if dc_tot_a is not None else "LOW"

    if fk_sev == "HIGH":
        h = _higher(fk_a, fk_b)
        lines.append(
            f"- Interpretation: {h} is substantially more complex by grade-level metrics "
            f"(FK grade: {_f(_maxv(fk_a, fk_b), 1)} vs {_f(_minv(fk_a, fk_b), 1)}); "
            f"a strong indicator of different authorial sophistication."
        )
    elif fk_sev == "MODERATE":
        h = _higher(fk_a, fk_b)
        lines.append(
            f"- Interpretation: {h} is measurably more complex "
            f"(FK grade: {_f(_maxv(fk_a, fk_b), 1)} vs {_f(_minv(fk_a, fk_b), 1)})."
        )
    elif pxl_sev == "HIGH" and pxl_a is not None:
        # Lower perplexity = more predictable / fluent writing
        h = _higher(pxl_b, pxl_a)  # flip: higher perplexity = LESS predictable
        lines.append(
            f"- Interpretation: {h} is written with considerably more predictable "
            f"language patterns (lower GPT-2 perplexity: "
            f"{_f(_minv(pxl_a, pxl_b), 1)} vs {_f(_maxv(pxl_a, pxl_b), 1)})."
        )
    elif dc_sev != "LOW" and dc_tot_a is not None:
        h = _higher(dc_tot_a, dc_tot_b)
        lines.append(
            f"- Interpretation: {h} employs discourse connectives at a higher rate "
            f"({_f(_maxv(dc_tot_a, dc_tot_b), 2)} vs {_f(_minv(dc_tot_a, dc_tot_b), 2)} "
            f"per 100 words), suggesting more explicit argumentative structuring."
        )
    else:
        lines.append(
            "- Interpretation: Readability and discourse organisation are at comparable levels."
        )

    return "\n".join(lines)


# ──────────────────────────── Section: PUNCTUATION & STYLE ───────────────────

def _build_punctuation(fa: dict, fb: dict) -> str:
    lines: List[str] = ["PUNCTUATION & STYLE:"]

    comma_a = _g(fa, "punct_comma_per_sentence")
    comma_b = _g(fb, "punct_comma_per_sentence")
    sc_a    = _g(fa, "punct_semicolon_colon_rate")
    sc_b    = _g(fb, "punct_semicolon_colon_rate")
    excl_a  = _g(fa, "punct_exclamation_ratio")
    excl_b  = _g(fb, "punct_exclamation_ratio")
    quest_a = _g(fa, "punct_question_ratio")
    quest_b = _g(fb, "punct_question_ratio")
    para_a  = _g(fa, "punct_avg_paragraph_length")
    para_b  = _g(fb, "punct_avg_paragraph_length")

    if comma_a is not None:
        lines.append(
            f"- Text A: commas/sentence {_f(comma_a, 2)}, "
            f"semicolon+colon rate {_f(sc_a, 2)}/100 tokens, "
            f"exclamation ratio {_pct(excl_a)}, question ratio {_pct(quest_a)}"
        )
        lines.append(
            f"- Text B: commas/sentence {_f(comma_b, 2)}, "
            f"semicolon+colon rate {_f(sc_b, 2)}/100 tokens, "
            f"exclamation ratio {_pct(excl_b)}, question ratio {_pct(quest_b)}"
        )
    else:
        lines.append("- Punctuation features: not available")

    if para_a is not None:
        lines.append(
            f"- Text A avg paragraph length: {_f(para_a, 1)} words  |  "
            f"Text B: {_f(para_b, 1)} words"
        )

    # ── Contraction rate ──────────────────────────────────────────────────
    contr_a, contr_b = _g(fa, "contraction_per_100"), _g(fb, "contraction_per_100")
    if contr_a is not None:
        lines.append(
            f"- Text A contractions /100 words: {_f(contr_a, 2)}  |  Text B: {_f(contr_b, 2)}"
        )

    # ── Interpretation ─────────────────────────────────────────────────────
    comma_d = abs((comma_b or 0.0) - (comma_a or 0.0)) if comma_a is not None else 0.0
    para_d  = abs((para_b  or 0.0) - (para_a  or 0.0)) if para_a  is not None else 0.0
    sc_d    = abs((sc_b    or 0.0) - (sc_a    or 0.0)) if sc_a    is not None else 0.0

    comma_sev = _severity("punct_comma",   comma_d) if comma_a is not None else "LOW"
    para_sev  = _severity("punct_avg_para", para_d) if para_a  is not None else "LOW"
    sc_sev    = _severity("punct_semicolon", sc_d)  if sc_a   is not None else "LOW"

    if comma_sev == "HIGH":
        h = _higher(comma_a, comma_b)
        lines.append(
            f"- Interpretation: {h} uses significantly more comma-separated clauses "
            f"({_f(_maxv(comma_a, comma_b), 2)} vs {_f(_minv(comma_a, comma_b), 2)} "
            f"per sentence), indicating a denser internal sentence structure."
        )
    elif para_sev != "LOW" and para_a is not None:
        h = _higher(para_a, para_b)
        lines.append(
            f"- Interpretation: {h} uses substantially longer paragraphs "
            f"({_f(_maxv(para_a, para_b), 1)} vs {_f(_minv(para_a, para_b), 1)} words), "
            f"reflecting different text-organisation habits."
        )
    elif sc_sev != "LOW" and sc_a is not None:
        h = _higher(sc_a, sc_b)
        lines.append(
            f"- Interpretation: {h} makes heavier use of semicolons and colons "
            f"({_f(_maxv(sc_a, sc_b), 2)} vs {_f(_minv(sc_a, sc_b), 2)} per 100 tokens), "
            f"a stylistically distinctive punctuation habit."
        )
    else:
        lines.append(
            "- Interpretation: Punctuation and paragraph patterns are consistent between texts."
        )

    return "\n".join(lines)


# ──────────────────────────── Section: SIMILARITY DELTAS ─────────────────────

def _build_deltas(fa: dict, fb: dict) -> str:
    """Build the sorted SIMILARITY DELTAS summary section."""

    deltas: List[Tuple[str, float, str, str]] = []

    def _reg(display: str, val_a, val_b, threshold_key: str, topic_tag: Optional[str] = None) -> None:
        """Register a named delta if both values are present.
        topic_tag: 'independent' → ⚠ topic-independent; 'influenced' → (topic-influenced, lower weight)
        """
        if val_a is None or val_b is None:
            return
        d = abs(float(val_b) - float(val_a))
        sev = _severity(threshold_key, d)
        tag = ""
        if topic_tag == "independent":
            tag = " ⚠ topic-independent"
        elif topic_tag == "influenced":
            tag = " (topic-influenced, lower weight)"
        deltas.append((display, d, sev, tag))

    # ── Derived compound features ──────────────────────────────────────────
    c1c2_a = (_g(fa, "cefr_C1") or 0.0) + (_g(fa, "cefr_C2") or 0.0)
    c1c2_b = (_g(fb, "cefr_C1") or 0.0) + (_g(fb, "cefr_C2") or 0.0)
    if _g(fa, "cefr_C1") is not None:
        _reg("cefr_c1c2_ratio", c1c2_a, c1c2_b, "cefr_")

    # ── Vocabulary ────────────────────────────────────────────────────────
    _reg("vocab_type_token_ratio",   _g(fa, "vocab_ttr"),                _g(fb, "vocab_ttr"),                "vocab_ttr")
    _reg("vocab_avg_word_length",    _g(fa, "vocab_avg_word_length"),     _g(fb, "vocab_avg_word_length"),    "vocab_avg_word_length")
    _reg("vocab_lexical_density",    _g(fa, "vocab_lexical_density"),     _g(fb, "vocab_lexical_density"),    "vocab_lexical")
    _reg("vocab_hapax_ratio",        _g(fa, "vocab_hapax_legomena_ratio"),_g(fb, "vocab_hapax_legomena_ratio"),"vocab_hapax", "independent")

    # ── Sentence-level ────────────────────────────────────────────────────
    _reg("sentence_length_mean",         _g(fa, "sent_mean"),               _g(fb, "sent_mean"),               "sent_mean", "influenced")
    _reg("sentence_length_std",          _g(fa, "sent_std"),                _g(fb, "sent_std"),                "sent_std")
    _reg("sentence_structure_variation", _g(fa, "sent_structure_variation"),_g(fb, "sent_structure_variation"),"sent_structure")
    _reg("subordinate_clause_freq",      _g(fa, "sent_subordinate_freq"),   _g(fb, "sent_subordinate_freq"),   "sent_subordinate")
    _reg("sent_type_simple_ratio",       _g(fa, "sent_type_Simple"),        _g(fb, "sent_type_Simple"),        "sent_type_")
    _reg("sent_type_complex_ratio",      _g(fa, "sent_type_Complex"),       _g(fb, "sent_type_Complex"),       "sent_type_")
    _reg("sent_opening_subord_conj",     _g(fa, "sent_opening_subordinate_conj"), _g(fb, "sent_opening_subordinate_conj"), "sent_opening_", "independent")
    _reg("sent_opening_adverb",          _g(fa, "sent_opening_adverb"),      _g(fb, "sent_opening_adverb"),     "sent_opening_", "independent")
    _reg("sent_opening_pronoun",         _g(fa, "sent_opening_pronoun"),    _g(fb, "sent_opening_pronoun"),    "sent_opening_", "independent")
    _reg("sent_opening_article",         _g(fa, "sent_opening_article"),    _g(fb, "sent_opening_article"),    "sent_opening_", "independent")

    # ── Syntactic / POS ───────────────────────────────────────────────────
    _reg("passive_voice_freq",           _g(fa, "passive_voice_freq"),      _g(fb, "passive_voice_freq"),      "passive_voice", "independent")
    _reg("coord_subord_ratio",           _g(fa, "coord_subord_ratio"),      _g(fb, "coord_subord_ratio"),      "coord_subord_", "independent")
    _reg("pronoun_first_person_ratio",   _g(fa, "pronoun_first_person_singular_ratio"), _g(fb, "pronoun_first_person_singular_ratio"), "pronoun_", "independent")
    _reg("dep_depth_mean",               _g(fa, "dep_depth_mean_depth"),    _g(fb, "dep_depth_mean_depth"),   "dep_depth_", "independent")
    _reg("adv_sentence_initial",         _g(fa, "adv_sentence_initial"),    _g(fb, "adv_sentence_initial"),    "adv_", "independent")
    _reg("pos_noun_ratio",               _g(fa, "pos_NOUN"),                _g(fb, "pos_NOUN"),                "pos_")
    _reg("pos_verb_ratio",               _g(fa, "pos_VERB"),                _g(fb, "pos_VERB"),                "pos_")
    _reg("pos_adj_ratio",                _g(fa, "pos_ADJ"),                 _g(fb, "pos_ADJ"),                 "pos_")
    _reg("pos_adv_ratio",                _g(fa, "pos_ADV"),                 _g(fb, "pos_ADV"),                 "pos_")

    # ── Discourse / Readability ───────────────────────────────────────────
    _reg("readability_fk_grade",         _g(fa, "readability_flesch_kincaid_grade"),
                                         _g(fb, "readability_flesch_kincaid_grade"),
                                         "readability_flesch_kincaid_grade", "influenced")
    _reg("readability_gunning_fog",      _g(fa, "readability_gunning_fog"), _g(fb, "readability_gunning_fog"), "readability_gunning_fog", "influenced")
    _reg("perplexity",                   _g(fa, "perplexity"),              _g(fb, "perplexity"),              "perplexity", "influenced")
    _reg("discourse_connective_total",   _g(fa, "discourse_total_per_100"), _g(fb, "discourse_total_per_100"), "discourse_total")
    _reg("hedging_per_100",              _g(fa, "hedging_per_100"),        _g(fb, "hedging_per_100"),        "hedging_", "independent")

    # ── Punctuation / Style ───────────────────────────────────────────────
    _reg("comma_per_sentence",           _g(fa, "punct_comma_per_sentence"),   _g(fb, "punct_comma_per_sentence"),   "punct_comma")
    _reg("avg_paragraph_length",         _g(fa, "punct_avg_paragraph_length"), _g(fb, "punct_avg_paragraph_length"), "punct_avg_para", "influenced")
    _reg("contraction_per_100",          _g(fa, "contraction_per_100"),        _g(fb, "contraction_per_100"),        "contraction_per_100", "independent")

    # Sort: HIGH first, then MODERATE, then LOW; descending abs_delta within tier
    SEV_RANK = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    deltas.sort(key=lambda x: (SEV_RANK[x[2]], -x[1]))

    lines: List[str] = [
        "SIMILARITY DELTAS (|A - B| per feature):",
        "⚠ topic-independent = reliable across subject changes",
        "(topic-influenced) = may reflect subject matter, not authorship",
        "",
    ]
    if not deltas:
        lines.append("- (No features available to compare)")
    else:
        for name, delta, sev, tag in deltas:
            lines.append(f"- {name}_delta: {_f(delta, 3)} ({sev}){tag}")

    return "\n".join(lines)


# ──────────────────────────── Public API ─────────────────────────────────────

def format_evidence_packet(features_a: dict, features_b: dict) -> str:
    """Format two per-document feature dicts into a structured prompt string.

    Parameters
    ----------
    features_a : Feature dict for Text A (verified author reference text).
                 Must be the flat dict produced by ``extract_all()`` in
                 ``features.handcrafted``.
    features_b : Feature dict for Text B (unknown / test text).

    Returns
    -------
    A multi-line plain-text string exactly matching the format specified in
    AV_V2_Project_Spec.md, ready for direct injection into an LLM agent
    prompt.  No LLM calls are made here.

    Missing features (e.g. CEFR when the wordlist is not loaded, perplexity
    when GPT-2 is not available) produce graceful "not available" lines so the
    packet is always a complete, well-formed string.
    """
    if not features_a and not features_b:
        return (
            "=== STYLOMETRIC EVIDENCE PACKET ===\n\n"
            "(No features available — feature extraction may not have been run.)"
        )

    sections = [
        "=== STYLOMETRIC EVIDENCE PACKET ===",
        "",
        _build_vocab(features_a, features_b),
        "",
        _build_sentence(features_a, features_b),
        "",
        _build_syntactic(features_a, features_b),
        "",
        _build_discourse(features_a, features_b),
        "",
        _build_punctuation(features_a, features_b),
        "",
        _build_deltas(features_a, features_b),
    ]
    return "\n".join(sections)
