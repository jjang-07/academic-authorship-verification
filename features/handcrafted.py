"""Single-document handcrafted stylometric feature extraction for V2.

Every public function in this module accepts a single document — either a raw
text string or a pre-computed spaCy ``Doc`` — and returns a ``float`` (scalar
feature) or a ``dict`` (multi-valued feature family).

The pre-computed-Doc overload exists for performance: ``features/extractor.py``
calls ``_get_nlp()(text)`` exactly once per document and passes the resulting
``Doc`` to every feature function, avoiding repeated full parses.

Feature families implemented
-----------------------------
Vocabulary        : cefr_distribution, type_token_ratio, avg_word_length,
                    hapax_legomena_ratio, lexical_density
Sentence-level    : sentence_length_stats, sentence_type_distribution,
                    sentence_structure_variation, subordinate_clause_frequency
Syntactic / POS   : pos_unigram_distribution, pos_bigram_patterns,
                    adverbial_placement_distribution, passive_voice_frequency
Discourse /
 Readability      : readability_metrics, calculate_perplexity,
                    discourse_connective_frequency         ← new in V2
Punctuation/Style : punctuation_style                     ← new in V2

V1 functions ported (renamed for clarity)
------------------------------------------
V1 name                           → V2 name
calculate_pxl_optimized           → calculate_perplexity
calculate_sentence_length_stats   → sentence_length_stats  (extended)
calculate_sentence_type           → sentence_type_distribution
count_cefr_levels_exclude_unknown → cefr_distribution
readability_metric (Flesch only)  → readability_metrics  (3 scores)
adverbial_placement               → adverbial_placement_distribution

POS-tagging dependency
-----------------------
spaCy (``en_core_web_md`` by default) is the sole POS/dependency-parse
dependency.  The V1 Brill tagger (``pos_tagger/treebank_brill_aubt.pickle``)
is not used anywhere here.  The spaCy pipeline is loaded lazily on the first
feature call so that importing this module stays fast.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Union

import numpy as np
import textstat
import torch
from scipy.stats import entropy as scipy_entropy

import spacy
from spacy.language import Language
from spacy.tokens import Doc


# ─────────────────────────── spaCy Setup ──────────────────────────────────────

def _av_sentencizer(doc: Doc) -> Doc:
    """Custom sentence boundary detector carried over from V1.

    Prevents false splits at abbreviations, coordinating conjunctions that
    continue a clause, and quotation marks.  Registered once under the name
    ``"av_sentencizer"`` to avoid conflicts with other pipelines in the same
    Python process.
    """
    SENT_BOUNDARY = {".", "!", "?"}
    NON_BOUNDARY_CONJ = {
        "and", "but", "or", "nor", "for", "so", "yet", "either", "neither"
    }
    OPEN_QUOTES = {'"', "\u201c", "\u2018"}
    CLOSE_QUOTES = {'"', "\u201d", "\u2019"}

    for i, token in enumerate(doc[:-1]):
        if token.text not in SENT_BOUNDARY:
            continue
        next_tok = doc[i + 1]
        prev_is_quote = i > 0 and doc[i - 1].text in CLOSE_QUOTES
        next_is_quote = next_tok.text in OPEN_QUOTES
        if prev_is_quote or next_is_quote:
            continue
        if next_tok.text.lower() in NON_BOUNDARY_CONJ:
            continue
        doc[i + 1].is_sent_start = True
    return doc


# Safe one-time registration — survive repeated module imports in notebooks
try:
    Language.component("av_sentencizer", func=_av_sentencizer)
except ValueError:
    pass  # already registered in this Python session

_nlp: Optional[spacy.Language] = None


def _get_nlp() -> spacy.Language:
    """Return the shared spaCy pipeline, loading it on first call."""
    global _nlp
    if _nlp is None:
        from config import SPACY_MODEL
        _nlp = spacy.load(SPACY_MODEL)
        if "sentencizer" in _nlp.pipe_names:
            _nlp.remove_pipe("sentencizer")
        if "av_sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("av_sentencizer", before="parser")
    return _nlp


def _as_doc(text_or_doc: Union[str, Doc]) -> Doc:
    """Parse a string into a spaCy Doc, or return the Doc unchanged."""
    if isinstance(text_or_doc, str):
        return _get_nlp()(text_or_doc)
    return text_or_doc


# ─────────────────────────── CEFR Helpers ─────────────────────────────────────

def load_cefr_dict(path: str) -> Dict[str, str]:
    """Load a CEFR vocabulary CSV into a ``{word -> level}`` lookup dict.

    The CSV must contain at minimum columns ``headword`` and ``CEFR``.  Both
    the surface form and a simple lower-case variant are stored so that lookup
    works without full lemmatization when the spaCy pipeline is unavailable.

    For full lemmatised lookup (as in V1) call with the spaCy pipeline already
    loaded; ``cefr_distribution`` uses the ``Doc`` token lemmas directly.
    """
    import pandas as pd
    df = pd.read_csv(path)
    result: Dict[str, str] = {}
    for word, level in zip(df["headword"], df["CEFR"]):
        result[str(word).lower()] = str(level)
    return result


# ─────────────────────────── Vocabulary Features ──────────────────────────────

def cefr_distribution(
    text_or_doc: Union[str, Doc],
    cefr_dict: Dict[str, str],
) -> Dict[str, float]:
    """CEFR vocabulary level distribution, excluding words not in the wordlist.

    Ported from V1 ``count_cefr_levels_exclude_unknown``.

    Returns proportions for A1, A2, B1, B2, C1, C2 summing to 1.0.
    Words not found in the CEFR dict are tallied but excluded from the
    denominator so proportions are computed over recognised vocabulary only.
    """
    doc = _as_doc(text_or_doc)
    counts: Dict[str, int] = {
        "A1": 0, "A2": 0, "B1": 0, "B2": 0, "C1": 0, "C2": 0, "Unknown": 0
    }
    for token in doc:
        if not token.is_alpha:
            continue
        lemma = token.lemma_.lower()
        surface = token.text.lower()
        level = cefr_dict.get(lemma, cefr_dict.get(surface, "Unknown"))
        counts[level] += 1

    known_total = sum(v for k, v in counts.items() if k != "Unknown")
    if known_total > 0:
        return {
            lvl: counts[lvl] / known_total
            for lvl in ("A1", "A2", "B1", "B2", "C1", "C2")
        }
    return {lvl: 0.0 for lvl in ("A1", "A2", "B1", "B2", "C1", "C2")}


def type_token_ratio(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Type-Token Ratio (TTR) and Corrected TTR (CTTR).

    TTR  = unique_types / total_tokens   (sensitive to text length)
    CTTR = unique_types / sqrt(2 * total_tokens)  (length-normalised)

    Only alphabetic, non-stop tokens are counted so punctuation and function
    words do not inflate the type set.
    """
    doc = _as_doc(text_or_doc)
    tokens = [t.lower_ for t in doc if t.is_alpha and not t.is_stop]
    n = len(tokens)
    if n == 0:
        return {"ttr": 0.0, "cttr": 0.0}
    types = len(set(tokens))
    ttr = types / n
    cttr = types / math.sqrt(2 * n)
    return {"ttr": ttr, "cttr": cttr}


def avg_word_length(text_or_doc: Union[str, Doc]) -> float:
    """Mean number of characters per alphabetic token."""
    doc = _as_doc(text_or_doc)
    lengths = [len(t.text) for t in doc if t.is_alpha]
    return float(np.mean(lengths)) if lengths else 0.0


def hapax_legomena_ratio(text_or_doc: Union[str, Doc]) -> float:
    """Hapax legomena ratio: words appearing once / words appearing twice.

    Ported from V1 (originally from writeprints-static by Ashenoy).
    Returns 0 when there are no dis-legomena (avoids divide-by-zero).
    """
    doc = _as_doc(text_or_doc)
    tokens = [t.lower_ for t in doc if t.is_alpha]
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    hapax = sum(1 for v in freq.values() if v == 1)
    dis = sum(1 for v in freq.values() if v == 2)
    return hapax / dis if dis > 0 else 0.0


def lexical_density(text_or_doc: Union[str, Doc]) -> float:
    """Lexical density: content words / total tokens.

    Content POS categories: NOUN, VERB (excluding AUX), ADJ, ADV.
    """
    CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
    doc = _as_doc(text_or_doc)
    total = sum(1 for t in doc if not t.is_space)
    content = sum(
        1 for t in doc if t.pos_ in CONTENT_POS and not t.is_space
    )
    return content / total if total > 0 else 0.0


# ─────────────────────────── Sentence-Level Features ─────────────────────────

def sentence_length_stats(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Sentence length statistics and histogram.

    Ported and extended from V1 ``calculate_sentence_length_stats``.
    V1 used regex splitting; V2 uses spaCy sentence segmentation for
    consistency with other features.  Token count per sentence excludes
    punctuation and whitespace-only tokens.

    Returns
    -------
    mean          : mean tokens per sentence
    std           : standard deviation
    entropy       : Shannon entropy (base-2) of the length frequency distribution
    bin_very_short: proportion of sentences with < 8 tokens
    bin_short     : proportion of sentences with 8–15 tokens
    bin_medium    : proportion of sentences with 15–25 tokens
    bin_long      : proportion of sentences with > 25 tokens
    """
    doc = _as_doc(text_or_doc)
    lengths: List[int] = [
        sum(1 for t in sent if not t.is_punct and not t.is_space)
        for sent in doc.sents
    ]
    lengths = [l for l in lengths if l > 0]
    if not lengths:
        return {
            "mean": 0.0, "std": 0.0, "entropy": 0.0,
            "bin_very_short": 0.0, "bin_short": 0.0,
            "bin_medium": 0.0, "bin_long": 0.0,
        }

    arr = np.array(lengths, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())

    freq = Counter(lengths)
    probs = np.array(list(freq.values()), dtype=float) / len(lengths)
    ent = float(scipy_entropy(probs, base=2))

    n = len(lengths)
    bins = {
        "bin_very_short": sum(1 for l in lengths if l < 8) / n,
        "bin_short":      sum(1 for l in lengths if 8 <= l <= 15) / n,
        "bin_medium":     sum(1 for l in lengths if 15 < l <= 25) / n,
        "bin_long":       sum(1 for l in lengths if l > 25) / n,
    }
    return {"mean": mean, "std": std, "entropy": ent, **bins}


def sentence_type_distribution(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Normalised distribution over sentence structural types.

    Ported directly from V1 ``calculate_sentence_type``.

    Types: Simple, Compound, Complex, Compound-Complex, Fragment, Unclassified.
    Classification uses the spaCy dependency parse (ROOT, conj, advcl, etc.).
    """
    doc = _as_doc(text_or_doc)
    counts: Dict[str, int] = {
        "Simple": 0, "Compound": 0, "Complex": 0,
        "Compound-Complex": 0, "Fragment": 0, "Unclassified": 0,
    }

    for sent in doc.sents:
        if not sent.text.strip() or not re.search(r"[A-Za-z0-9]", sent.text):
            counts["Unclassified"] += 1
            continue

        finite_verbs = [
            t for t in sent
            if t.pos_ in {"VERB", "AUX"}
            and t.tag_ in {"VBD", "VBP", "VBZ", "VBN", "VBG", "MD"}
        ]
        if not finite_verbs:
            counts["Fragment"] += 1
            continue

        independent = [
            t for t in sent
            if t.dep_ == "ROOT"
            or (t.dep_ == "conj" and t.head.dep_ == "ROOT")
        ]
        n_ind = len(independent)

        dep_clause_heads = {
            t.head for t in sent
            if t.dep_ in {"advcl", "ccomp", "relcl", "csubj", "csubjpass"}
        }
        n_dep = len(dep_clause_heads)

        coord_conj = [t for t in sent if t.dep_ == "cc"]

        if n_ind == 1 and n_dep == 0:
            counts["Simple"] += 1
        elif n_ind > 1 and n_dep == 0:
            counts["Compound" if coord_conj else "Unclassified"] += 1
        elif n_ind == 1 and n_dep >= 1:
            counts["Complex"] += 1
        elif n_ind > 1 and n_dep >= 1:
            counts["Compound-Complex"] += 1
        else:
            verbs = [t for t in sent if t.pos_ == "VERB"]
            counts["Fragment" if verbs and n_ind == 0 else "Unclassified"] += 1

    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def sentence_structure_variation(text_or_doc: Union[str, Doc]) -> float:
    """Ratio of unique syntactic argument templates to total sentences.

    A "syntactic template" for a sentence is the sorted tuple of dependency
    labels of the ROOT token's direct children.  This captures predicate-
    argument structure without being topic-sensitive.

    Score = unique_templates / total_sentences  in [0, 1].
    Higher = more structurally diverse writing.
    """
    doc = _as_doc(text_or_doc)
    templates: List[tuple] = []
    for sent in doc.sents:
        root_tokens = [t for t in sent if t.dep_ == "ROOT"]
        if not root_tokens:
            continue
        root = root_tokens[0]
        child_deps = tuple(sorted(t.dep_ for t in root.children))
        templates.append(child_deps)
    if not templates:
        return 0.0
    return len(set(templates)) / len(templates)


def subordinate_clause_frequency(text_or_doc: Union[str, Doc]) -> float:
    """Mean number of subordinate clauses per sentence.

    Counts dependency labels: advcl (adverbial clause), ccomp (clausal
    complement), relcl (relative clause), csubj (clausal subject).
    """
    doc = _as_doc(text_or_doc)
    SUB_DEPS = {"advcl", "ccomp", "relcl", "csubj"}
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    total_sub = sum(1 for t in doc if t.dep_ in SUB_DEPS)
    return total_sub / len(sentences)


# ─────────────────────────── Syntactic / POS Features ─────────────────────────

def pos_unigram_distribution(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Normalised ratio of each major POS category over all non-space tokens.

    Categories reported: NOUN, VERB, AUX, ADJ, ADV, CCONJ, SCONJ, PRON, DET.
    """
    CATEGORIES = ("NOUN", "VERB", "AUX", "ADJ", "ADV", "CCONJ", "SCONJ", "PRON", "DET")
    doc = _as_doc(text_or_doc)
    counts: Dict[str, int] = Counter(
        t.pos_ for t in doc if not t.is_space and not t.is_punct
    )
    total = sum(counts.values())
    if total == 0:
        return {cat: 0.0 for cat in CATEGORIES}
    return {cat: counts.get(cat, 0) / total for cat in CATEGORIES}


def pos_bigram_patterns(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Frequency of linguistically meaningful adjacent POS-tag bigrams.

    Only bigrams between non-punctuation, non-space tokens are considered.
    Counts are normalised by the total number of bigrams in the document.

    Bigrams tracked (rate per total bigrams):
        ADJ_NOUN, NOUN_VERB, VERB_NOUN, VERB_ADV, ADV_ADJ, NOUN_NOUN, DET_NOUN
    """
    TRACKED = {
        ("ADJ", "NOUN"): "ADJ_NOUN",
        ("NOUN", "VERB"): "NOUN_VERB",
        ("VERB", "NOUN"): "VERB_NOUN",
        ("VERB", "ADV"):  "VERB_ADV",
        ("ADV",  "ADJ"):  "ADV_ADJ",
        ("NOUN", "NOUN"): "NOUN_NOUN",
        ("DET",  "NOUN"): "DET_NOUN",
    }
    doc = _as_doc(text_or_doc)
    pos_seq = [t.pos_ for t in doc if not t.is_space and not t.is_punct]
    if len(pos_seq) < 2:
        return {v: 0.0 for v in TRACKED.values()}

    total_bigrams = len(pos_seq) - 1
    counts: Dict[str, int] = {v: 0 for v in TRACKED.values()}
    for a, b in zip(pos_seq, pos_seq[1:]):
        key = (a, b)
        if key in TRACKED:
            counts[TRACKED[key]] += 1

    return {name: count / total_bigrams for name, count in counts.items()}


def adverbial_placement_distribution(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Distribution of adverb placements within sentences.

    Ported from V1 ``adverbial_placement``; returns only the proportions dict
    (V1 also returned raw positions and counts, which belong in debugging only).

    Positions: sentence-initial, preverbal, postverbal, sentence-final, other.
    Proportions sum to 1 over all adverbs in the document.
    """
    doc = _as_doc(text_or_doc)
    placement_counts: Dict[str, int] = {
        "sentence_initial": 0,
        "preverbal":        0,
        "postverbal":       0,
        "sentence_final":   0,
        "other":            0,
    }
    num_adverbs = 0

    for sent in doc.sents:
        sent_tokens = list(sent)
        sent_len = len(sent_tokens)
        non_adverbs = [t for t in sent if t.pos_ != "ADV" and not t.is_punct]

        if not non_adverbs:
            for t in sent_tokens:
                if t.pos_ == "ADV":
                    placement_counts["other"] += 1
                    num_adverbs += 1
            continue

        first_non_adv = non_adverbs[0]
        last_non_adv = non_adverbs[-1]

        for idx, token in enumerate(sent_tokens):
            if token.pos_ != "ADV":
                continue
            num_adverbs += 1

            is_sent_final = (
                idx == sent_len - 1
                or (idx + 1 < sent_len and sent_tokens[idx + 1].is_punct)
            )

            try:
                first_non_adv_idx = sent_tokens.index(first_non_adv)
                last_non_adv_idx = sent_tokens.index(last_non_adv)
            except ValueError:
                placement_counts["other"] += 1
                continue

            if idx < first_non_adv_idx:
                placement_counts["sentence_initial"] += 1
            elif token.head in sent_tokens:
                head_idx = sent_tokens.index(token.head)
                if idx < head_idx:
                    placement_counts["preverbal"] += 1
                elif is_sent_final:
                    placement_counts["sentence_final"] += 1
                elif idx < last_non_adv_idx:
                    placement_counts["postverbal"] += 1
                else:
                    placement_counts["other"] += 1
            else:
                placement_counts[
                    "sentence_final" if is_sent_final else "other"
                ] += 1

    if num_adverbs == 0:
        return {k: 0.0 for k in placement_counts}
    return {k: v / num_adverbs for k, v in placement_counts.items()}


def passive_voice_frequency(text_or_doc: Union[str, Doc]) -> float:
    """Passive voice ratio: passive-subject tokens / total sentences.

    Detects passive constructions via dependency labels ``nsubjpass`` and
    ``nsubj:pass`` (the latter used in Universal Dependencies / newer spaCy
    transformer models).
    """
    PASSIVE_DEPS = {"nsubjpass", "nsubj:pass"}
    doc = _as_doc(text_or_doc)
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    passive_count = sum(1 for t in doc if t.dep_ in PASSIVE_DEPS)
    return passive_count / len(sentences)


# ─────────────────────────── Discourse / Readability Features ─────────────────

def readability_metrics(text: str) -> Dict[str, float]:
    """Three readability indices using the ``textstat`` library.

    Ported and extended from V1 ``readability_metric`` (which returned only
    Flesch reading ease). V2 adds Gunning Fog and Coleman-Liau as specified.

    Returns
    -------
    flesch_kincaid_grade : Flesch-Kincaid Grade Level (higher = harder)
    gunning_fog          : Gunning Fog Index
    coleman_liau         : Coleman-Liau Index
    """
    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog":          textstat.gunning_fog(text),
        "coleman_liau":         textstat.coleman_liau_index(text),
    }


def calculate_perplexity(
    text: str,
    model,
    tokenizer,
    device,
    max_length: int = 1024,
    overlap: int = 32,
) -> float:
    """GPT-2 perplexity of a document, averaged over overlapping chunks.

    Ported directly from V1 ``calculate_pxl_optimized``.

    Long texts are split into sliding windows of ``max_length`` tokens with
    ``overlap`` tokens of context carried forward from the previous chunk.
    Returns the mean perplexity across all chunks, or 0.0 for empty input.

    Parameters
    ----------
    model / tokenizer / device : pre-loaded HuggingFace GPT-2 objects.
        Call ``pipeline_model_setup.setup_gpt2()`` to obtain these.
    """
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    if len(tokens) == 0:
        return 0.0

    n_chunks = (
        len(tokens) // (max_length - overlap)
        + (1 if len(tokens) % (max_length - overlap) > 0 else 0)
    )
    chunks: List[torch.Tensor] = []
    for i in range(n_chunks):
        start = i * (max_length - overlap)
        end = min((i + 1) * max_length, len(tokens))
        chunk = tokens[start:end]
        if len(chunk) < max_length and i != 0:
            chunk = torch.cat([chunks[-1][-overlap:], chunk], dim=0)
        chunks.append(chunk)

    perplexities: List[float] = []
    for chunk in chunks:
        inputs = chunk.unsqueeze(0).to(device)
        attention_mask = (inputs != tokenizer.pad_token_id).float()
        with torch.no_grad():
            loss = model(inputs, attention_mask=attention_mask, labels=inputs).loss
        perplexities.append(torch.exp(loss).item())

    return sum(perplexities) / len(perplexities) if perplexities else 0.0


# Discourse connective lexicon (ordered general → specific to minimise
# double-counting for multi-word expressions).
_DISCOURSE_CONNECTIVES: Dict[str, List[str]] = {
    "contrastive": [
        "however", "nevertheless", "nonetheless", "on the other hand",
        "in contrast", "conversely", "whereas", "while", "although",
        "though", "despite", "in spite of", "yet", "still",
    ],
    "additive": [
        "furthermore", "moreover", "in addition", "additionally",
        "besides", "what is more", "not only", "as well",
    ],
    "causal": [
        "therefore", "thus", "hence", "consequently", "as a result",
        "for this reason", "owing to", "accordingly",
    ],
    "temporal": [
        "subsequently", "meanwhile", "previously", "in conclusion",
        "in summary", "to summarize", "to conclude", "initially", "finally",
    ],
}

# Flattened set of all connective strings for fast presence checks
_ALL_CONNECTIVES: List[str] = [
    c for group in _DISCOURSE_CONNECTIVES.values() for c in group
]


def discourse_connective_frequency(
    text_or_doc: Union[str, Doc],
) -> Dict[str, float]:
    """Frequency of discourse connectives, grouped by rhetorical function.

    Multi-word connectives (e.g. 'on the other hand') are found via substring
    search on the lower-cased raw text to avoid tokenisation mismatches.
    Single-word connectives are matched at word boundaries.

    Returns rates per 100 words for the total and each rhetorical group:
        total_per_100, contrastive_per_100, additive_per_100,
        causal_per_100, temporal_per_100
    """
    # Resolve text and word count
    if isinstance(text_or_doc, Doc):
        raw_text = text_or_doc.text.lower()
        word_count = sum(1 for t in text_or_doc if t.is_alpha)
    else:
        raw_text = text_or_doc.lower()
        word_count = len(re.findall(r"\b[a-z]+\b", raw_text))

    if word_count == 0:
        return {
            "total_per_100": 0.0,
            "contrastive_per_100": 0.0,
            "additive_per_100": 0.0,
            "causal_per_100": 0.0,
            "temporal_per_100": 0.0,
        }

    group_counts: Dict[str, int] = {g: 0 for g in _DISCOURSE_CONNECTIVES}
    for group, connectives in _DISCOURSE_CONNECTIVES.items():
        for conn in connectives:
            if " " in conn:
                # Multi-word: simple substring count with word-boundary anchors
                pattern = r"\b" + re.escape(conn) + r"\b"
                group_counts[group] += len(re.findall(pattern, raw_text))
            else:
                group_counts[group] += len(
                    re.findall(r"\b" + re.escape(conn) + r"\b", raw_text)
                )

    scale = 100.0 / word_count
    total = sum(group_counts.values())
    return {
        "total_per_100":        total * scale,
        "contrastive_per_100":  group_counts["contrastive"] * scale,
        "additive_per_100":     group_counts["additive"] * scale,
        "causal_per_100":       group_counts["causal"] * scale,
        "temporal_per_100":     group_counts["temporal"] * scale,
    }


# ─────────────────────────── Punctuation / Style Features ─────────────────────

def punctuation_style(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Punctuation usage and paragraph-level style features.

    New in V2 — not present in V1.  All rates are normalised to be comparable
    across documents of different lengths.

    Returns
    -------
    comma_per_sentence      : mean comma count per sentence
    semicolon_colon_rate    : (semicolons + colons) per 100 tokens
    exclamation_ratio       : exclamation marks / total sentence-ending marks
    question_ratio          : question marks / total sentence-ending marks
    avg_paragraph_length    : mean word count per paragraph (paragraphs are
                              separated by blank lines or ``\\n\\n``)
    """
    doc = _as_doc(text_or_doc)
    sentences = list(doc.sents)
    n_sents = len(sentences)
    n_tokens = sum(1 for t in doc if not t.is_space)

    # Per-sentence comma count
    commas = sum(1 for t in doc if t.text == ",")
    comma_per_sentence = commas / n_sents if n_sents > 0 else 0.0

    # Semicolons and colons rate
    semi_colon_count = sum(1 for t in doc if t.text in {";", ":"})
    semicolon_colon_rate = (semi_colon_count / n_tokens * 100) if n_tokens > 0 else 0.0

    # Exclamation / question ratios over terminal punctuation
    excl = sum(1 for t in doc if t.text == "!")
    quest = sum(1 for t in doc if t.text == "?")
    period = sum(1 for t in doc if t.text == ".")
    terminal_total = excl + quest + period
    exclamation_ratio = excl / terminal_total if terminal_total > 0 else 0.0
    question_ratio   = quest / terminal_total if terminal_total > 0 else 0.0

    # Average paragraph length in words
    raw = doc.text
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if paragraphs:
        para_word_counts = [
            len(re.findall(r"\b[A-Za-z]+\b", para)) for para in paragraphs
        ]
        avg_para_len = float(np.mean(para_word_counts))
    else:
        avg_para_len = float(
            len(re.findall(r"\b[A-Za-z]+\b", raw))
        )  # whole text = one paragraph

    return {
        "comma_per_sentence":   comma_per_sentence,
        "semicolon_colon_rate": semicolon_colon_rate,
        "exclamation_ratio":    exclamation_ratio,
        "question_ratio":       question_ratio,
        "avg_paragraph_length": avg_para_len,
    }


# ─────────────────────────── Topic-Independent Stylistic Features ──────────────

def sentence_opening_patterns(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Ratio of sentences opening with subordinate conjunction, adverb, pronoun, article.

    Topic-independent: how an author typically begins sentences.  For each sentence,
    the first non-punctuation, non-space token is classified by POS.

    Returns
    -------
    subordinate_conj : proportion of sentences starting with SCONJ
    adverb           : proportion starting with ADV
    pronoun          : proportion starting with PRON
    article          : proportion starting with DET (a/an/the)
    """
    doc = _as_doc(text_or_doc)
    counts: Dict[str, int] = {
        "subordinate_conj": 0,
        "adverb":           0,
        "pronoun":          0,
        "article":          0,
    }
    n_sents = 0
    for sent in doc.sents:
        first = None
        for t in sent:
            if not t.is_punct and not t.is_space:
                first = t
                break
        if first is None:
            continue
        n_sents += 1
        if first.pos_ == "SCONJ":
            counts["subordinate_conj"] += 1
        elif first.pos_ == "ADV":
            counts["adverb"] += 1
        elif first.pos_ == "PRON":
            counts["pronoun"] += 1
        elif first.pos_ == "DET":
            counts["article"] += 1
    if n_sents == 0:
        return {k: 0.0 for k in counts}
    return {k: v / n_sents for k, v in counts.items()}


def coordination_subordination_ratio(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Coordinating vs. subordinating conjunctions per sentence.

    Coordinating (CCONJ): and, but, or, so, yet, for, nor.
    Subordinating (SCONJ): although, because, when, if, etc.
    Ratio = coord_per_sent / (subord_per_sent + epsilon) to avoid division by zero.

    Returns
    -------
    coord_per_sent  : coordinating conjunctions per sentence
    subord_per_sent : subordinating conjunctions per sentence
    ratio           : coord / subord (capped at 100.0 when subord is near zero)
    """
    doc = _as_doc(text_or_doc)
    n_sents = max(1, len(list(doc.sents)))
    coord = sum(1 for t in doc if t.pos_ == "CCONJ")
    subord = sum(1 for t in doc if t.pos_ == "SCONJ")
    coord_per_sent = coord / n_sents
    subord_per_sent = subord / n_sents
    ratio = coord_per_sent / (subord_per_sent + 1e-6)
    ratio = min(ratio, 100.0)  # cap for near-zero subord
    return {
        "coord_per_sent":  coord_per_sent,
        "subord_per_sent": subord_per_sent,
        "ratio":           ratio,
    }


def hedging_language_frequency(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Frequency per 100 words of hedging expressions.

    Hedging words: perhaps, might, seems, appears, possibly, likely, suggest,
    indicate, tend, appear.  Matched on lemma to capture inflected forms
    (e.g. suggested, suggested, suggesting → suggest).

    Returns
    -------
    hedging_per_100 : count of hedging lemmas per 100 words
    """
    HEDGING_LEMMAS = {
        "perhaps", "might", "seem", "appear", "possibly", "likely",
        "suggest", "indicate", "tend",
    }
    doc = _as_doc(text_or_doc)
    word_count = sum(1 for t in doc if t.is_alpha)
    if word_count == 0:
        return {"per_100": 0.0}
    count = sum(1 for t in doc if t.is_alpha and t.lemma_.lower() in HEDGING_LEMMAS)
    return {"per_100": count * 100.0 / word_count}


def contraction_rate(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Contractions per 100 words (e.g. don't, it's, we're).

    Detects tokens containing an apostrophe (n't, 's, 're, 've, 'd, 'll, 'm,
    or full forms like don't).  Topic-independent stylistic habit.
    """
    doc = _as_doc(text_or_doc)
    word_count = sum(1 for t in doc if t.is_alpha)
    if word_count == 0:
        return {"contraction_per_100": 0.0}
    count = sum(1 for t in doc if "'" in t.text and re.search(r"[a-zA-Z]", t.text))
    return {"contraction_per_100": count * 100.0 / word_count}


def pronoun_distribution(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """First-person singular pronouns (I/me/my/mine/myself) as fraction of all pronouns.

    Topic-independent: reflects authorial voice and self-reference habits.
    """
    FIRST_PERSON_LEMMAS = {"i", "me", "my", "mine", "myself"}
    doc = _as_doc(text_or_doc)
    pron_tokens = [t for t in doc if t.pos_ == "PRON"]
    total_pron = len(pron_tokens)
    if total_pron == 0:
        return {"first_person_singular_ratio": 0.0}
    first_sing = sum(
        1 for t in pron_tokens
        if t.lemma_.lower() in FIRST_PERSON_LEMMAS
    )
    return {"first_person_singular_ratio": first_sing / total_pron}


def avg_dependency_depth(text_or_doc: Union[str, Doc]) -> Dict[str, float]:
    """Average dependency parse tree depth per sentence.

    For each token, depth = number of steps from token to root.  Root has depth 0.
    Returns mean depth over all tokens, and mean over sentences (sentence-level mean).
    """
    doc = _as_doc(text_or_doc)
    all_depths: List[int] = []
    per_sent_means: List[float] = []
    for sent in doc.sents:
        tok_depths: List[int] = []
        for token in sent:
            if token.is_space or token.is_punct:
                continue
            d = 0
            head = token.head
            while head != head.head:
                d += 1
                head = head.head
            tok_depths.append(d)
            all_depths.append(d)
        if tok_depths:
            per_sent_means.append(float(np.mean(tok_depths)))
    mean_depth = float(np.mean(all_depths)) if all_depths else 0.0
    mean_per_sent = float(np.mean(per_sent_means)) if per_sent_means else 0.0
    return {"mean_depth": mean_depth, "mean_per_sent": mean_per_sent}


# ─────────────────────────── Aggregate Extractor ──────────────────────────────

def extract_all(
    text: str,
    cefr_dict: Optional[Dict[str, str]] = None,
    perplexity_model=None,
    perplexity_tokenizer=None,
    perplexity_device=None,
) -> Dict[str, float]:
    """Extract ALL handcrafted features for a single document.

    Parses the spaCy ``Doc`` exactly once and passes it to every feature
    function to avoid repeated full parses.  Returns a single flat dict that
    ``features/evidence_packet.py`` and ``retrieval/indexer.py`` consume.

    Parameters
    ----------
    text                : Raw (or pre-cleaned) document string.
    cefr_dict           : Output of ``load_cefr_dict()``. If ``None``, CEFR
                          features are omitted from the output.
    perplexity_model /
    perplexity_tokenizer /
    perplexity_device   : Pre-loaded GPT-2 objects.  If any is ``None``,
                          the perplexity feature is omitted.
    """
    doc = _get_nlp()(text)
    features: Dict[str, float] = {}

    # ── Vocabulary ────────────────────────────────────────────────────────────
    if cefr_dict is not None:
        for k, v in cefr_distribution(doc, cefr_dict).items():
            features[f"cefr_{k}"] = v
    for k, v in type_token_ratio(doc).items():
        features[f"vocab_{k}"] = v
    features["vocab_avg_word_length"]      = avg_word_length(doc)
    features["vocab_hapax_legomena_ratio"] = hapax_legomena_ratio(doc)
    features["vocab_lexical_density"]      = lexical_density(doc)

    # ── Sentence-Level ────────────────────────────────────────────────────────
    for k, v in sentence_length_stats(doc).items():
        features[f"sent_{k}"] = v
    for k, v in sentence_type_distribution(doc).items():
        features[f"sent_type_{k}"] = v
    features["sent_structure_variation"] = sentence_structure_variation(doc)
    features["sent_subordinate_freq"]    = subordinate_clause_frequency(doc)

    # ── Syntactic / POS ───────────────────────────────────────────────────────
    for k, v in pos_unigram_distribution(doc).items():
        features[f"pos_{k}"] = v
    for k, v in pos_bigram_patterns(doc).items():
        features[f"bigram_{k}"] = v
    for k, v in adverbial_placement_distribution(doc).items():
        features[f"adv_{k}"] = v
    features["passive_voice_freq"] = passive_voice_frequency(doc)

    # ── Discourse / Readability ───────────────────────────────────────────────
    for k, v in readability_metrics(text).items():
        features[f"readability_{k}"] = v
    if (perplexity_model is not None
            and perplexity_tokenizer is not None
            and perplexity_device is not None):
        features["perplexity"] = calculate_perplexity(
            text, perplexity_model, perplexity_tokenizer, perplexity_device
        )
    for k, v in discourse_connective_frequency(doc).items():
        features[f"discourse_{k}"] = v

    # ── Punctuation / Style ───────────────────────────────────────────────────
    for k, v in punctuation_style(doc).items():
        features[f"punct_{k}"] = v

    # ── Topic-independent stylistic (sentence openings, coordination, etc.) ─
    for k, v in sentence_opening_patterns(doc).items():
        features[f"sent_opening_{k}"] = v
    for k, v in coordination_subordination_ratio(doc).items():
        features[f"coord_subord_{k}"] = v
    for k, v in hedging_language_frequency(doc).items():
        features[f"hedging_{k}"] = v
    for k, v in contraction_rate(doc).items():
        features[k] = v  # e.g. contraction_per_100
    for k, v in pronoun_distribution(doc).items():
        features[f"pronoun_{k}"] = v
    for k, v in avg_dependency_depth(doc).items():
        features[f"dep_depth_{k}"] = v

    return features
