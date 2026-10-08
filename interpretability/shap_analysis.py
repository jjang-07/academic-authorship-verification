"""SHAP-based interpretability for the V2 authorship verification pipeline.

Architecture
------------
Alongside the LLM debate pipeline, a lightweight GradientBoostingClassifier is
trained on pairwise |f_a − f_b| feature difference vectors — exactly the same
representation used by the V1 baseline.  SHAP TreeExplainer then attributes each
prediction to individual features, giving a quantitative, reproducible
interpretability layer that is fully independent of LLM reasoning.

This mirrors the V1 SHAP pipeline but exposes it as a clean module-level API
so it can be called from ``trace_parser.py`` and the ablation runner.

Public API
----------
    # One-time training (or load from cache):
    model = train_shap_model(pairs, cefr_dict=cefr_dict)
    save_shap_model(model)          # persists to config.SHAP_MODEL_PATH

    # Per-pair inference:
    result = explain_pair(features_a, features_b, model)
    # result.top_features → [ShapFeature(name, shap_value, direction, delta), ...]

    # Convenience: train-then-explain in one call:
    result = explain_pair_from_texts(text_a, text_b, model, cefr_dict=cefr_dict)

Data flow
---------
1. extract_all(text_a)  → features_a  (dict[str, float])
2. extract_all(text_b)  → features_b  (dict[str, float])
3. pair_vector = [|f_a[k] - f_b[k]|  for k in sorted common keys]
4. model.predict_proba(pair_vector)   → same-author probability
5. shap.TreeExplainer(model).shap_values(pair_vector) → per-feature attributions
6. top 5 attribution magnitudes       → ShapResult.top_features

Persistence
-----------
The fitted model (sklearn Pipeline: StandardScaler + GBC) is serialised with
``joblib`` to avoid re-training on every run.  The path is read from
``config.SHAP_MODEL_PATH`` (default: ``models/shap_gbc.joblib``).
Feature key ordering is saved alongside the model so the same vector layout is
reproduced at inference time regardless of dict ordering.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

# Feature human-readable descriptions for report formatting
_FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # Vocabulary
    "cefr_A1":                    "fraction of very basic (A1) words",
    "cefr_A2":                    "fraction of elementary (A2) words",
    "cefr_B1":                    "fraction of intermediate (B1) words",
    "cefr_B2":                    "fraction of upper-intermediate (B2) words",
    "cefr_C1":                    "fraction of advanced (C1) words",
    "cefr_C2":                    "fraction of proficiency-level (C2) words",
    "vocab_ttr":                  "type-token ratio (vocabulary diversity)",
    "vocab_avg_word_length":      "average word length in characters",
    "vocab_hapax_legomena_ratio": "proportion of words used only once (rare vocabulary)",
    "vocab_lexical_density":      "lexical density (content vs. function words)",
    # Sentence structure
    "sent_mean":                  "mean sentence length in tokens",
    "sent_std":                   "standard deviation of sentence length",
    "sent_entropy":               "entropy of sentence length distribution",
    "sent_structure_variation":   "sentence structure variety",
    "sent_subordinate_freq":      "frequency of subordinate clauses",
    # Sentence opening patterns
    "sent_opening_subordinate_conj": "rate of sentences opening with a subordinate conjunction",
    "sent_opening_adverb":           "rate of sentences opening with an adverb",
    "sent_opening_pronoun":          "rate of sentences opening with a pronoun",
    "sent_opening_article":          "rate of sentences opening with an article",
    # Syntactic / POS
    "pos_NOUN":                   "proportion of nouns",
    "pos_VERB":                   "proportion of verbs",
    "pos_ADJ":                    "proportion of adjectives",
    "pos_ADV":                    "proportion of adverbs",
    "pos_PRON":                   "proportion of pronouns",
    "passive_voice_freq":         "passive voice usage frequency",
    "adv_sentence_initial":       "rate of sentence-initial adverbs",
    "adv_sentence_medial":        "rate of sentence-medial adverbs",
    "adv_sentence_final":         "rate of sentence-final adverbs",
    # Coordination / subordination
    "coord_subord_coord_per_sent":  "coordinating conjunctions per sentence",
    "coord_subord_subord_per_sent": "subordinating conjunctions per sentence",
    # Hedging / contraction / pronouns
    "hedging_per_100":            "hedging language frequency per 100 words",
    "contraction_per_100":        "contraction rate per 100 words",
    "pronoun_first_person_ratio": "first-person pronoun share of all pronouns",
    # Dependency depth
    "dep_depth_mean":             "mean dependency parse tree depth",
    "dep_depth_max":              "maximum dependency parse tree depth",
    # Discourse
    "discourse_total_per_100":    "total discourse connective frequency",
    "discourse_causal_per_100":   "causal connective frequency (because, so, therefore)",
    "discourse_concessive_per_100": "concessive connective frequency (but, however, although)",
    # Readability
    "readability_flesch_kincaid_grade": "Flesch-Kincaid reading grade level",
    "readability_flesch_reading_ease":  "Flesch reading ease score",
    "readability_gunning_fog":          "Gunning Fog readability index",
    # Punctuation
    "punct_comma_per_sentence":     "commas per sentence",
    "punct_semicolon_per_sentence":  "semicolons per sentence",
    "punct_exclamation_per_sentence":"exclamation marks per sentence",
    "punct_question_per_sentence":   "question marks per sentence",
    "punct_avg_paragraph_length":    "average paragraph length in tokens",
}


def _describe_feature(key: str) -> str:
    """Return a human-readable description for a feature key."""
    for prefix, desc in _FEATURE_DESCRIPTIONS.items():
        if key == prefix or key.startswith(prefix):
            return desc
    # Fallback: prettify the key name
    return key.replace("_", " ")


# ──────────────────────────────── Data Contracts ──────────────────────────────

@dataclass
class ShapFeature:
    """A single feature's SHAP attribution for one pair.

    Fields
    ------
    name        : Feature key (e.g. ``sent_mean``).
    description : Human-readable description of what this feature measures.
    shap_value  : Signed SHAP value; positive pushes toward DIFFERENT_AUTHOR,
                  negative toward SAME_AUTHOR.
    direction   : "DIFFERENT_AUTHOR" or "SAME_AUTHOR" — which verdict this
                  feature pushed the model toward.
    delta       : The raw |f_a − f_b| value (before SHAP weighting).
    """
    name: str
    description: str
    shap_value: float
    direction: str
    delta: float


@dataclass
class ShapResult:
    """Complete SHAP attribution for one text pair.

    Fields
    ------
    pred_score      : Model's same-author probability (0 = different, 1 = same).
    pred_verdict    : "SAME_AUTHOR" if pred_score >= 0.5, else "DIFFERENT_AUTHOR".
    top_features    : Top 5 features by |shap_value|, descending.
    all_shap_values : Full feature-name → shap_value dict for further analysis.
    feature_vector  : The |f_a − f_b| pair vector passed to the model.
    feature_keys    : Ordered list of feature key names matching feature_vector.
    """
    pred_score: float
    pred_verdict: str
    top_features: List[ShapFeature]
    all_shap_values: Dict[str, float]
    feature_vector: List[float]
    feature_keys: List[str]


@dataclass
class ShapModel:
    """Fitted SHAP model bundle.

    Fields
    ------
    pipeline     : Fitted sklearn Pipeline (StandardScaler + GBC).
    explainer    : shap.TreeExplainer bound to the fitted GBC.
    feature_keys : Ordered list of feature keys used at training time.
    train_metrics: Optional dict of training-set evaluation metrics.
    """
    pipeline: object
    explainer: object
    feature_keys: List[str]
    train_metrics: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────── Feature Matrix ─────────────────────────────

def build_pair_matrix(
    pairs,                          # List[TextPair]
    cefr_dict: Optional[Dict] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Build an |f_a − f_b| feature matrix from a list of TextPair objects.

    Returns
    -------
    X           : (n_pairs × n_features) float array.
    true_labels : list[int]  (1 = same author, 0 = different).
    feature_keys: sorted list of feature names defining the column order.
    """
    from features.handcrafted import extract_all

    feat_pairs: List[Tuple[Dict, Dict]] = []
    for i, pair in enumerate(pairs):
        if verbose and (i % 50 == 0):
            print(f"  Extracting features: {i}/{len(pairs)} …")
        fa = extract_all(pair.text_a, cefr_dict=cefr_dict)
        fb = extract_all(pair.text_b, cefr_dict=cefr_dict)
        feat_pairs.append((fa, fb))

    # Common numeric keys across all pairs
    key_sets = [
        {k for k, v in fa.items() if isinstance(v, (int, float))}
        & {k for k, v in fb.items() if isinstance(v, (int, float))}
        for fa, fb in feat_pairs
    ]
    feature_keys = sorted(set.intersection(*key_sets)) if key_sets else []

    rows: List[List[float]] = []
    labels: List[int] = []
    for (fa, fb), pair in zip(feat_pairs, pairs):
        row = [abs(float(fa.get(k, 0.0)) - float(fb.get(k, 0.0))) for k in feature_keys]
        rows.append(row)
        labels.append(int(pair.label))

    X = np.array(rows, dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    return X, labels, feature_keys


def pair_vector(
    features_a: Dict[str, float],
    features_b: Dict[str, float],
    feature_keys: List[str],
) -> np.ndarray:
    """Construct a single |f_a − f_b| row vector given a fixed key ordering."""
    row = [abs(float(features_a.get(k, 0.0)) - float(features_b.get(k, 0.0)))
           for k in feature_keys]
    return np.array(row, dtype=float).reshape(1, -1)


# ──────────────────────────────── Training ────────────────────────────────────

def train_shap_model(
    pairs,                          # List[TextPair]
    cefr_dict: Optional[Dict] = None,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> ShapModel:
    """Train a GradientBoostingClassifier on pairwise feature difference vectors.

    A StandardScaler is fitted inside a Pipeline to ensure the same
    preprocessing is applied at inference time.  SHAP's TreeExplainer is
    then bound to the fitted GBC (the second pipeline step) — not the whole
    pipeline — because TreeExplainer requires the raw tree object.

    Parameters
    ----------
    pairs        : TextPair list to train on.
    cefr_dict    : Optional CEFR wordlist for vocabulary features.
    n_estimators : GBC hyperparameter.
    max_depth    : GBC hyperparameter.
    learning_rate: GBC hyperparameter.
    seed         : Random seed for reproducibility.
    verbose      : Print progress messages.

    Returns
    -------
    ShapModel
    """
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from evaluation.metrics import evaluate_all

    if verbose:
        print(f"[SHAP] Building feature matrix for {len(pairs)} pairs …")
    X, labels, feature_keys = build_pair_matrix(pairs, cefr_dict=cefr_dict, verbose=verbose)
    y = np.array(labels)

    if verbose:
        print(f"[SHAP] Feature matrix: {X.shape}  ({len(feature_keys)} features)")
        print(f"[SHAP] Class balance: {y.sum()} same-author / {(1-y).sum()} different-author")

    scaler = StandardScaler()
    gbc    = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
    )
    pipeline = Pipeline([("scaler", scaler), ("gbc", gbc)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X, y)

    # Evaluate on training set (indicative only — use ablation for held-out metrics)
    pred_scores = pipeline.predict_proba(X)[:, 1].tolist()
    train_metrics = evaluate_all(labels, pred_scores)
    if verbose:
        print(f"[SHAP] Training metrics (in-sample): {train_metrics}")

    # TreeExplainer needs the GBC estimator, not the pipeline
    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(gbc)

    if verbose:
        print("[SHAP] Model and explainer ready.")

    return ShapModel(
        pipeline=pipeline,
        explainer=explainer,
        feature_keys=feature_keys,
        train_metrics=train_metrics,
    )


# ──────────────────────────────── Persistence ─────────────────────────────────

def _default_model_path() -> Path:
    raw = getattr(config, "SHAP_MODEL_PATH", "models/shap_gbc.joblib")
    return Path(raw)


def save_shap_model(model: ShapModel, path: Optional[str] = None) -> str:
    """Serialise the model bundle to disk with joblib.

    Parameters
    ----------
    model : Trained ShapModel.
    path  : Override save path.  Defaults to ``config.SHAP_MODEL_PATH``.

    Returns
    -------
    Absolute path of the saved file.
    """
    import joblib

    fpath = Path(path) if path else _default_model_path()
    fpath.parent.mkdir(parents=True, exist_ok=True)

    # Save ShapModel fields individually so joblib only serialises sklearn objects
    bundle = {
        "pipeline":      model.pipeline,
        "feature_keys":  model.feature_keys,
        "train_metrics": model.train_metrics,
    }
    joblib.dump(bundle, fpath)

    # Persist feature keys as JSON for human inspection
    keys_path = fpath.with_suffix(".keys.json")
    keys_path.write_text(json.dumps(model.feature_keys, indent=2), encoding="utf-8")

    print(f"[SHAP] Model saved → {fpath}")
    return str(fpath)


def load_shap_model(path: Optional[str] = None) -> ShapModel:
    """Load a previously saved ShapModel bundle.

    Parameters
    ----------
    path : Override load path.  Defaults to ``config.SHAP_MODEL_PATH``.

    Returns
    -------
    ShapModel with a freshly instantiated TreeExplainer.

    Raises
    ------
    FileNotFoundError if the model file does not exist.
    """
    import joblib
    import shap

    fpath = Path(path) if path else _default_model_path()
    if not fpath.exists():
        raise FileNotFoundError(
            f"SHAP model not found at {fpath}. "
            "Run train_shap_model() and save_shap_model() first."
        )

    bundle = joblib.load(fpath)
    pipeline     = bundle["pipeline"]
    feature_keys = bundle["feature_keys"]
    train_metrics = bundle.get("train_metrics", {})

    gbc = pipeline.named_steps["gbc"]
    explainer = shap.TreeExplainer(gbc)

    return ShapModel(
        pipeline=pipeline,
        explainer=explainer,
        feature_keys=feature_keys,
        train_metrics=train_metrics,
    )


# ──────────────────────────────── Inference ───────────────────────────────────

def explain_pair(
    features_a: Dict[str, float],
    features_b: Dict[str, float],
    model: ShapModel,
    top_k: int = 5,
) -> ShapResult:
    """Compute SHAP attribution for a single text pair.

    Parameters
    ----------
    features_a : extract_all() output for text A.
    features_b : extract_all() output for text B.
    model      : Fitted ShapModel from ``train_shap_model`` or ``load_shap_model``.
    top_k      : Number of top features to include in the result.

    Returns
    -------
    ShapResult
    """
    scaler = model.pipeline.named_steps["scaler"]

    x = pair_vector(features_a, features_b, model.feature_keys)
    x_scaled = scaler.transform(x)

    pred_prob = model.pipeline.predict_proba(x)[0]
    # Index 1 = same-author probability
    pred_score = float(pred_prob[1])
    pred_verdict = "SAME_AUTHOR" if pred_score >= 0.5 else "DIFFERENT_AUTHOR"

    # SHAP values: shape (1 × n_features) for binary, or (2 × 1 × n_features)
    raw_shap = model.explainer.shap_values(x_scaled)

    # TreeExplainer returns list of arrays for binary classification
    # raw_shap[1] is the attribution toward class 1 (same-author)
    if isinstance(raw_shap, list):
        # shap_vals_same shape: (1, n_features) → flatten
        shap_vals = np.array(raw_shap[1]).flatten()
    else:
        shap_vals = np.array(raw_shap).flatten()

    # Map to feature keys
    all_shap: Dict[str, float] = {
        k: float(v) for k, v in zip(model.feature_keys, shap_vals)
    }
    deltas_row = x.flatten()

    # Top-k by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:top_k]
    top_features: List[ShapFeature] = []
    for idx in sorted_idx:
        k   = model.feature_keys[idx]
        sv  = float(shap_vals[idx])
        # Positive SHAP toward class 1 (same-author) → feature supports SAME_AUTHOR
        direction = "SAME_AUTHOR" if sv > 0 else "DIFFERENT_AUTHOR"
        top_features.append(ShapFeature(
            name=k,
            description=_describe_feature(k),
            shap_value=sv,
            direction=direction,
            delta=float(deltas_row[idx]),
        ))

    return ShapResult(
        pred_score=pred_score,
        pred_verdict=pred_verdict,
        top_features=top_features,
        all_shap_values=all_shap,
        feature_vector=deltas_row.tolist(),
        feature_keys=model.feature_keys,
    )


def explain_pair_from_texts(
    text_a: str,
    text_b: str,
    model: ShapModel,
    cefr_dict: Optional[Dict] = None,
    top_k: int = 5,
) -> ShapResult:
    """Convenience wrapper: extract features then explain.

    Parameters
    ----------
    text_a / text_b : Raw document strings.
    model           : Fitted ShapModel.
    cefr_dict       : CEFR wordlist (optional).
    top_k           : Number of top features to return.
    """
    from features.handcrafted import extract_all

    fa = extract_all(text_a, cefr_dict=cefr_dict)
    fb = extract_all(text_b, cefr_dict=cefr_dict)
    return explain_pair(fa, fb, model, top_k=top_k)


# ──────────────────────────────── Batch Analysis ─────────────────────────────

def explain_pairs_batch(
    pairs,                      # List[TextPair]
    model: ShapModel,
    cefr_dict: Optional[Dict] = None,
    top_k: int = 5,
    verbose: bool = True,
) -> List[ShapResult]:
    """Run explain_pair_from_texts over a list of TextPair objects.

    Parameters
    ----------
    pairs    : Text pairs to analyse.
    model    : Fitted ShapModel.
    cefr_dict: CEFR wordlist (optional).
    top_k    : Features per result.
    verbose  : Print progress.

    Returns
    -------
    List of ShapResult, one per pair (same order as input).
    """
    results: List[ShapResult] = []
    for i, pair in enumerate(pairs):
        if verbose and i % 20 == 0:
            print(f"[SHAP] Explaining pair {i+1}/{len(pairs)} …")
        results.append(
            explain_pair_from_texts(pair.text_a, pair.text_b, model,
                                    cefr_dict=cefr_dict, top_k=top_k)
        )
    return results
