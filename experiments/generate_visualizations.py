#!/usr/bin/env python3
"""
Publication-quality figures for science-fair / paper-style reporting.

Run from repo root:
    ./venv/bin/python experiments/generate_visualizations.py

Outputs high-resolution PNGs under experiments/results/figures/.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

# Matplotlib needs a writable config dir in some CI/sandbox environments.
_mpl_cache = Path(__file__).resolve().parent.parent / ".matplotlib-cache"
if "MPLCONFIGDIR" not in os.environ:
    try:
        _mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_cache)
    except OSError:
        pass

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from plottable import Table
from plottable.column_def import ColDef

# Prefer Calibri when installed (common on Windows / Mac with Office); else bundled DejaVu Sans.
_available_fonts = [f.name for f in fm.fontManager.ttflist]
FONT_FAMILY = "Calibri" if "Calibri" in _available_fonts else "DejaVu Sans"
plt.rcParams["font.family"] = FONT_FAMILY

# Global typography & axes style (publication defaults)
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 12,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Transparent canvas defaults (poster-friendly PNGs on any background)
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["savefig.transparent"] = True

# ── Paths (repo root = cwd or parent of experiments/) ────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.metrics import evaluate_extended, verdict_to_pred_score  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUN_JSONL = {
    1: _REPO_ROOT / "experiments/results/full_v2_textonly_details_20260319T035936Z.jsonl",
    2: _REPO_ROOT / "experiments/results/full_v2_textonly_details_20260321T061033Z.jsonl",
    3: _REPO_ROOT / "experiments/results/full_v2_textonly_details_20260321T193506Z.jsonl",
}

# Canonical result bundles for the comparison table (100-pair CEFR eval where noted).
_TABLE_V1_JSON = _REPO_ROOT / "experiments/results/v1_baseline_20260317T023651Z.json"
_TABLE_NAIVE_DEBATE_JSON = _REPO_ROOT / "experiments/results/naive_debate_textonly_20260323T013326Z.json"
_TABLE_NAIVE_MAD_JSON = _REPO_ROOT / "experiments/results/naive_mad_20260317T024535Z.json"
_TABLE_FULL_V2_JSON = _REPO_ROOT / "experiments/results/full_v2_textonly_20260321T113344Z.json"

# Exclusive palette: primary → secondary → tertiary
C_PRIMARY = "#26cbc6"  # teal
C_SECONDARY = "#52c9e8"  # light blue
C_TERTIARY = "#8ab5ef"  # periwinkle

CMAP_CONFUSION = LinearSegmentedColormap.from_list(
    "white_teal", ["#ffffff", C_PRIMARY], N=256
)

DPI = 300


def _transparent_figure_axes(fig, axes) -> None:
    """Make figure and all axes patches fully transparent (works for single ax or ndarray of axes)."""
    fig.patch.set_alpha(0)
    for ax in np.atleast_1d(axes).ravel():
        ax.patch.set_alpha(0)


def _load_valid(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSONL: {path}")
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("error"):
                out.append(r)
    return out


def _confusion_matrix(records: list[dict]) -> np.ndarray:
    """Rows = Actual, Cols = Predicted. Order: SAME_AUTHOR, DIFFERENT_AUTHOR."""
    cm = np.zeros((2, 2), dtype=float)
    labels = ("SAME_AUTHOR", "DIFFERENT_AUTHOR")
    for r in records:
        a = labels.index(r["ground_truth"])
        p = labels.index(r["final_verdict"])
        cm[a, p] += 1
    return cm


def _stance_accuracy(records: list[dict]) -> dict[str, tuple[int, int]]:
    """stance -> (correct_count, total)."""
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        stance = r.get("skeptic_stance") or "UNKNOWN"
        agg[stance][1] += 1
        if r.get("correct"):
            agg[stance][0] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}


def _flip_counts(records: list[dict]) -> tuple[int, int]:
    """
    Count debate flips where the judge differs from the analyst.
    Harmful: analyst matched ground truth, final did not (judge broke a correct call).
    Helpful: analyst was wrong, final matched ground truth (judge fixed it).
    """
    harmful = helpful = 0
    for r in records:
        av, fv, gt = r["analyst_verdict"], r["final_verdict"], r["ground_truth"]
        if av == fv:
            continue
        if av == gt and fv != gt:
            harmful += 1
        elif av != gt and fv == gt:
            helpful += 1
    return harmful, helpful


def fig_main_results() -> Path:
    names = [
        "Wan LR Baseline",
        "V1 Traditional ML",
        "V2 Analyst Alone",
        "V2 Full Debate System",
    ]
    scores = [0.562, 0.707, 0.782, 0.723]
    # Tertiary, secondary, primary, tertiary — repeat tertiary for fourth bar
    colors = [C_TERTIARY, C_SECONDARY, C_PRIMARY, C_TERTIARY]
    v1_ref = 0.707

    fig, ax = plt.subplots(figsize=(10, 6))
    _transparent_figure_axes(fig, ax)
    x = np.arange(len(names))
    bars = ax.bar(x, scores, color=colors, edgecolor="#2a2a2a", linewidth=0.6, width=0.72)

    ax.axhline(
        v1_ref,
        color=C_SECONDARY,
        linestyle="--",
        linewidth=1.5,
        label=f"V1 reference ({v1_ref:.3f})",
        zorder=0,
    )
    ax.set_ylabel("Overall Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(0.45, 0.92)
    ax.set_title("Overall Score Comparison")
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="medium",
        )

    # V2 Analyst Alone = third bar (index 2)
    bar_analyst = bars[2]
    ax.text(
        bar_analyst.get_x() + bar_analyst.get_width() / 2,
        bar_analyst.get_height() + 0.055,
        "+10.6% vs V1",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="medium",
        color="#2a2a2a",
    )

    fig.tight_layout()
    out = FIG_DIR / "main_results_overall_scores.png"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return out


def fig_confusion_matrices() -> Path:
    run2 = _load_valid(RUN_JSONL[2])
    run3 = _load_valid(RUN_JSONL[3])
    cm2 = _confusion_matrix(run2)
    cm3 = _confusion_matrix(run3)
    tick = ["Same Author", "Different Author"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    _transparent_figure_axes(fig, axes)

    titles = (
        "Confusion Matrix\nRun 2 (Initial Judge Prompts)",
        "Confusion Matrix\nRun 3 (Conservative Judge Prompts)",
    )
    for ax, cm, title in zip(axes, (cm2, cm3), titles):
        sns.heatmap(
            cm,
            annot=False,
            cmap=CMAP_CONFUSION,
            square=True,
            cbar=True,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            vmin=0,
            vmax=float(cm.max()),
            xticklabels=tick,
            yticklabels=tick,
        )
        # Custom count labels: bold 18pt; white if count > 20 (dark cell), else dark gray
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = int(cm[i, j])
                txt_color = "white" if val > 20 else "#1a1a1a"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color=txt_color,
                )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(title)

    # Transparent backgrounds on any extra axes (e.g. colorbars)
    for axx in fig.axes:
        axx.patch.set_alpha(0)

    plt.tight_layout(pad=2.0)
    out = FIG_DIR / "confusion_matrices_run2_run3.png"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return out


def fig_skeptic_stance_accuracy() -> Path:
    stances = ["AGREE", "PARTIALLY_DISAGREE", "DISAGREE"]
    run2 = _load_valid(RUN_JSONL[2])
    run3 = _load_valid(RUN_JSONL[3])
    s2 = _stance_accuracy(run2)
    s3 = _stance_accuracy(run3)

    def pct(stance: str, d: dict[str, tuple[int, int]]) -> float:
        c, t = d.get(stance, (0, 0))
        return 100.0 * c / t if t else float("nan")

    r2_vals = [pct(st, s2) for st in stances]
    r3_vals = [pct(st, s3) for st in stances]

    x = np.arange(len(stances))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    _transparent_figure_axes(fig, ax)
    bars2 = ax.bar(
        x - width / 2,
        r2_vals,
        width,
        label="Run 2",
        color=C_PRIMARY,
        edgecolor="#2a2a2a",
        linewidth=0.5,
    )
    bars3 = ax.bar(
        x + width / 2,
        r3_vals,
        width,
        label="Run 3",
        color=C_SECONDARY,
        edgecolor="#2a2a2a",
        linewidth=0.5,
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Agree", "Partially Disagree", "Disagree"])
    ax.set_title("Skeptic Stance vs. Final Verdict Accuracy")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )
    for bar in bars3:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    fig.tight_layout()
    out = FIG_DIR / "skeptic_stance_accuracy_run2_run3.png"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return out


def fig_debate_flip_progression() -> Path:
    harmful, helpful = [], []
    for run_id in (1, 2, 3):
        rec = _load_valid(RUN_JSONL[run_id])
        h, hl = _flip_counts(rec)
        harmful.append(h)
        helpful.append(hl)

    iterations = ["Run 1", "Run 2", "Run 3"]
    x = np.arange(len(iterations))
    fig, ax = plt.subplots(figsize=(10, 6))
    _transparent_figure_axes(fig, ax)
    ax.plot(
        x,
        harmful,
        marker="o",
        linewidth=2.4,
        markersize=8,
        color=C_PRIMARY,
        label="Harmful Flips (Analyst Correct → Judge Wrong)",
    )
    ax.plot(
        x,
        helpful,
        marker="o",
        linewidth=2.4,
        markersize=8,
        color=C_SECONDARY,
        label="Helpful Flips (Analyst Wrong → Judge Corrected)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(iterations)
    ax.set_xlabel("Prompt Iteration")
    ax.set_ylabel("Number of Flips")
    ax.set_title("Debate Flip Progression Across Prompt Iterations")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = FIG_DIR / "debate_flip_progression.png"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return out


def fig_confidence_histogram_run2() -> Path:
    run2 = _load_valid(RUN_JSONL[2])
    conf_correct = [float(r["judge_confidence"]) for r in run2 if r.get("correct")]
    conf_wrong = [float(r["judge_confidence"]) for r in run2 if not r.get("correct")]

    fig, ax = plt.subplots(figsize=(10, 6))
    _transparent_figure_axes(fig, ax)
    bins = np.linspace(0.45, 1.0, 23)
    ax.hist(
        conf_correct,
        bins=bins,
        alpha=0.6,
        color=C_PRIMARY,
        label="Correct prediction",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        conf_wrong,
        bins=bins,
        alpha=0.6,
        color=C_SECONDARY,
        label="Incorrect prediction",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Judge Confidence")
    ax.set_ylabel("Number of Pairs")
    ax.set_title("Judge Confidence Distribution — Run 2")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#cccccc")
    fig.tight_layout()
    out = FIG_DIR / "confidence_histogram_run2.png"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return out


def _threshold_accuracy(
    true_y: list | np.ndarray,
    pred_scores: list | np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Fraction correct at 0.5 boundary; scores exactly equal to threshold count as incorrect."""
    t = np.asarray(true_y, dtype=int)
    s = np.asarray(pred_scores, dtype=float)
    correct = ((s > threshold) & (t == 1)) | ((s < threshold) & (t == 0))
    return round(float(np.mean(correct)), 3)


def _fmt_metric_scalar(x: float | None) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(xf):
        return "—"
    return f"{xf:.3f}"


def _table_cells_from_extended(
    ext: dict[str, float],
    true_y: Sequence,
    pred_scores: Sequence[float],
) -> dict[str, str]:
    acc = _threshold_accuracy(true_y, pred_scores)
    return {
        "Overall": _fmt_metric_scalar(ext.get("overall")),
        "AUC": _fmt_metric_scalar(ext.get("auc")),
        "c@1": _fmt_metric_scalar(ext.get("c@1")),
        "F1": _fmt_metric_scalar(ext.get("F1")),
        "Accuracy": _fmt_metric_scalar(acc),
        "Same-author acc": _fmt_metric_scalar(ext.get("same_author_accuracy")),
        "Diff-author acc": _fmt_metric_scalar(ext.get("different_author_accuracy")),
    }


def fig_results_comparison_table() -> Path:
    """Neutral LaTeX/Overleaf-style comparison table via plottable (300 DPI, transparent PNG).

    Rows use bundled JSON/JSONL under ``experiments/results/`` (see ``_TABLE_*`` paths).
    Wan: literature overall only (sub-metrics unavailable). V1: full Reuters run in the
    V1 JSON (~2.4k pairs). Naive debate, V2 analyst, and V2 full debate: 100-pair CEFR
    benchmark. Approach 2 (MAD): pilot subset (n=10) from the saved MAD JSON.
    """
    # ── Load / compute metric rows (100-pair CEFR eval unless noted) ─────────
    wan_row = {
        "Approach": "Wan LR Baseline",
        "Model": "Logistic regression (Reuters-50-50)",
        "Overall": "0.562",
        "AUC": "—",
        "c@1": "—",
        "F1": "—",
        "Accuracy": "—",
        "Same-author acc": "—",
        "Diff-author acc": "—",
    }

    v1_data = json.loads(_TABLE_V1_JSON.read_text(encoding="utf-8"))
    v1_y, v1_p = v1_data["true_labels"], v1_data["pred_scores"]
    v1_ext = evaluate_extended(v1_y, v1_p)
    v1_row = {
        "Approach": "V1 Traditional ML",
        "Model": "V1 baseline (GB + hand-crafted features)",
        **_table_cells_from_extended(v1_ext, v1_y, v1_p),
    }

    nd = json.loads(_TABLE_NAIVE_DEBATE_JSON.read_text(encoding="utf-8"))
    nd_y, nd_p = nd["true_labels"], nd["pred_scores"]
    nd_ext = evaluate_extended(nd_y, nd_p)
    naive_row = {
        "Approach": "Approach 1 Naive Debate",
        "Model": "Naive debate (text-only, same model)",
        **_table_cells_from_extended(nd_ext, nd_y, nd_p),
    }

    mad = json.loads(_TABLE_NAIVE_MAD_JSON.read_text(encoding="utf-8"))
    mad_y, mad_p = mad["true_labels"], mad["pred_scores"]
    mad_ext = evaluate_extended(mad_y, mad_p)
    mad_row = {
        "Approach": "Approach 2 Feature Deltas",
        "Model": "MAD-style agents (pilot subset)",
        **_table_cells_from_extended(mad_ext, mad_y, mad_p),
    }

    rec2 = _load_valid(RUN_JSONL[2])
    y_llm = [1 if r["ground_truth"] == "SAME_AUTHOR" else 0 for r in rec2]
    p_analyst = [verdict_to_pred_score(r["analyst_verdict"], r["analyst_confidence"]) for r in rec2]
    ext_analyst = evaluate_extended(y_llm, p_analyst)
    analyst_row = {
        "Approach": "V2 Analyst Alone",
        "Model": "V2 analyst only (Run 2 JSONL)",
        **_table_cells_from_extended(ext_analyst, y_llm, p_analyst),
    }

    fv2 = json.loads(_TABLE_FULL_V2_JSON.read_text(encoding="utf-8"))
    fv2_y, fv2_p = fv2["true_labels"], fv2["pred_scores"]
    fv2_ext = evaluate_extended(fv2_y, fv2_p)
    full_row = {
        "Approach": "V2 Full Debate",
        "Model": "Analyst + Skeptic + Judge (Run 2 bundle)",
        **_table_cells_from_extended(fv2_ext, fv2_y, fv2_p),
    }

    rows = [wan_row, v1_row, naive_row, mad_row, analyst_row, full_row]
    df = pd.DataFrame(rows)
    primary_approach = "V2 Analyst Alone"
    assert df["Approach"].tolist().index(primary_approach) == 4

    # LaTeX / Overleaf-style neutrals (booktabs-like: no vertical rules, gray header, zebra body).
    _tbl_header = "#d0d0d0"
    _tbl_row_white = "#ffffff"
    _tbl_row_gray = "#f0f0f0"
    _tbl_primary_row = "#e4e4e4"  # subtle emphasis for main result (still grayscale)
    _tbl_rule = "#a8a8a8"
    _tbl_text = "#111111"

    fig_w, fig_h = 16.0, 4.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _transparent_figure_axes(fig, ax)

    col_defs = [
        ColDef(
            "Approach",
            width=1.35,
            textprops={"ha": "left", "fontsize": 10, "color": _tbl_text},
        ),
        ColDef("Model", width=1.5, textprops={"ha": "left", "fontsize": 9, "color": _tbl_text}),
        ColDef("Overall", width=0.65, textprops={"ha": "center", "fontsize": 10, "color": _tbl_text}),
        ColDef("AUC", width=0.55, textprops={"ha": "center", "fontsize": 10, "color": _tbl_text}),
        ColDef("c@1", width=0.55, textprops={"ha": "center", "fontsize": 10, "color": _tbl_text}),
        ColDef("F1", width=0.55, textprops={"ha": "center", "fontsize": 10, "color": _tbl_text}),
        ColDef("Accuracy", width=0.75, textprops={"ha": "center", "fontsize": 10, "color": _tbl_text}),
        ColDef(
            "Same-author acc",
            width=0.95,
            textprops={"ha": "center", "fontsize": 10, "color": _tbl_text},
        ),
        ColDef(
            "Diff-author acc",
            width=0.95,
            textprops={"ha": "center", "fontsize": 10, "color": _tbl_text},
        ),
    ]

    tab = Table(
        df,
        ax=ax,
        index_col="Approach",
        column_definitions=col_defs,
        textprops={"fontsize": 10, "color": _tbl_text},
        col_label_cell_kw={
            "facecolor": _tbl_header,
            "edgecolor": "none",
            "linewidth": 0,
            "height": 1.05,
        },
        cell_kw={"linewidth": 0, "edgecolor": "none"},
        even_row_color=_tbl_row_white,
        odd_row_color=_tbl_row_gray,
        row_dividers=True,
        row_divider_kw={"color": _tbl_rule, "linewidth": 0.6, "alpha": 1.0},
        col_label_divider=True,
        col_label_divider_kw={"color": _tbl_rule, "linewidth": 0.9, "alpha": 1.0},
        footer_divider=True,
        footer_divider_kw={"color": _tbl_rule, "linewidth": 0.9, "alpha": 1.0},
    )

    # Header: bold dark text on gray
    for cell in tab.col_label_row.cells:
        if hasattr(cell, "text"):
            cell.text.set_color(_tbl_text)
            cell.text.set_fontweight("bold")

    # Primary result row: slightly darker band + bold (typical `\rowcolor{gray!25}` feel)
    primary_idx = df["Approach"].tolist().index(primary_approach)
    tab.rows[primary_idx].set_facecolor(_tbl_primary_row)
    for cell in tab.rows[primary_idx].cells:
        if hasattr(cell, "text"):
            cell.text.set_fontweight("bold")

    fig.savefig(
        FIG_DIR / "results_table.png",
        transparent=True,
        bbox_inches="tight",
        dpi=DPI,
        pad_inches=0.08,
    )
    plt.close(fig)
    return FIG_DIR / "results_table.png"


def main() -> None:
    print(f"Using matplotlib font: {FONT_FAMILY}")
    sns.set_theme(style="white", rc={"axes.spines.top": False, "axes.spines.right": False})
    # Re-apply full rc after seaborn (fonts, sizes)
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": DPI,
            "savefig.transparent": True,
        }
    )
    outputs = [
        fig_main_results(),
        fig_confusion_matrices(),
        fig_skeptic_stance_accuracy(),
        fig_debate_flip_progression(),
        fig_confidence_histogram_run2(),
        fig_results_comparison_table(),
    ]
    print("Wrote figures:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
