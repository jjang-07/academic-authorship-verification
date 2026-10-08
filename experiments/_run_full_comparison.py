import json
import sys
from pathlib import Path

# Repo root on sys.path when run as experiments/_run_full_comparison.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import evaluate_extended

files = [
    ("Run 1 (seed 42)", "experiments/results/full_v2_textonly_details_20260319T035936Z.jsonl"),
    ("Run 2 (seed 123)", "experiments/results/full_v2_textonly_details_20260321T061033Z.jsonl"),
    ("Run 3 (seed 456)", "experiments/results/full_v2_textonly_details_20260321T193506Z.jsonl"),
]


def verdict_to_score(verdict, conf):
    if verdict == "SAME_AUTHOR":
        return 0.5 + (conf - 0.5)
    return 0.5 - (conf - 0.5)


def pct(num, den):
    return num / den * 100 if den else 0.0


for label, path in files:
    try:
        results = [json.loads(l) for l in open(path)]
        valid = [r for r in results if not r.get("error")]
        n = len(valid)

        true_labels = [1 if r["ground_truth"] == "SAME_AUTHOR" else 0 for r in valid]
        analyst_scores = [
            verdict_to_score(r["analyst_verdict"], r["analyst_confidence"]) for r in valid
        ]
        final_scores = [
            verdict_to_score(r["final_verdict"], r["judge_confidence"]) for r in valid
        ]

        am = evaluate_extended(true_labels, analyst_scores)
        fm = evaluate_extended(true_labels, final_scores)

        analyst_correct = sum(1 for r in valid if r["analyst_verdict"] == r["ground_truth"])
        final_correct = sum(1 for r in valid if r["correct"])

        same_pairs = [r for r in valid if r["ground_truth"] == "SAME_AUTHOR"]
        diff_pairs = [r for r in valid if r["ground_truth"] == "DIFFERENT_AUTHOR"]
        analyst_same_acc = sum(
            1 for r in same_pairs if r["analyst_verdict"] == r["ground_truth"]
        )
        analyst_diff_acc = sum(
            1 for r in diff_pairs if r["analyst_verdict"] == r["ground_truth"]
        )
        final_same_acc = sum(1 for r in same_pairs if r["correct"])
        final_diff_acc = sum(1 for r in diff_pairs if r["correct"])

        flip_bad = sum(
            1
            for r in valid
            if r["analyst_verdict"] == r["ground_truth"] and not r["correct"]
        )
        flip_good = sum(
            1
            for r in valid
            if r["analyst_verdict"] != r["ground_truth"] and r["correct"]
        )
        no_flip = sum(
            1
            for r in valid
            if r["analyst_verdict"] == r["ground_truth"] and r["correct"]
        )
        both_wrong = sum(
            1
            for r in valid
            if r["analyst_verdict"] != r["ground_truth"] and not r["correct"]
        )

        skeptic_agree = sum(1 for r in valid if r.get("skeptic_stance") == "AGREE")
        skeptic_partial = sum(
            1 for r in valid if r.get("skeptic_stance") == "PARTIALLY_DISAGREE"
        )
        skeptic_disagree = sum(1 for r in valid if r.get("skeptic_stance") == "DISAGREE")

        ns, nd = len(same_pairs), len(diff_pairs)
        print()
        print("=" * 60)
        print(f"{label}  (n={n})")
        print("=" * 60)
        print()
        print("  OVERALL METRICS")
        print(
            f"    Analyst alone:   Overall={am['overall']:.3f}  "
            f"AUC={am['auc']:.3f}  c@1={am['c@1']:.3f}"
        )
        print(
            f"    Full system:     Overall={fm['overall']:.3f}  "
            f"AUC={fm['auc']:.3f}  c@1={fm['c@1']:.3f}"
        )
        print()
        print("  ACCURACY")
        print(f"    Analyst:  {analyst_correct}/{n} ({pct(analyst_correct, n):.1f}%)")
        print(f"    System:   {final_correct}/{n} ({pct(final_correct, n):.1f}%)")
        print()
        print(f"  SAME-AUTHOR PAIRS  (n={ns})")
        print(
            f"    Analyst correct: {analyst_same_acc}/{ns} ({pct(analyst_same_acc, ns):.1f}%)"
        )
        print(f"    System correct:  {final_same_acc}/{ns} ({pct(final_same_acc, ns):.1f}%)")
        print()
        print(f"  DIFFERENT-AUTHOR PAIRS  (n={nd})")
        print(
            f"    Analyst correct: {analyst_diff_acc}/{nd} ({pct(analyst_diff_acc, nd):.1f}%)"
        )
        print(
            f"    System correct:  {final_diff_acc}/{nd} ({pct(final_diff_acc, nd):.1f}%)"
        )
        print()
        print("  DEBATE OUTCOMES")
        print(f"    Both correct (no flip needed):     {no_flip}")
        print(f"    Analyst wrong, Judge corrected:    {flip_good}")
        print(f"    Analyst right, Judge flipped wrong:{flip_bad}")
        print(f"    Both wrong (unfixable):            {both_wrong}")
        print(f"    Net debate contribution:           {flip_good - flip_bad:+d}")
        print()
        print("  SKEPTIC STANCE DISTRIBUTION")
        print(f"    AGREE:             {skeptic_agree}/{n} ({pct(skeptic_agree, n):.1f}%)")
        print(
            f"    PARTIALLY_DISAGREE:{skeptic_partial}/{n} ({pct(skeptic_partial, n):.1f}%)"
        )
        print(
            f"    DISAGREE:          {skeptic_disagree}/{n} ({pct(skeptic_disagree, n):.1f}%)"
        )

    except FileNotFoundError:
        print(f"{label}: FILE NOT FOUND")
