"""Orchestrates the full Noma alert-triage pipeline end to end.

    python run.py

regenerates the synthetic cohort, builds features, fits both models under
GroupKFold cross-validation, evaluates against the naive threshold baseline,
and writes figures + report/summary.md. Every stage also writes its output
to disk so it can be rerun independently (see data/, features/, models/).

This is a synthetic-data demonstration of a triage approach, not a claim
about real clinical performance.
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HORIZON_HOURS, PATHS, SEED
from data.generate import generate_cohort, _plot_overlap_check
from eval.metrics import (
    alert_volume_vs_missed_events,
    alerts_per_patient_day_at_threshold,
    baseline_false_alarm_breakdown,
    fold_variance_report,
    normalization_ablation,
    ppv_at_sensitivity,
    threshold_for_sensitivity,
)
from features.build import build_features, _check_no_lookahead
from models.baseline import apply_baseline_rule, summarize as summarize_baseline
from models.supervised import cross_validate, fit_final_models

REPORT_DIR = Path(PATHS["report_dir"])
TARGET_SENSITIVITY = 0.90


def stage_data():
    print("[1/6] Generating synthetic cohort...")
    df = generate_cohort(seed=SEED)
    out_path = Path(PATHS["raw"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_overlap_check(df, REPORT_DIR)
    n_pat = df.patient_id.nunique()
    print(f"  {len(df):,} rows, {n_pat} patients, "
          f"{df.groupby('patient_id')['deteriorating'].any().sum()} with deterioration events, "
          f"{df.groupby('patient_id')['artifact'].any().sum()} with artifact episodes")
    return df


def stage_baseline(df):
    print("[2/6] Applying naive threshold baseline rule...")
    df_alerted = apply_baseline_rule(df)
    stats = summarize_baseline(df_alerted)
    print(f"  {stats['alerts_per_patient_day']} alerts/patient-day, "
          f"{stats['n_false_alarms']:,} false alarms, patient-level sensitivity {stats['patient_level_sensitivity']}")
    return df_alerted, stats


def stage_features(df):
    print("[3/6] Building per-patient-normalized features...")
    _check_no_lookahead()
    feats = build_features(df)
    out_path = Path(PATHS["features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    return feats


def stage_models():
    print("[4/6] Fitting logistic regression + GBT under GroupKFold CV (normalized features)...")
    result_norm, cols_norm = cross_validate("normalized", seed=SEED)
    print("[4/6] Fitting ablation models on raw (non-normalized) features...")
    result_raw, cols_raw = cross_validate("raw", seed=SEED)
    interp = fit_final_models("normalized", seed=SEED)

    models_dir = Path("models")
    result_norm.to_parquet(models_dir / "oof_predictions_normalized.parquet", index=False)
    result_raw.to_parquet(models_dir / "oof_predictions_raw.parquet", index=False)
    interp["lr_coefficients"].to_csv(models_dir / "lr_coefficients.csv")
    interp["gbt_feature_importances"].to_csv(models_dir / "gbt_feature_importances.csv")

    return result_norm, result_raw, interp, cols_norm


def baseline_pr_point(feats: pd.DataFrame, df_alerted_full: pd.DataFrame):
    """Baseline rule as a single (recall, precision) point, evaluated on the
    same modelable-row / label_onset definition used for the models."""
    mask = (~feats["in_calibration"]) & (~feats["currently_deteriorating"])
    sub = feats.loc[mask, ["patient_id", "timestamp"]].copy()
    alert_lookup = df_alerted_full.set_index(["patient_id", "timestamp"])["alert"]
    sub_idx = pd.MultiIndex.from_frame(sub[["patient_id", "timestamp"]])
    alert = alert_lookup.reindex(sub_idx).fillna(False).to_numpy()
    y = feats.loc[mask, "label_onset"].to_numpy()

    tp = (alert & y).sum()
    fp = (alert & ~y).sum()
    fn = (~alert & y).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def make_pr_curve_figure(result_norm, baseline_point, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for col, label in [("prob_lr", "Logistic regression"), ("prob_gbt", "Gradient-boosted trees")]:
        precision, recall, _ = precision_recall_curve(result_norm["y_true"], result_norm[col])
        ap = average_precision_score(result_norm["y_true"], result_norm[col])
        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    ax.scatter([baseline_point[1]], [baseline_point[0]], color="red", zorder=5, s=80,
               label=f"Naive threshold baseline (P={baseline_point[0]:.3f}, R={baseline_point[1]:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"Precision-recall: predicting deterioration onset within {HORIZON_HOURS}h\n"
                 "(GroupKFold out-of-fold predictions, grouped by patient)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_tradeoff_figure(result_norm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for col, label in [("prob_lr", "Logistic regression"), ("prob_gbt", "Gradient-boosted trees")]:
        sweep = alert_volume_vs_missed_events(result_norm, col)
        ax.plot(sweep["alerts_per_patient_day"], sweep["missed_fraction"] * 100, marker="o", markersize=3, label=label)
    ax.set_xlabel("Alerts per patient-day")
    ax.set_ylabel("Missed onset-labeled windows (%)")
    ax.set_title("Alert volume vs. missed events\n(sweep across decision thresholds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_false_alarm_figure(causes, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    causes.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("False-alarm count (baseline rule)")
    ax.set_title("What causes baseline-rule false alarms?")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_ablation_figure(ablation_stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Raw (no per-patient\nnormalization)", "Per-patient\nnormalized"]
    ap_vals = [ablation_stats["gbt_ap_raw"], ablation_stats["gbt_ap_normalized"]]
    ppv_vals = [ablation_stats["gbt_ppv_at_90_raw"] or 0, ablation_stats["gbt_ppv_at_90_normalized"] or 0]
    axes[0].bar(labels, ap_vals, color=["#999999", "#4C72B0"])
    axes[0].set_title("Average precision (GBT)")
    axes[1].bar(labels, ppv_vals, color=["#999999", "#4C72B0"])
    axes[1].set_title(f"PPV @ {int(TARGET_SENSITIVITY*100)}% sensitivity (GBT)")
    fig.suptitle("Does per-patient normalization matter?")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(ctx):
    lines = []
    lines.append("# Noma alert-triage prototype — summary report\n")
    lines.append(
        "**This is a synthetic-data demonstration of a triage approach, not a claim about "
        "real clinical performance.** All patients, vitals, and events in this report are "
        "generated by `data/generate.py`; no real patient data was used.\n"
    )

    lines.append("## Headline: alert volume at matched sensitivity\n")
    b = ctx["baseline_stats"]
    lines.append(f"- Naive threshold baseline: **{b['alerts_per_patient_day']:.1f} alerts/patient-day** "
                 f"({b['n_alerts']:,} alerts over {b['n_patient_days']:.0f} patient-days).\n")
    lines.append(f"- On the row-level onset-window label used to evaluate both approaches, the "
                 f"baseline achieves **{ctx['baseline_row_recall']:.1%} sensitivity** at "
                 f"**{ctx['baseline_row_precision']:.4f} precision** (it was never designed to hit "
                 "90% sensitivity — a fixed threshold with no persistence check catches some onsets "
                 "by chance while flooding clinicians with unrelated alerts).\n")
    gbt_thresh_m = ctx["gbt_thresh_matched"]
    gbt_apd_m = ctx["gbt_alerts_per_day_matched"]
    lines.append(f"- GBT model matched to that **same {ctx['matched_sensitivity']:.1%} sensitivity** "
                 f"(probability threshold {gbt_thresh_m:.3f}): **{gbt_apd_m:.3f} alerts/patient-day**.\n")
    if b["alerts_per_patient_day"] > 0 and gbt_apd_m is not None:
        reduction = 1 - gbt_apd_m / b["alerts_per_patient_day"]
        lines.append(f"- That is a **{reduction:.1%} reduction** in alert volume at the same "
                     "sensitivity, apples-to-apples on the same row-level label.\n")
    gbt_thresh_90 = ctx["gbt_thresh_90"]
    gbt_apd_90 = ctx["gbt_alerts_per_day_90"]
    lines.append(f"- If instead the target is a strict **{int(TARGET_SENSITIVITY*100)}% sensitivity** "
                 f"(catch nearly every onset, the baseline cannot reach on this label at all): the GBT "
                 f"model needs **{gbt_apd_90:.1f} alerts/patient-day** (threshold {gbt_thresh_90:.3f}) "
                 "— the honest cost of insisting on near-total sensitivity against a rare, only "
                 "partially-predictable event. See the tradeoff figure below for the full curve.\n")

    lines.append("\n## Cross-validated performance (GroupKFold, 5 folds, grouped by patient)\n")
    lines.append("| Model | Mean AP | Std AP | Mean PPV@90% sens. | Std PPV@90% sens. |")
    lines.append("|---|---|---|---|---|")
    for name, summ in [("Logistic regression", ctx["fold_summary_lr"]), ("Gradient-boosted trees", ctx["fold_summary_gbt"])]:
        lines.append(f"| {name} | {summ['mean_ap']:.3f} | {summ['std_ap']:.3f} | "
                     f"{summ['mean_ppv_at_90']:.3f} | {summ['std_ppv_at_90']:.3f} |")
    lines.append(
        f"\nWith only {ctx['n_positive_patients']} deterioration-event patients, fold-to-fold "
        "variance is wide — reported honestly rather than collapsed to a single number.\n"
    )

    lines.append("\n## Precision-recall\n")
    lines.append("![PR curves](pr_curves.png)\n")

    lines.append("\n## The tradeoff: alert volume vs. missed events\n")
    lines.append("![Alert volume tradeoff](alert_volume_tradeoff.png)\n")
    lines.append(
        "This is the figure that matters operationally: it shows the full tradeoff curve so a "
        "clinician (not this notebook) can choose the operating point.\n"
    )

    lines.append("\n## Why are baseline-rule alerts false alarms?\n")
    lines.append("![False alarm causes](false_alarm_causes.png)\n")
    causes = ctx["false_alarm_causes"]
    total = causes.sum()
    for cause, count in causes.items():
        lines.append(f"- **{cause}**: {count:,} ({count/total:.1%})")
    lines.append(
        "\n`baseline_mismatch` — a fixed population threshold flagging a patient whose personal "
        "normal is outside the population's normal — is the single largest cause, which is exactly "
        "the failure mode per-patient normalization is meant to fix.\n"
    )

    lines.append("\n## Does per-patient normalization matter?\n")
    lines.append("![Normalization ablation](normalization_ablation.png)\n")
    a = ctx["ablation"]
    lines.append(f"- GBT average precision: **{a['gbt_ap_normalized']:.3f} (normalized)** vs. "
                 f"{a['gbt_ap_raw']:.3f} (raw values, no per-patient baseline).")
    lines.append(f"- GBT PPV @ {int(TARGET_SENSITIVITY*100)}% sensitivity: "
                 f"**{a['gbt_ppv_at_90_normalized'] or 0:.3f} (normalized)** vs. "
                 f"{a['gbt_ppv_at_90_raw'] or 0:.3f} (raw).\n")
    lines.append(
        "This is the central thesis of the project: one patient's normal resting heart rate is "
        "another patient's alarm, and the ablation above tests whether accounting for that "
        "actually helps, rather than assuming it does.\n"
    )

    lines.append("\n## Which features carried the signal\n")
    lines.append("Top logistic-regression coefficients (normalized features, by |coefficient|):\n")
    lines.append("```")
    lines.append(ctx["lr_coefs"].head(8).to_string())
    lines.append("```")
    lines.append("\nTop GBT permutation importances (normalized features):\n")
    lines.append("```")
    lines.append(ctx["gbt_importances"].head(8).to_string())
    lines.append("```\n")

    lines.append("\n## Limitations\n")
    lines.append(
        "- **All data is synthetic.** Generated by a hand-written simulator (`data/generate.py`); "
        "it has not been validated against any real patient population. Nothing here is a claim "
        "about real-world sensitivity, specificity, or alert burden.\n"
        "- **Event count is small.** Only "
        f"{ctx['n_positive_patients']} of {ctx['n_patients']} simulated patients ever deteriorate. "
        "Cross-validated metrics are reported with fold-to-fold variance for exactly this reason — "
        "point estimates from this few positive events are not reliable on their own.\n"
        "- **No external validity.** The cohort, thresholds, and event generation were built for "
        "this repo. They do not represent any real device, hospital, or patient population.\n"
        "- **The generator's assumptions bound what the model can learn.** Per-patient baseline "
        "variance, the overlap between benign activity and deterioration, and the artifact model "
        "were all authored choices (see the overlap-check figure below) — the model's apparent "
        "skill reflects those choices, not a property of real physiology.\n"
        "- **Gradient boosting here is scikit-learn's `HistGradientBoostingClassifier`, not "
        "LightGBM/XGBoost** — both required an OpenMP runtime (`libomp`) not available on this "
        "machine's Python/Homebrew architecture combination; sklearn's GBT implementation avoids "
        "that dependency without changing the method.\n"
    )
    lines.append("### Generator overlap check\n")
    lines.append(
        "The generator was deliberately tuned so deterioration overlaps in magnitude with benign "
        "activity on every single channel (see below) — otherwise a naive single-vital rule could "
        "separate the classes and the exercise would demonstrate nothing.\n"
    )
    lines.append("![Generator overlap check](generator_overlap_check.png)\n")

    out_path = REPORT_DIR / "summary.md"
    out_path.write_text("\n".join(lines))
    print(f"  Wrote {out_path}")


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = stage_data()
    df_alerted, baseline_stats = stage_baseline(df)
    feats = stage_features(df)
    result_norm, result_raw, interp, cols = stage_models()

    print("[5/6] Computing evaluation metrics and figures...")
    fold_df_lr, fold_summary_lr = fold_variance_report(result_norm, "prob_lr")
    fold_df_gbt, fold_summary_gbt = fold_variance_report(result_norm, "prob_gbt")

    baseline_p, baseline_r = baseline_pr_point(feats, df_alerted)

    # Headline comparison is matched to the baseline's own achieved row-level
    # sensitivity (not an arbitrary 90%) — the baseline never reaches 90%
    # row-level recall on the onset-window label, so comparing there would
    # not be apples-to-apples. The separate "PPV at 90% sensitivity" metric
    # required below answers a different question: the cost of insisting on
    # near-total sensitivity, regardless of what the baseline achieves.
    matched_sensitivity = max(baseline_r, 0.01)
    gbt_thresh_matched = threshold_for_sensitivity(result_norm["y_true"], result_norm["prob_gbt"], matched_sensitivity)
    gbt_apd_matched = alerts_per_patient_day_at_threshold(result_norm, "prob_gbt", gbt_thresh_matched)

    gbt_thresh_90 = threshold_for_sensitivity(result_norm["y_true"], result_norm["prob_gbt"], TARGET_SENSITIVITY)
    gbt_apd_90 = alerts_per_patient_day_at_threshold(result_norm, "prob_gbt", gbt_thresh_90)

    false_alarm_causes = baseline_false_alarm_breakdown(df)
    ablation = normalization_ablation(result_norm, result_raw)

    make_pr_curve_figure(result_norm, (baseline_p, baseline_r), REPORT_DIR / "pr_curves.png")
    make_tradeoff_figure(result_norm, REPORT_DIR / "alert_volume_tradeoff.png")
    make_false_alarm_figure(false_alarm_causes, REPORT_DIR / "false_alarm_causes.png")
    make_ablation_figure(ablation, REPORT_DIR / "normalization_ablation.png")

    n_patients = feats["patient_id"].nunique()
    n_positive_patients = int(df.groupby("patient_id")["deteriorating"].any().sum())

    ctx = dict(
        baseline_stats=baseline_stats,
        baseline_row_precision=baseline_p,
        baseline_row_recall=baseline_r,
        matched_sensitivity=matched_sensitivity,
        gbt_thresh_matched=gbt_thresh_matched,
        gbt_alerts_per_day_matched=gbt_apd_matched,
        gbt_thresh_90=gbt_thresh_90,
        gbt_alerts_per_day_90=gbt_apd_90,
        fold_summary_lr=fold_summary_lr,
        fold_summary_gbt=fold_summary_gbt,
        false_alarm_causes=false_alarm_causes,
        ablation=ablation,
        lr_coefs=interp["lr_coefficients"],
        gbt_importances=interp["gbt_feature_importances"],
        n_patients=n_patients,
        n_positive_patients=n_positive_patients,
    )

    print("[6/6] Writing report...")
    write_report(ctx)
    print("Done. See report/summary.md")


if __name__ == "__main__":
    main()
