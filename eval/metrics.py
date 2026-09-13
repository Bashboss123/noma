"""Evaluation: PR curves, PPV at fixed sensitivity, alert-volume tradeoffs,
alerts-per-patient-day, and false-alarm attribution.

Deliberately does not report accuracy: with ~0.36% positive rate, a model
that never alerts scores >99% accuracy while being useless.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS
from models.baseline import apply_baseline_rule, classify_false_alarm_cause


def ppv_at_sensitivity(y_true, y_score, target_sensitivity=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns recall descending as thresholds increase;
    # find the operating point with recall closest to but >= target.
    valid = recall[:-1] >= target_sensitivity  # last point has no threshold
    if not valid.any():
        return None, None
    idx = np.where(valid)[0][-1]  # highest-threshold point still meeting target
    return precision[idx], thresholds[idx]


def fold_variance_report(result: pd.DataFrame, prob_col: str):
    rows = []
    for fold, g in result.groupby("fold"):
        ap = average_precision_score(g["y_true"], g[prob_col])
        ppv, thresh = ppv_at_sensitivity(g["y_true"], g[prob_col])
        rows.append({"fold": fold, "average_precision": ap, "ppv_at_90pct_sensitivity": ppv})
    df = pd.DataFrame(rows)
    summary = {
        "mean_ap": df["average_precision"].mean(),
        "std_ap": df["average_precision"].std(),
        "mean_ppv_at_90": df["ppv_at_90pct_sensitivity"].mean(),
        "std_ppv_at_90": df["ppv_at_90pct_sensitivity"].std(),
    }
    return df, summary


def alerts_per_patient_day_at_threshold(result: pd.DataFrame, prob_col: str, threshold: float):
    result = result.copy()
    result["model_alert"] = result[prob_col] >= threshold
    n_patient_days = result.groupby("patient_id")["timestamp"].apply(
        lambda s: (s.max() - s.min()).total_seconds() / 86400 + 1
    ).sum()
    return result["model_alert"].sum() / n_patient_days


def threshold_for_sensitivity(y_true, y_score, target_sensitivity=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = recall[:-1] >= target_sensitivity
    if not valid.any():
        return 0.0
    idx = np.where(valid)[0][-1]
    return thresholds[idx]


def alert_volume_vs_missed_events(result: pd.DataFrame, prob_col: str, n_points=25):
    """Sweep thresholds; return alerts-per-patient-day vs fraction of
    onset-labeled rows missed, for the tradeoff figure."""
    scores = result[prob_col].to_numpy()
    thresholds = np.quantile(scores, np.linspace(0.0, 0.999, n_points))
    thresholds = np.unique(thresholds)
    n_patient_days = result.groupby("patient_id")["timestamp"].apply(
        lambda s: (s.max() - s.min()).total_seconds() / 86400 + 1
    ).sum()
    n_pos = result["y_true"].sum()

    rows = []
    for t in thresholds:
        alert = scores >= t
        alerts_per_day = alert.sum() / n_patient_days
        caught = (alert & (result["y_true"] == 1)).sum()
        missed_frac = 1 - caught / n_pos if n_pos else np.nan
        rows.append({"threshold": t, "alerts_per_patient_day": alerts_per_day, "missed_fraction": missed_frac})
    return pd.DataFrame(rows)


def baseline_false_alarm_breakdown(df_raw: pd.DataFrame):
    df_alerted = apply_baseline_rule(df_raw)
    false_alarms = df_alerted[df_alerted["alert"] & ~df_alerted["deteriorating"]]
    causes = false_alarms.apply(classify_false_alarm_cause, axis=1)
    return causes.value_counts()


def normalization_ablation(result_norm: pd.DataFrame, result_raw: pd.DataFrame):
    ap_norm = average_precision_score(result_norm["y_true"], result_norm["prob_gbt"])
    ap_raw = average_precision_score(result_raw["y_true"], result_raw["prob_gbt"])
    ppv_norm, _ = ppv_at_sensitivity(result_norm["y_true"], result_norm["prob_gbt"])
    ppv_raw, _ = ppv_at_sensitivity(result_raw["y_true"], result_raw["prob_gbt"])
    return {
        "gbt_ap_normalized": ap_norm,
        "gbt_ap_raw": ap_raw,
        "gbt_ppv_at_90_normalized": ppv_norm,
        "gbt_ppv_at_90_raw": ppv_raw,
    }
