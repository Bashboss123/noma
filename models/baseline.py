"""Naive fixed-threshold alerting rule — the strawman representing Noma's
current alerting approach.

Fires whenever any vital crosses a fixed population-level threshold. No
smoothing, no persistence requirement, no contact-quality check. This is
intentionally crude: it is what the project is measured against.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

# Fixed population-level thresholds (not personalized).
THRESHOLDS = {
    "heart_rate": {"high": 100, "low": 50},
    "temperature": {"high": 38.0, "low": 35.5},
    "resp_rate": {"high": 20, "low": 10},
}


def apply_baseline_rule(df):
    """Return df with an `alert` column: True if any vital crosses threshold."""
    alert = pd.Series(False, index=df.index)
    for col, bounds in THRESHOLDS.items():
        vals = df[col]
        alert |= (vals > bounds["high"]) | (vals < bounds["low"])
    out = df.copy()
    out["alert"] = alert.fillna(False)
    return out


def classify_false_alarm_cause(row):
    """For a false-alarm row, attribute the cause: artifact, benign activity
    (high movement at time of alert), or baseline mismatch (persistently
    outside population threshold despite not deteriorating/artifact)."""
    if row["artifact"]:
        return "artifact"
    if row["movement"] > 0.5:
        return "benign_activity"
    return "baseline_mismatch"


def summarize(df_alerted):
    n_patient_days = df_alerted.groupby("patient_id")["timestamp"].apply(
        lambda s: (s.max() - s.min()).total_seconds() / 86400 + 1
    ).sum()
    n_alerts = df_alerted["alert"].sum()
    alerts_per_patient_day = n_alerts / n_patient_days

    false_alarms = df_alerted[df_alerted["alert"] & ~df_alerted["deteriorating"]]
    hits = df_alerted[df_alerted["alert"] & df_alerted["deteriorating"]]

    causes = false_alarms.apply(classify_false_alarm_cause, axis=1)
    cause_counts = causes.value_counts()

    # Per-patient-day "hit": did the patient have >=1 alert during a
    # deterioration window, for patients who actually deteriorated?
    det_patients = df_alerted[df_alerted["deteriorating"]]["patient_id"].unique()
    caught = 0
    for pid in det_patients:
        sub = df_alerted[(df_alerted.patient_id == pid) & df_alerted.deteriorating]
        if sub["alert"].any():
            caught += 1
    sensitivity_patient_level = caught / len(det_patients) if len(det_patients) else float("nan")

    return {
        "n_alerts": int(n_alerts),
        "n_patient_days": round(n_patient_days, 1),
        "alerts_per_patient_day": round(alerts_per_patient_day, 3),
        "n_false_alarms": int(len(false_alarms)),
        "n_hits": int(len(hits)),
        "false_alarm_cause_counts": cause_counts.to_dict(),
        "patient_level_sensitivity": round(sensitivity_patient_level, 3),
        "n_deteriorating_patients": int(len(det_patients)),
    }


def main():
    df = pd.read_parquet(PATHS["raw"])
    df_alerted = apply_baseline_rule(df)
    stats = summarize(df_alerted)
    print("Baseline threshold rule summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return df_alerted, stats


if __name__ == "__main__":
    main()
