"""Per-patient normalization and derived features.

Every rolling/trend feature is backward-looking only (no `center=True`,
no reliance on future rows) — see `_check_no_lookahead` for a runnable
verification. The first 24h calibration window per patient defines the
per-patient baseline and is excluded from training/evaluation, since those
rows are what the baseline was computed from.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CALIBRATION_SAMPLES, HORIZON_HOURS, PATHS, SAMPLE_MINUTES

VITALS = ["heart_rate", "temperature", "resp_rate", "movement"]

SAMPLES_PER_HOUR = 60 // SAMPLE_MINUTES
WINDOW_1H = 1 * SAMPLES_PER_HOUR
WINDOW_4H = 4 * SAMPLES_PER_HOUR
HORIZON_SAMPLES = HORIZON_HOURS * SAMPLES_PER_HOUR

NORMALIZED_FEATURES = (
    [f"{c}_z" for c in VITALS]
    + [f"{c}_trend_1h" for c in VITALS]
    + [f"{c}_trend_4h" for c in VITALS]
    + ["cross_resp_move", "resp_persistence", "contact_quality_roll_1h", "implausible_roll_1h"]
)

RAW_FEATURES = (
    VITALS
    + [f"{c}_raw_trend_1h" for c in VITALS]
    + [f"{c}_raw_trend_4h" for c in VITALS]
    + ["cross_resp_move_raw", "resp_persistence_raw", "contact_quality_roll_1h", "implausible_roll_1h"]
)

IMPLAUSIBLE_BOUNDS = {
    "heart_rate": (25, 220),
    "temperature": (33, 42),
    "resp_rate": (3, 65),
}


def _consecutive_run_length(flag: pd.Series) -> pd.Series:
    """Backward-looking count of consecutive True values ending at each row."""
    flag = flag.fillna(False)
    groups = (flag != flag.shift()).cumsum()
    run = flag.groupby(groups).cumcount() + 1
    return run.where(flag, 0)


def _build_patient_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").reset_index(drop=True)
    n = len(g)

    calib = g.iloc[:CALIBRATION_SAMPLES]
    g["in_calibration"] = False
    g.loc[g.index[:CALIBRATION_SAMPLES], "in_calibration"] = True

    baseline = {}
    for col in VITALS:
        mu = calib[col].mean()
        sd = calib[col].std()
        sd = sd if (sd is not None and sd > 1e-6) else 1.0
        baseline[col] = (mu, sd)
        g[f"{col}_z"] = (g[col] - mu) / sd

    for col in VITALS:
        z = g[f"{col}_z"]
        g[f"{col}_trend_1h"] = z.diff(WINDOW_1H) / 1.0
        g[f"{col}_trend_4h"] = z.diff(WINDOW_4H) / 4.0
        # Raw-scale counterparts, for the with/without-normalization ablation.
        raw = g[col]
        g[f"{col}_raw_trend_1h"] = raw.diff(WINDOW_1H) / 1.0
        g[f"{col}_raw_trend_4h"] = raw.diff(WINDOW_4H) / 4.0

    g["cross_resp_move"] = g["resp_rate_z"] - g["movement_z"]
    g["cross_resp_move_raw"] = g["resp_rate"] - g["movement"]

    resp_dev_flag = g["resp_rate_z"].abs() > 1.0
    g["resp_persistence"] = _consecutive_run_length(resp_dev_flag)
    # Raw-scale persistence uses a fixed population cutoff (no per-patient
    # calibration at all), mirroring the baseline rule's thresholds.
    resp_dev_flag_raw = (g["resp_rate"] > 20) | (g["resp_rate"] < 10)
    g["resp_persistence_raw"] = _consecutive_run_length(resp_dev_flag_raw)

    g["contact_quality_roll_1h"] = g["contact_quality"].rolling(WINDOW_1H, min_periods=1).mean()

    implausible = pd.Series(False, index=g.index)
    for col, (lo, hi) in IMPLAUSIBLE_BOUNDS.items():
        implausible |= (g[col] < lo) | (g[col] > hi) | g[col].isna()
    g["implausible_flag"] = implausible
    g["implausible_roll_1h"] = implausible.rolling(WINDOW_1H, min_periods=1).mean()

    # Forward-looking LABEL only (not a feature): does deterioration begin
    # within the next HORIZON_HOURS, given the patient is not already
    # deteriorating right now?
    det = g["deteriorating"].astype(bool)
    shifted = det.shift(-1, fill_value=False)
    future_onset = shifted[::-1].rolling(HORIZON_SAMPLES, min_periods=1).max()[::-1]
    g["label_onset"] = future_onset.astype(bool).to_numpy()
    g["currently_deteriorating"] = det

    return g


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = [_build_patient_features(g) for _, g in df.groupby("patient_id", sort=True)]
    return pd.concat(parts, ignore_index=True)


def _check_no_lookahead():
    """Sanity check: perturbing a future value must not change any feature
    computed at or before the current row."""
    rng = np.random.default_rng(0)
    n = CALIBRATION_SAMPLES + 200
    t = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(n) * SAMPLE_MINUTES, unit="m")
    base = pd.DataFrame(
        {
            "patient_id": 0,
            "timestamp": t,
            "heart_rate": 75 + rng.normal(0, 2, n),
            "temperature": 36.8 + rng.normal(0, 0.1, n),
            "resp_rate": 16 + rng.normal(0, 1, n),
            "movement": 0.3 + rng.normal(0, 0.05, n),
            "contact_quality": 1.0,
            "deteriorating": False,
            "artifact": False,
        }
    )
    perturb_idx = n - 5
    check_idx = n - 10  # strictly before the perturbation

    feats_a = _build_patient_features(base.copy())
    perturbed = base.copy()
    perturbed.loc[perturb_idx, "heart_rate"] += 1000  # huge future change
    perturbed.loc[perturb_idx, "deteriorating"] = True
    feats_b = _build_patient_features(perturbed)

    feature_cols = [c for c in feats_a.columns if c not in ("label_onset",)]
    row_a = feats_a.loc[check_idx, feature_cols]
    row_b = feats_b.loc[check_idx, feature_cols]
    mismatches = [c for c in feature_cols if not _safe_eq(row_a[c], row_b[c])]
    assert not mismatches, f"Lookahead leakage detected in features: {mismatches}"
    print("No-lookahead check passed: features at row t are unaffected by changes after t.")


def _safe_eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def main():
    _check_no_lookahead()
    df = pd.read_parquet(PATHS["raw"])
    feats = build_features(df)
    out_path = Path(PATHS["features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    n_modelable = ((~feats["in_calibration"]) & (~feats["currently_deteriorating"]) & feats["heart_rate"].notna()).sum()
    print(f"Wrote {len(feats):,} feature rows to {out_path}")
    print(f"Rows usable for modeling (post-calibration, not currently deteriorating, no NaN vitals): {n_modelable:,}")
    print(f"Positive rate among modelable rows: {feats.loc[(~feats['in_calibration']) & (~feats['currently_deteriorating']) & feats['heart_rate'].notna(), 'label_onset'].mean():.4f}")


if __name__ == "__main__":
    main()
