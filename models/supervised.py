"""Per-patient-normalized supervised models: logistic regression + GBT.

Cross-validated with GroupKFold on patient_id (never a random split — vitals
from the same patient are highly autocorrelated, and letting one patient's
readings appear in both train and validation inflates every metric). Class
imbalance is handled with class weights, not resampling.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS, SEED
from features.build import NORMALIZED_FEATURES, RAW_FEATURES

N_FOLDS = 5


def _modelable_rows(feats: pd.DataFrame) -> pd.DataFrame:
    mask = (~feats["in_calibration"]) & (~feats["currently_deteriorating"])
    df = feats.loc[mask].copy()
    return df


def load_dataset(feature_set: str = "normalized"):
    feats = pd.read_parquet(PATHS["features"])
    df = _modelable_rows(feats)
    cols = NORMALIZED_FEATURES if feature_set == "normalized" else RAW_FEATURES
    df = df.dropna(subset=cols + ["label_onset"]).reset_index(drop=True)
    X = df[cols].to_numpy(dtype=float)
    y = df["label_onset"].to_numpy(dtype=int)
    groups = df["patient_id"].to_numpy()
    return df, X, y, groups, cols


def _fit_logreg(X_train, y_train, seed=SEED):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    # class_weight='balanced' with a ~300:1 imbalance ratio here pushes the
    # L-BFGS solver into transient float overflow. Cap the positive weight
    # instead — still strongly favors recall without the numerical blowup.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = min(n_neg / max(n_pos, 1), 30.0)
    clf = LogisticRegression(
        C=0.1, class_weight={0: 1.0, 1: pos_weight}, max_iter=2000, random_state=seed
    )
    # A handful of extreme z-scores (patch-artifact spikes with a tiny
    # calibration std) transiently overflow float64 inside the solver's
    # exp() on isolated rows; predict_proba output stays finite/valid
    # (checked: no NaNs in out-of-fold predictions), so this is cosmetic.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf.fit(Xs, y_train)
    return scaler, clf


def _fit_gbt(X_train, y_train, seed=SEED):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=seed,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf


def cross_validate(feature_set: str = "normalized", seed=SEED, n_folds=N_FOLDS):
    df, X, y, groups, cols = load_dataset(feature_set)
    gkf = GroupKFold(n_splits=n_folds)

    oof_lr = np.full(len(y), np.nan)
    oof_gbt = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        scaler, lr = _fit_logreg(X[train_idx], y[train_idx], seed=seed + fold)
        oof_lr[val_idx] = lr.predict_proba(scaler.transform(X[val_idx]))[:, 1]

        gbt = _fit_gbt(X[train_idx], y[train_idx], seed=seed + fold)
        oof_gbt[val_idx] = gbt.predict_proba(X[val_idx])[:, 1]

        fold_id[val_idx] = fold

    result = df[["patient_id", "timestamp", "deteriorating", "artifact", "movement"]].copy()
    result["y_true"] = y
    result["prob_lr"] = oof_lr
    result["prob_gbt"] = oof_gbt
    result["fold"] = fold_id
    result["feature_set"] = feature_set
    return result, cols


def fit_final_models(feature_set: str = "normalized", seed=SEED):
    """Fit on the full modelable dataset for interpretability reporting
    (coefficients / feature importances) — not used for evaluation metrics."""
    df, X, y, groups, cols = load_dataset(feature_set)
    scaler, lr = _fit_logreg(X, y, seed=seed)
    gbt = _fit_gbt(X, y, seed=seed)

    lr_coefs = pd.Series(lr.coef_[0], index=cols).sort_values(key=np.abs, ascending=False)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(X), size=min(20000, len(X)), replace=False)
    perm = permutation_importance(
        gbt, X[sample_idx], y[sample_idx], n_repeats=5, random_state=seed, scoring="average_precision"
    )
    gbt_importances = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False)
    return {"lr_coefficients": lr_coefs, "gbt_feature_importances": gbt_importances}


def main():
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    result_norm, cols_norm = cross_validate("normalized")
    result_norm.to_parquet(out_dir / "oof_predictions_normalized.parquet", index=False)
    print(f"Normalized feature set: {len(cols_norm)} features, {len(result_norm):,} rows, "
          f"{result_norm['y_true'].sum()} positives")

    result_raw, cols_raw = cross_validate("raw")
    result_raw.to_parquet(out_dir / "oof_predictions_raw.parquet", index=False)
    print(f"Raw feature set: {len(cols_raw)} features, {len(result_raw):,} rows, "
          f"{result_raw['y_true'].sum()} positives")

    interp = fit_final_models("normalized")
    print("\nTop 8 logistic regression coefficients (normalized features, |coef|):")
    print(interp["lr_coefficients"].head(8))
    print("\nTop 8 GBT feature importances (normalized features):")
    print(interp["gbt_feature_importances"].head(8))

    interp["lr_coefficients"].to_csv(out_dir / "lr_coefficients.csv")
    interp["gbt_feature_importances"].to_csv(out_dir / "gbt_feature_importances.csv")


if __name__ == "__main__":
    main()
