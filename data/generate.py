"""Synthetic cohort generator for the Noma alert-triage prototype.

Simulates ~200 patients over 7 days at 5-minute resolution across four vitals
(heart rate, temperature, respiratory rate, movement) plus a contact-quality
channel. Deliberately makes deterioration events overlap in magnitude with
benign activity episodes on any single channel, so that separating them
requires trend / persistence / cross-signal information rather than a
single-vital threshold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ARTIFACT_FRACTION,
    CALIBRATION_HOURS,
    DAYS,
    DETERIORATION_FRACTION,
    N_PATIENTS,
    N_SAMPLES,
    PATHS,
    SAMPLE_MINUTES,
    SEED,
)

TOTAL_HOURS = DAYS * 24


def _hann_window(n):
    if n <= 1:
        return np.ones(max(n, 1))
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def _add_activity_episodes(prng, t_hours, hr, resp, movement):
    """Benign episodes: movement rises, HR and resp rise with it."""
    n = len(t_hours)
    n_episodes = prng.poisson(5 * DAYS)  # ~5/day average
    samples_per_hour = 60 // SAMPLE_MINUTES
    for _ in range(n_episodes):
        start = prng.integers(0, n)
        duration_min = prng.uniform(15, 60)
        duration_samples = max(2, int(duration_min / SAMPLE_MINUTES))
        end = min(n, start + duration_samples)
        length = end - start
        if length < 2:
            continue
        window = _hann_window(length)
        move_peak = prng.uniform(0.3, 0.85)
        hr_peak = move_peak * prng.uniform(15, 32) + prng.normal(0, 2)
        resp_peak = move_peak * prng.uniform(4, 10) + prng.normal(0, 0.5)
        movement[start:end] += move_peak * window
        hr[start:end] += hr_peak * window
        resp[start:end] += resp_peak * window
    return None


def _add_deterioration(prng, t_hours, hr, resp, movement, move_base):
    """Gradual multi-hour drift: resp climbs, movement falls.

    Amplitudes are drawn from ranges that overlap with the benign activity
    episode ranges above, so single-channel thresholds cannot cleanly
    separate deterioration from benign activity.
    """
    n = len(t_hours)
    earliest = CALIBRATION_HOURS
    latest = TOTAL_HOURS - 4
    if latest <= earliest:
        return np.zeros(n, dtype=bool)
    onset_hour = prng.uniform(earliest, latest)
    duration_hours = prng.uniform(3, 10)
    duration_hours = min(duration_hours, TOTAL_HOURS - onset_hour)
    onset_idx = int(onset_hour * 60 / SAMPLE_MINUTES)
    duration_samples = max(4, int(duration_hours * 60 / SAMPLE_MINUTES))
    end_idx = min(n, onset_idx + duration_samples)
    length = end_idx - onset_idx
    if length < 4:
        return np.zeros(n, dtype=bool)

    ramp = np.linspace(0, 1, length) ** 1.3  # gradual, slightly convex onset

    resp_peak = prng.uniform(3.5, 9.5)  # overlaps activity's 4-10 range
    move_drop = prng.uniform(0.15, 0.35) * max(move_base, 0.15)
    hr_peak = prng.uniform(2, 12)  # overlaps low end of activity's HR range

    resp[onset_idx:end_idx] += resp_peak * ramp
    movement[onset_idx:end_idx] -= move_drop * ramp
    hr[onset_idx:end_idx] += hr_peak * ramp

    # Subtle, unlabeled prodrome before the officially recognized onset: real
    # deterioration usually shows faint precursor drift before it crosses
    # whatever threshold gets it clinically labeled. Without this, "predict
    # onset within N hours" has literally nothing to learn from, since
    # nothing in the signal differs before onset_idx. This prodrome is
    # smaller in amplitude than the labeled event and NOT marked as
    # `deteriorating` — it is what the model has to learn to pick up on.
    precursor_hours = prng.uniform(4, 7)
    precursor_start_idx = max(0, onset_idx - int(precursor_hours * 60 / SAMPLE_MINUTES))
    precursor_len = onset_idx - precursor_start_idx
    if precursor_len > 2:
        precursor_ramp = np.linspace(0, 1, precursor_len) ** 1.5
        prodrome_frac = prng.uniform(0.45, 0.8)
        resp[precursor_start_idx:onset_idx] += resp_peak * prodrome_frac * precursor_ramp
        movement[precursor_start_idx:onset_idx] -= move_drop * prodrome_frac * precursor_ramp
        hr[precursor_start_idx:onset_idx] += hr_peak * prodrome_frac * precursor_ramp

    label = np.zeros(n, dtype=bool)
    label[onset_idx:end_idx] = True
    return label


def _add_artifact(prng, t_hours, hr, temp, resp, movement, contact_quality):
    """Progressive patch contact loss: rising noise, dropouts, possible removal."""
    n = len(t_hours)
    onset_hour = prng.uniform(0, TOTAL_HOURS - 6)
    onset_idx = int(onset_hour * 60 / SAMPLE_MINUTES)
    duration_hours = prng.uniform(4, TOTAL_HOURS - onset_hour)
    duration_samples = max(4, int(duration_hours * 60 / SAMPLE_MINUTES))
    end_idx = min(n, onset_idx + duration_samples)
    length = end_idx - onset_idx
    if length < 4:
        return np.zeros(n, dtype=bool)

    decay = np.linspace(1, 0, length)  # contact quality decays toward 0
    full_removal = prng.random() < 0.5
    removal_frac = prng.uniform(0.5, 0.85) if full_removal else 1.1  # >1 => never fully removed
    removal_idx_local = int(length * removal_frac)

    contact_quality[onset_idx:end_idx] = np.clip(decay, 0, 1)
    noise_scale = (1 - decay) * 4  # grows as contact worsens

    hr[onset_idx:end_idx] += prng.normal(0, noise_scale)
    temp[onset_idx:end_idx] += prng.normal(0, noise_scale * 0.05)
    resp[onset_idx:end_idx] += prng.normal(0, noise_scale * 0.5)
    movement[onset_idx:end_idx] += prng.normal(0, noise_scale * 0.1)

    # Occasional physiologically implausible spikes as contact degrades.
    spike_prob = np.clip((1 - decay) * 0.05, 0, 0.3)
    spikes = prng.random(length) < spike_prob
    hr[onset_idx:end_idx] = np.where(spikes, prng.choice([0, 220, 250]), hr[onset_idx:end_idx])

    if full_removal and removal_idx_local < length:
        removed_start = onset_idx + removal_idx_local
        contact_quality[removed_start:end_idx] = 0.0
        hr[removed_start:end_idx] = np.nan
        temp[removed_start:end_idx] = np.nan
        resp[removed_start:end_idx] = np.nan
        movement[removed_start:end_idx] = np.nan

    flag = np.zeros(n, dtype=bool)
    flag[onset_idx:end_idx] = True
    return flag


def _generate_patient(pid, prng, t_hours):
    n = len(t_hours)

    hr_base = np.clip(prng.normal(75, 12), 50, 110)
    temp_base = np.clip(prng.normal(36.8, 0.4), 35.8, 37.8)
    resp_base = np.clip(prng.normal(16, 3.5), 10, 24)
    move_base = np.clip(prng.normal(0.35, 0.15), 0.05, 0.7)

    phase = prng.uniform(0, 2 * np.pi)
    hr_circadian_amp = max(prng.normal(6, 1.5), 0.5)
    temp_circadian_amp = max(prng.normal(0.4, 0.1), 0.05)

    circadian_hr = hr_circadian_amp * np.sin(2 * np.pi * t_hours / 24 + phase)
    circadian_temp = temp_circadian_amp * np.sin(2 * np.pi * t_hours / 24 + phase)

    hr = hr_base + circadian_hr + prng.normal(0, 2.5, n)
    temp = temp_base + circadian_temp + prng.normal(0, 0.15, n)
    resp = resp_base + prng.normal(0, 1.2, n)
    movement = np.clip(move_base + prng.normal(0, 0.05, n), 0, None)

    contact_quality = np.ones(n)

    _add_activity_episodes(prng, t_hours, hr, resp, movement)

    deteriorating = np.zeros(n, dtype=bool)
    if prng.random() < DETERIORATION_FRACTION:
        deteriorating = _add_deterioration(prng, t_hours, hr, resp, movement, move_base)

    artifact = np.zeros(n, dtype=bool)
    if prng.random() < ARTIFACT_FRACTION:
        artifact = _add_artifact(prng, t_hours, hr, temp, resp, movement, contact_quality)

    movement = np.clip(movement, 0, None)

    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(t_hours, unit="h")

    return pd.DataFrame(
        {
            "patient_id": pid,
            "timestamp": timestamps,
            "heart_rate": hr,
            "temperature": temp,
            "resp_rate": resp,
            "movement": movement,
            "contact_quality": contact_quality,
            "deteriorating": deteriorating,
            "artifact": artifact,
        }
    )


def generate_cohort(seed=SEED, n_patients=N_PATIENTS):
    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, 2**32 - 1, size=n_patients)

    t_hours = np.arange(N_SAMPLES) * SAMPLE_MINUTES / 60.0

    frames = []
    for pid in range(n_patients):
        prng = np.random.default_rng(child_seeds[pid])
        frames.append(_generate_patient(pid, prng, t_hours))
    return pd.concat(frames, ignore_index=True)


def _plot_overlap_check(df, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = df[~df["artifact"]]
    vitals = ["heart_rate", "temperature", "resp_rate", "movement"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, col in zip(axes, vitals):
        det = clean.loc[clean.deteriorating, col].dropna()
        non = clean.loc[~clean.deteriorating, col].dropna()
        ax.hist(non, bins=60, density=True, alpha=0.5, label="non-deteriorating")
        ax.hist(det, bins=60, density=True, alpha=0.5, label="deteriorating")
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Raw vital distributions: deteriorating vs non-deteriorating windows\n"
                 "(generator is intentionally overlapping — no single channel cleanly separates the classes)")
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "generator_overlap_check.png", dpi=120)
    plt.close(fig)


def main():
    df = generate_cohort()
    out_path = Path(PATHS["raw"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    n_det_patients = df.groupby("patient_id")["deteriorating"].any().sum()
    n_art_patients = df.groupby("patient_id")["artifact"].any().sum()
    print(f"Wrote {len(df):,} rows for {df.patient_id.nunique()} patients to {out_path}")
    print(f"Patients with a deterioration event: {n_det_patients} ({n_det_patients/df.patient_id.nunique():.1%})")
    print(f"Patients with an artifact episode: {n_art_patients} ({n_art_patients/df.patient_id.nunique():.1%})")
    print(f"NaN rows (patch removed): {df['heart_rate'].isna().sum():,}")
    _plot_overlap_check(df, PATHS["report_dir"])
    print(f"Wrote overlap-check figure to {PATHS['report_dir']}/generator_overlap_check.png")


if __name__ == "__main__":
    main()
