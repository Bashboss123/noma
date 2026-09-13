"""Shared constants for the Noma alert-triage prototype."""

SEED = 42

N_PATIENTS = 200
DAYS = 7
SAMPLE_MINUTES = 5
SAMPLES_PER_DAY = 24 * 60 // SAMPLE_MINUTES
N_SAMPLES = DAYS * SAMPLES_PER_DAY

CALIBRATION_HOURS = 24
CALIBRATION_SAMPLES = CALIBRATION_HOURS * 60 // SAMPLE_MINUTES

# Fraction of patients who experience a deterioration event.
DETERIORATION_FRACTION = 0.10

# Fraction of patients who experience a sensor-artifact episode.
ARTIFACT_FRACTION = 0.20

# Prediction horizon: does deterioration onset occur within this many hours?
HORIZON_HOURS = 6

PATHS = dict(
    raw="data/cohort.parquet",
    features="features/features.parquet",
    report_dir="report",
)
