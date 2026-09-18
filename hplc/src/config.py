from pathlib import Path

HPLC_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = HPLC_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MAPPING_DIR = DATA_DIR / "mapping"
OUTPUTS_DIR = HPLC_DIR / "outputs"

HPLC_RAW_PATH = RAW_DIR / "hplc.xlsx"
MAPPING_PATH = MAPPING_DIR / "sample_mapping.csv"

SHEETS = {
    "440": "Crocin 440",
    "250": "picrocrocin 250",
    "308": "safranal 308",
}

TIME_MIN = 0.01666667
TIME_MAX = 30.0
EXPECTED_TIME_POINTS = 1800

CROCIN_WINDOW = (14.0, 30.0)
PICROCROCCIN_REFERENCE = 14.3
SAFRANAL_REFERENCE = 30.9

# Candidate baseline-correction settings used by the computational pipeline.
# These are analysis parameters, not claims about the laboratory method.
ASLS_LAMBDA = 1e6
ASLS_P = 0.01
ASLS_ITERATIONS = 10

# Robust HPLC pilot evaluation uses PLS-DA directly on the preprocessed
# chromatogram. PCA is intentionally not applied inside this pilot because
# the training folds are very small (40 independent observations / 11 groups).
PLS_COMPONENTS = 4

CV_REPEATS = 20
CV_N_SPLITS = 2
RANDOM_STATE = 42
