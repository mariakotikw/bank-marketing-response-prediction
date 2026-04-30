from pathlib import Path


RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR = ROOT_DIR / "models"

DATA_PATH = RAW_DATA_DIR / "bank-full.csv"

MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
TOP_K_ANALYSIS_PATH = REPORTS_DIR / "top_k_analysis.csv"

MODEL_PATH = MODELS_DIR / "gradient_boosting_response_model.joblib"

TARGET_COLUMN = "target"
ORIGINAL_TARGET_COLUMN = "y"
LEAKAGE_COLUMN = "duration"