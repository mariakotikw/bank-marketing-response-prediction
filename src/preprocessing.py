from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_STATE, ORIGINAL_TARGET_COLUMN, TARGET_COLUMN, LEAKAGE_COLUMN


def load_data(path: str | Path) -> pd.DataFrame:
    """Load Bank Marketing dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Put bank-full.csv into data/raw/bank-full.csv"
        )

    return pd.read_csv(path, sep=";")


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert target column from yes/no to 1/0."""
    df = df.copy()

    if ORIGINAL_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column '{ORIGINAL_TARGET_COLUMN}' was not found.")

    df[TARGET_COLUMN] = df[ORIGINAL_TARGET_COLUMN].map({"no": 0, "yes": 1})

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains unexpected values.")

    return df


def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove target string column and leakage-prone duration feature.

    Duration is known only after the call, so it should not be used
    for pre-campaign customer targeting.
    """
    columns_to_drop = [ORIGINAL_TARGET_COLUMN]

    if LEAKAGE_COLUMN in df.columns:
        columns_to_drop.append(LEAKAGE_COLUMN)

    return df.drop(columns=columns_to_drop)


def split_features_target(df: pd.DataFrame):
    """Split dataframe into features and target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column '{TARGET_COLUMN}' was not found.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def make_train_test_split(X, y, test_size: float = 0.2):
    """Make stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def get_feature_types(X: pd.DataFrame):
    """Return numeric and categorical feature names."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
    numeric_features, categorical_features = get_feature_types(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor