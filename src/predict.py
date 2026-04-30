import joblib
import pandas as pd

from config import MODEL_PATH, DATA_PATH


def load_model():
    """Load trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run `python src/train.py` first."
        )

    return joblib.load(MODEL_PATH)


def predict_response_probability(client_data: dict) -> dict:
    """Predict response probability for one client."""
    model = load_model()

    X = pd.DataFrame([client_data])
    response_probability = float(model.predict_proba(X)[:, 1][0])

    return {
        "response_probability": round(response_probability, 6),
    }


def load_example_client() -> dict:
    """
    Load one example client from raw dataset.

    Columns y and duration are removed:
    - y is the target;
    - duration is leakage-prone for pre-campaign targeting.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Put bank-full.csv into data/raw/bank-full.csv"
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    example = df.drop(columns=["y", "duration"]).iloc[0].to_dict()

    return example


if __name__ == "__main__":
    example_client = load_example_client()
    prediction = predict_response_probability(example_client)

    print("Example client:")
    print(example_client)
    print("\nPrediction:")
    print(prediction)