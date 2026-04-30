import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_metrics(model, X_test, y_test, model_name: str) -> dict:
    """Calculate model metrics at default threshold 0.5."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "model": model_name,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def get_confusion_matrix(model, X_test, y_test):
    """Return confusion matrix at threshold 0.5."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return confusion_matrix(y_test, y_pred)


def build_targeting_table(model, X_test, y_test) -> pd.DataFrame:
    """
    Build table with predicted response probabilities.

    The table is sorted by response_probability in descending order.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    targeting_df = X_test.copy()
    targeting_df["true_target"] = y_test.values
    targeting_df["response_probability"] = y_proba

    targeting_df = targeting_df.sort_values(
        "response_probability",
        ascending=False,
    )

    return targeting_df


def top_k_analysis(
    targeting_df: pd.DataFrame,
    k_values=None,
) -> pd.DataFrame:
    """Calculate response rate and lift for top-k client groups."""
    if k_values is None:
        k_values = [0.05, 0.10, 0.20, 0.30]

    baseline_response_rate = targeting_df["true_target"].mean()
    rows = []

    for k in k_values:
        top_n = int(len(targeting_df) * k)
        top_clients = targeting_df.head(top_n)

        response_rate = top_clients["true_target"].mean()
        lift = response_rate / baseline_response_rate

        rows.append(
            {
                "top_k_percent": int(k * 100),
                "clients_selected": top_n,
                "response_rate": response_rate,
                "baseline_response_rate": baseline_response_rate,
                "lift": lift,
                "positive_responses": int(top_clients["true_target"].sum()),
            }
        )

    return pd.DataFrame(rows)