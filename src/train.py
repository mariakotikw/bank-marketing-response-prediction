import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from config import (
    RANDOM_STATE,
    DATA_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODEL_PATH,
    MODEL_COMPARISON_PATH,
    TOP_K_ANALYSIS_PATH,
)
from preprocessing import (
    load_data,
    prepare_target,
    remove_leakage_features,
    split_features_target,
    make_train_test_split,
    build_preprocessor,
)
from evaluate import calculate_metrics, build_targeting_table, top_k_analysis


def build_logistic_regression_pipeline(preprocessor):
    """Build Logistic Regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(preprocessor):
    """Build Random Forest pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_gradient_boosting_pipeline(preprocessor):
    """Build Gradient Boosting pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_model_comparison_plot(results: pd.DataFrame):
    """Save model comparison plot by PR-AUC."""
    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=results,
        x="model",
        y="pr_auc",
    )

    plt.title("Model comparison by PR-AUC")
    plt.xlabel("Model")
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_pr_auc.png", dpi=300)
    plt.close()


def save_top_k_response_rate_plot(top_k_results: pd.DataFrame):
    """Save response rate chart for top-k client groups."""
    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=top_k_results,
        x="top_k_percent",
        y="response_rate",
    )

    plt.axhline(
        top_k_results["baseline_response_rate"].iloc[0],
        linestyle="--",
        label="baseline response rate",
    )

    plt.title("Response rate in top-k client groups")
    plt.xlabel("Top-k clients, %")
    plt.ylabel("Response rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_k_response_rate.png", dpi=300)
    plt.close()


def save_top_k_lift_plot(top_k_results: pd.DataFrame):
    """Save lift chart for top-k client groups."""
    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=top_k_results,
        x="top_k_percent",
        y="lift",
    )

    plt.axhline(
        1,
        linestyle="--",
        label="random targeting",
    )

    plt.title("Lift in top-k client groups")
    plt.xlabel("Top-k clients, %")
    plt.ylabel("Lift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_k_lift.png", dpi=300)
    plt.close()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Target distribution:")
    print(df["y"].value_counts(normalize=True))

    df = prepare_target(df)
    df_model = remove_leakage_features(df)

    X, y = split_features_target(df_model)

    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": build_logistic_regression_pipeline(preprocessor),
        "Random Forest": build_random_forest_pipeline(preprocessor),
        "Gradient Boosting": build_gradient_boosting_pipeline(preprocessor),
    }

    metrics_rows = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        metrics_rows[model_name] = calculate_metrics(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
        )

    results = pd.DataFrame(metrics_rows.values())
    results.to_csv(MODEL_COMPARISON_PATH, index=False)

    print("\nModel comparison:")
    print(results)

    best_model_name = "Gradient Boosting"
    best_model = models[best_model_name]

    print(f"\nSelected model for top-k targeting: {best_model_name}")

    targeting_df = build_targeting_table(best_model, X_test, y_test)
    top_k_results = top_k_analysis(targeting_df)

    top_k_results.to_csv(TOP_K_ANALYSIS_PATH, index=False)

    print("\nTop-k analysis:")
    print(top_k_results)

    print("\nSaving figures...")
    save_model_comparison_plot(results)
    save_top_k_response_rate_plot(top_k_results)
    save_top_k_lift_plot(top_k_results)

    print("\nSaving final model...")
    joblib.dump(best_model, MODEL_PATH)

    print("\nDone.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Model comparison saved to: {MODEL_COMPARISON_PATH}")
    print(f"Top-k analysis saved to: {TOP_K_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()