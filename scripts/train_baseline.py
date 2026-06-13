import pandas as pd
from pathlib import Path
from src.models.baseline_model import run_grid_search, save_pipeline
from scripts.evaluate import compute_metrics, plot_confusion_matrix, error_analysis, FIGURES_DIR, RESULTS_DIR

DATA_DIR = Path("data/splits")
MODEL_OUT = Path("models/baseline/tfidf_pipeline.joblib")
LABEL_NAMES = ["negative", "positive"]


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    print("Running grid search on TF-IDF + Logistic Regression...")
    best_pipeline, best_params = run_grid_search(train["text"].tolist(), train["label"].tolist())
    print(f"Best params: {best_params}")

    val_preds = best_pipeline.predict(val["text"].tolist())
    val_metrics = compute_metrics(val["label"].tolist(), val_preds, LABEL_NAMES)
    print("Validation results:")
    print(val_metrics["report"])

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    save_pipeline(best_pipeline, MODEL_OUT)
    print(f"Pipeline saved to {MODEL_OUT}")

    test_preds = best_pipeline.predict(test["text"].tolist())
    test_metrics = compute_metrics(test["label"].tolist(), test_preds, LABEL_NAMES)
    print("Test results:")
    print(test_metrics["report"])

    plot_confusion_matrix(
        test["label"].tolist(), test_preds, LABEL_NAMES,
        title="TF-IDF + LR — Test Confusion Matrix",
        save_path=FIGURES_DIR / "baseline_cm.png",
    )

    errors = error_analysis(test["text"].tolist(), test["label"].tolist(), test_preds, LABEL_NAMES)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(RESULTS_DIR / "baseline_errors.csv", index=False)
    print(f"Error analysis saved ({len(errors)} misclassified samples)")


if __name__ == "__main__":
    main()
