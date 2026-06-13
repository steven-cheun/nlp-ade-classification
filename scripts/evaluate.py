import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")


def compute_metrics(y_true, y_pred, label_names=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(y_true, y_pred, target_names=label_names),
    }


def plot_confusion_matrix(y_true, y_pred, label_names, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def error_analysis(texts, y_true, y_pred, label_names=None):
    errors = [
        {
            "text": texts[i],
            "true": label_names[y_true[i]] if label_names else y_true[i],
            "predicted": label_names[y_pred[i]] if label_names else y_pred[i],
        }
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]
    return pd.DataFrame(errors)


def compare_models(baseline_metrics, biobert_metrics):
    rows = []
    for key in ["accuracy", "f1_weighted", "f1_macro"]:
        rows.append({
            "metric": key,
            "TF-IDF + LR": round(baseline_metrics[key], 4),
            "BioBERT": round(biobert_metrics[key], 4),
        })
    return pd.DataFrame(rows)
