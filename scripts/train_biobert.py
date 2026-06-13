import pandas as pd
from pathlib import Path
from src.models.biobert_model import (
    load_tokenizer,
    load_model,
    tokenize_dataset,
    get_training_args,
    train,
)
from scripts.evaluate import compute_metrics, plot_confusion_matrix, error_analysis, FIGURES_DIR, RESULTS_DIR
import numpy as np

DATA_DIR = Path("data/splits")
MODEL_OUT = Path("models/biobert_finetuned")
LABEL_NAMES = ["negative", "positive"]
MAX_LENGTH = 128
EPOCHS = 3
LR = 2e-5


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    tokenizer = load_tokenizer()
    num_labels = len(LABEL_NAMES)
    model = load_model(num_labels)

    train_dataset = tokenize_dataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LENGTH)
    val_dataset = tokenize_dataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LENGTH)
    test_dataset = tokenize_dataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, MAX_LENGTH)

    training_args = get_training_args(str(MODEL_OUT), epochs=EPOCHS, lr=LR)

    print("Fine-tuning BioBERT...")
    trainer = train(train_dataset, val_dataset, model, training_args)

    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

    predictions = trainer.predict(test_dataset)
    test_preds = np.argmax(predictions.predictions, axis=-1)
    test_metrics = compute_metrics(test_df["label"].tolist(), test_preds.tolist(), LABEL_NAMES)
    print("Test results:")
    print(test_metrics["report"])

    plot_confusion_matrix(
        test_df["label"].tolist(), test_preds.tolist(), LABEL_NAMES,
        title="BioBERT — Test Confusion Matrix",
        save_path=FIGURES_DIR / "biobert_cm.png",
    )

    errors = error_analysis(test_df["text"].tolist(), test_df["label"].tolist(), test_preds.tolist(), LABEL_NAMES)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(RESULTS_DIR / "biobert_errors.csv", index=False)
    print(f"Error analysis saved ({len(errors)} misclassified samples)")


if __name__ == "__main__":
    main()
