from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import torch
import numpy as np


MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model(num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


def tokenize_dataset(texts, labels, tokenizer, max_length: int = 128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    dataset = Dataset.from_dict({**encodings, "labels": labels})
    return dataset


def get_training_args(output_dir: str, epochs: int = 3, lr: float = 2e-5):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )


def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, accuracy_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, average="weighted"),
        "accuracy": accuracy_score(labels, preds),
    }


def train(train_dataset, val_dataset, model, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
