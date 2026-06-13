from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/splits")
DATASET_NAME = "ade-benchmark-corpus/ade_corpus_v2"
SUBSET = "Ade_corpus_v2_classification"
SEED = 42


def print_distribution(name, df):
    total = len(df)
    counts = df["label"].value_counts().sort_index()
    print(f"\n{name} ({total} samples):")
    labels = {0: "non-ADE", 1: "ADE    "}
    for label, count in counts.items():
        print(f"  label {label} ({labels[label]}): {count:>5}  ({count / total * 100:.1f}%)")


def main():
    print(f"Loading {SUBSET} from {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, SUBSET)
    df = pd.DataFrame(dataset["train"])
    print(f"Total samples: {len(df)}")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"\nSplits saved to {DATA_DIR}/")

    print("\n--- Class distribution per split ---")
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print_distribution(name, split)


if __name__ == "__main__":
    main()
