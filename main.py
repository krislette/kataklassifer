"""
main.py

Entry point for the Gairaigo Origin Language Classifier.

This script orchestrates the full machine learning pipeline in eight steps:

  Step 1 — Load       : Parse JMdict XML into a (katakana, language) DataFrame.
  Step 2 — Preprocess : Remove duplicates, consolidate rare classes, encode labels.
  Step 3 — Featurize  : Build a TF-IDF character n-gram feature matrix.
  Step 4 — Split      : Divide data into training (80%) and test (20%) sets.
  Step 5 — Train      : Fit a LinearSVC classifier on the training features.
  Step 6 — Evaluate   : Compute accuracy, F1, and confusion matrix on the test set.
  Step 7 — Visualize  : Save all charts to output/plots/.
  Step 8 — Export     : Write per-word predictions to output/results/.

Usage:
  python main.py

  Make sure the JMdict file is placed at data/JMdict before running.
  Download it from: https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project
"""

import os
import pandas as pd

from src.loader import load_gairaigo
from src.preprocessor import preprocess, build_features
from src.trainer import split_data, train_model
from src.evaluator import evaluate
from src.visualizer import (
    save_class_distribution,
    save_confusion_matrix,
    save_top_features,
)

JMDICT_PATH = "data/JMdict"
RESULTS_DIR = "output/results"


def main():
    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading gairaigo entries from JMdict...")
    df_raw = load_gairaigo(JMDICT_PATH)
    print(f"  Loaded     : {len(df_raw):,} gairaigo entries")
    print(f'  Languages  : {df_raw["language"].nunique()} unique donor languages')

    # ------------------------------------------------------------------
    # Step 2: Preprocess
    # ------------------------------------------------------------------
    print("\n[Step 2] Preprocessing...")
    df, label_encoder = preprocess(df_raw)
    print(f"  After dedup + class consolidation: {len(df):,} entries")

    from src.preprocessor import KEEP_LANGUAGES

    print(
        f"  Class selection criteria  : only {len(KEEP_LANGUAGES)} target languages kept"
    )
    print(f"  Target languages          : {sorted(KEEP_LANGUAGES)}")
    print(
        f"  Final classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}"
    )

    # ------------------------------------------------------------------
    # Step 3: Featurize
    # ------------------------------------------------------------------
    print("\n[Step 3] Building character n-gram feature matrix...")
    X, vectorizer = build_features(df["katakana"])
    y = df["label"].values
    print(f"  Feature matrix : {X.shape[0]:,} samples × {X.shape[1]:,} n-gram features")

    # ------------------------------------------------------------------
    # Step 4: Split
    # ------------------------------------------------------------------
    print("\n[Step 4] Splitting into train / test sets (80 / 20, stratified)...")
    X_train, X_test, y_train, y_test, df_train, df_test = split_data(X, y, df)
    print(f"  Train size : {X_train.shape[0]:,} samples")
    print(f"  Test size  : {X_test.shape[0]:,} samples")

    # ------------------------------------------------------------------
    # Step 5: Train
    # ------------------------------------------------------------------
    print("\n[Step 5] Training LinearSVC classifier...")
    model = train_model(X_train, y_train)
    print("  Training complete.")

    # ------------------------------------------------------------------
    # Step 6: Evaluate
    # ------------------------------------------------------------------
    print("\n[Step 6] Evaluating on test set...")
    results = evaluate(model, X_test, y_test, label_encoder)

    # ------------------------------------------------------------------
    # Step 7: Visualize
    # ------------------------------------------------------------------
    print("\n[Step 7] Generating visualizations...")
    save_class_distribution(df)
    save_confusion_matrix(results["confusion_matrix"], results["class_names"])
    save_top_features(model, vectorizer, label_encoder)

    # ------------------------------------------------------------------
    # Step 8: Export
    # ------------------------------------------------------------------
    print("\n[Step 8] Exporting predictions to CSV...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    export_df = pd.DataFrame(
        {
            "katakana": df_test["katakana"].values,
            "true_language": label_encoder.inverse_transform(y_test),
            "predicted_language": label_encoder.inverse_transform(results["y_pred"]),
        }
    )
    export_df["correct"] = export_df["true_language"] == export_df["predicted_language"]

    csv_path = os.path.join(RESULTS_DIR, "classified_loanwords.csv")
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Exported {len(export_df):,} predictions → {csv_path}")

    # Print a small sample of predictions so we can do a quick sanity check
    print("\n  Sample predictions (first 10 rows):")
    print(export_df.head(10).to_string(index=False))

    print("\n[Done] All steps complete.\n")


if __name__ == "__main__":
    main()
