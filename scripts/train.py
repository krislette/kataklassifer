"""
Trains the gairaigo origin classifier on the full dataset and saves the
trained artifacts to disk so that predict.py can load and use them without
needing to re-train.

Saved artifacts (written to models/):
  model.joblib      — the fitted LinearSVC classifier
  vectorizer.joblib — the fitted TfidfVectorizer (char n-gram)
  encoder.joblib    — the fitted LabelEncoder (integer ↔ language name)

Usage:
  python scripts/train.py
"""

import sys
import os

# Allow imports from the project root regardless of where the script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib

from src.loader import load_gairaigo
from src.preprocessor import preprocess, build_features

JMDICT_PATH = "data/JMdict"
MODEL_DIR = "models"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load
    print("\n[train] Loading gairaigo entries from JMdict...")
    df_raw = load_gairaigo(JMDICT_PATH)
    print(f"        {len(df_raw):,} entries loaded.")

    # Preprocess
    print("\n[train] Preprocessing...")
    df, label_encoder = preprocess(df_raw)
    print(f"        {len(df):,} entries after dedup and class consolidation.")
    print(f"        Classes: {list(label_encoder.classes_)}")

    # Featurize
    print("\n[train] Building character n-gram features...")
    X, vectorizer = build_features(df["katakana"])
    y = df["label"].values
    print(f"        Feature matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features")

    # Split
    from src.trainer import split_data, train_model
    from src.evaluator import evaluate

    print("\n[train] Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test, df_train, df_test = split_data(X, y, df)
    print(
        f"        Train: {X_train.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples"
    )

    # Train
    print("\n[train] Fitting LinearSVC on training set...")
    model = train_model(X_train, y_train)
    print("        Training complete.")

    # Evaluate
    print("\n[train] Evaluating on test set...")
    evaluate(model, X_test, y_test, label_encoder)

    # Save artifacts
    print(f"\n[train] Saving model artifacts to {MODEL_DIR}/ ...")

    model_path = os.path.join(MODEL_DIR, "model.joblib")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    encoder_path = os.path.join(MODEL_DIR, "encoder.joblib")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"        Saved: {model_path}")
    print(f"        Saved: {vectorizer_path}")
    print(f"        Saved: {encoder_path}")
    print("\n[train] Done. Run predict.py to classify new katakana words.\n")


if __name__ == "__main__":
    main()
