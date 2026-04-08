"""
Loads the saved model artifacts and predicts the origin language of one or
more katakana loanwords provided by the user.

The model, vectorizer, and label encoder must already exist in models/.
Run scripts/train.py first if they are not there yet.

Usage (interactive prompt):
  python scripts/predict.py

Usage (pass words directly as arguments):
  python scripts/predict.py テレビ コーヒー アルバイト

Output example:
  テレビ       -> English
  コーヒー     -> Dutch
  アルバイト   -> German
"""

import sys
import os
import re

# Allow imports from the project root regardless of where the script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib

MODEL_DIR = "models"

# Matches strings made entirely of katakana characters (same rule as loader.py)
KATAKANA_PATTERN = re.compile(r"^[\u30A0-\u30FF]+$")


def load_artifacts():
    """
    Load the three saved model artifacts from the models/ directory.

    Exits with a clear message if the files are not found, so the user
    knows to run train.py first rather than seeing a raw FileNotFoundError.
    """
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    encoder_path = os.path.join(MODEL_DIR, "encoder.joblib")

    missing = [
        p for p in (model_path, vectorizer_path, encoder_path) if not os.path.exists(p)
    ]
    if missing:
        print("\n[predict] Error: the following model files were not found:")
        for path in missing:
            print(f"           {path}")
        print("\n          Run scripts/train.py first to generate them.\n")
        sys.exit(1)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)
    return model, vectorizer, label_encoder


def predict(
    words: list[str], model, vectorizer, label_encoder
) -> list[tuple[str, str]]:
    """
    Predict the origin language for each katakana word.

    Words that are not pure katakana are flagged as invalid and skipped
    rather than silently passed to the model with meaningless features.

    Args:
        words         : List of katakana strings to classify.
        model         : Fitted LinearSVC.
        vectorizer    : Fitted TfidfVectorizer.
        label_encoder : Fitted LabelEncoder.

    Returns:
        List of (word, prediction) tuples. Invalid words get '(invalid input)'
        as their prediction so the output table stays aligned.
    """
    results = []

    # Separate valid katakana words from invalid inputs
    valid_words = [w for w in words if KATAKANA_PATTERN.match(w)]
    invalid_words = {w for w in words if not KATAKANA_PATTERN.match(w)}

    if valid_words:
        # Vectorize all valid words in one batch for efficiency
        X = vectorizer.transform(valid_words)
        predictions = label_encoder.inverse_transform(model.predict(X))
        word_to_pred = dict(zip(valid_words, predictions))
    else:
        word_to_pred = {}

    for word in words:
        if word in invalid_words:
            results.append((word, "(invalid — not pure katakana)"))
        else:
            results.append((word, word_to_pred[word]))

    return results


def print_results(results: list[tuple[str, str]]):
    """Print predictions in a clean aligned two-column table."""
    if not results:
        return

    max_word_len = max(len(word) for word, _ in results)

    print()
    print(f'  {"Katakana":<{max_word_len + 2}}  Predicted Origin Language')
    print(f'  {"-" * (max_word_len + 2)}  {"-" * 28}')
    for word, prediction in results:
        print(f"  {word:<{max_word_len + 2}}  {prediction}")
    print()


def interactive_mode(model, vectorizer, label_encoder):
    """
    Run a loop that prompts the user for katakana words and prints predictions.

    Multiple words can be entered space-separated on a single line.
    Type 'quit' or press Ctrl+C to exit.
    """
    print("\n[predict] Gairaigo Origin Classifier — interactive mode")
    print('          Enter katakana words (space-separated). Type "quit" to exit.\n')

    while True:
        try:
            raw = input("  >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  じゃあね！\n")
            break

        if raw.lower() in ("quit", "exit", "q"):
            print("  じゃあね！\n")
            break

        if not raw:
            continue

        words = raw.split()
        results = predict(words, model, vectorizer, label_encoder)
        print_results(results)


def main():
    model, vectorizer, label_encoder = load_artifacts()
    print(f"\n[predict] Model loaded. Known classes: {list(label_encoder.classes_)}")

    # If words were passed as CLI arguments, classify them and exit
    if len(sys.argv) > 1:
        words = sys.argv[1:]
        results = predict(words, model, vectorizer, label_encoder)
        print_results(results)
    else:
        # No arguments, launch interactive prompt
        interactive_mode(model, vectorizer, label_encoder)


if __name__ == "__main__":
    main()
