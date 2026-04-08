"""
Prepares the raw gairaigo DataFrame for machine learning.

Processing steps (in order):
  1. Drop exact duplicate (katakana, language) pairs.
  2. Consolidate language classes that have too few samples into 'other'.
  3. Encode string language labels into integer class indices.
  4. Build a TF-IDF character n-gram feature matrix from the katakana strings.

Why character n-grams?
  Katakana phonetically transcribes the donor word's pronunciation. Different
  donor languages leave distinct phonetic fingerprints in the resulting katakana:
    - German words often end in ルト (-ruto), ンゲ (-nge), or ツ (tsu).
    - French words often contain ージュ (-aju) or アン (-an) nasals.
    - Portuguese words carry パン, タバコ, or イン patterns.
  A model trained on bigram-4-gram features can learn these sub-word signals.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.constants import decode_language


# Languages to observe
KEEP_LANGUAGES = {"eng", "fre", "ger"}


def preprocess(df: pd.DataFrame):
    """
    Clean the gairaigo DataFrame and encode its language labels.

    Processing order matters here:
      1. Deduplicate first, so rare-class counts reflect unique words only.
      2. Consolidate rare classes into 'other' based on those true counts.
      3. Decode ISO 639-2 codes into full language names (e.g. 'fre' → 'French')
         so that every downstream output — charts, reports, and CSVs — uses
         human-readable labels instead of three-letter codes.
      4. Encode the final string labels into integers for scikit-learn.

    Args:
        df : Raw DataFrame with columns ['katakana', 'language'].

    Returns:
        Tuple of (cleaned_df, label_encoder).
        cleaned_df has an additional 'label' column of integer class indices.
        label_encoder is fitted and can inverse-transform predictions later.
    """
    # Remove duplicates so the same loanword does not appear in both
    # the training and test sets, which would inflate accuracy artificially
    df = df.drop_duplicates(subset=["katakana", "language"]).copy()

    # Consolidate before decoding so the threshold applies to raw codes,
    # which is what the dataset uses internally
    df = _consolidate_rare_classes(df)

    # Replace short codes with full names now that class boundaries are set
    df["language"] = df["language"].apply(decode_language)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["language"])

    return df, label_encoder


def build_features(katakana_series: pd.Series, vectorizer: TfidfVectorizer = None):
    """
    Convert katakana strings into a sparse TF-IDF character n-gram matrix.

    Uses char_wb mode, which pads each word with boundary markers before
    extracting n-grams. This lets the model distinguish patterns at the
    start/end of a word from those in the middle — useful here because
    donor-language phonemes often appear at word edges.

    Args:
        katakana_series : Series of katakana strings to vectorize.
        vectorizer      : A pre-fitted TfidfVectorizer for transform-only mode.
                          Pass None to fit a new vectorizer on katakana_series.

    Returns:
        Tuple of (feature_matrix, vectorizer).
        feature_matrix is a sparse scipy matrix of shape (n_samples, n_features).
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",  # character n-grams with word-boundary padding
            ngram_range=(2, 4),  # bigrams, trigrams, and 4-grams
            sublinear_tf=True,  # replace raw TF with 1 + log(TF) to dampen outliers
            min_df=2,  # discard n-grams that appear in fewer than 2 words
        )
        feature_matrix = vectorizer.fit_transform(katakana_series)
    else:
        feature_matrix = vectorizer.transform(katakana_series)

    return feature_matrix, vectorizer


def _consolidate_rare_classes(df: pd.DataFrame) -> pd.DataFrame:
    df["language"] = df["language"].apply(
        lambda lang: lang if lang in KEEP_LANGUAGES else None
    )
    return df.dropna(subset=["language"])
