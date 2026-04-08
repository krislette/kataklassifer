"""
Handles the train/test split and fitting of the classifier.

Model: LinearSVC (Linear Support Vector Classifier)
  - Designed for high-dimensional sparse feature spaces, which is exactly
    what TF-IDF character n-gram matrices produce.
  - Performs multi-class classification using a one-vs-rest strategy
    (one binary classifier per class; the highest-scoring class wins).
  - class_weight='balanced' adjusts the penalty weight inversely
    proportional to class frequency, which corrects for the strong
    imbalance caused by English being the dominant donor language.
  - Significantly faster to train than kernel SVMs on large sparse inputs.
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


# 80 % of the data goes to training; 20 % is held out for evaluation
TEST_SIZE = 0.2

# Fixed seed for reproducibility across runs
RANDOM_STATE = 42


def split_data(X, y, df):
    """
    Split the feature matrix, label array, and source DataFrame simultaneously.

    Splitting all three together ensures that the test-set rows in df line up
    exactly with the rows in X_test and y_test, which we need for the CSV export.

    Stratification preserves the class distribution in both splits, which is
    important given the heavy imbalance toward English entries.

    Args:
        X  : Sparse TF-IDF feature matrix (n_samples, n_features).
        y  : Integer label array of shape (n_samples,).
        df : Cleaned DataFrame aligned with X and y (same row order).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, df_train, df_test).
    """
    # Generate a row-index array so we can split the DataFrame in sync with X and y
    indices = np.arange(len(df))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, df_train, df_test


def train_model(X_train, y_train) -> LinearSVC:
    """
    Fit a LinearSVC on the training data and return the trained model.

    Args:
        X_train : Sparse training feature matrix.
        y_train : Integer training labels.

    Returns:
        Fitted LinearSVC model ready for prediction and evaluation.
    """
    model = LinearSVC(
        class_weight="balanced",  # compensates for English-heavy class imbalance
        max_iter=2000,  # extra iterations for convergence on larger datasets
        random_state=RANDOM_STATE,
        dual=False,
    )
    model.fit(X_train, y_train)
    return model
