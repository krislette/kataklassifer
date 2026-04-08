"""
Generates and saves all output charts for the gairaigo origin classifier.

Charts produced:
  1. class_distribution.png
       Bar chart showing how many loanword samples exist per origin language.
       Useful for understanding class imbalance before modeling.

  2. confusion_matrix.png
       Heatmap of true vs. predicted language labels on the test set.
       The diagonal shows correct predictions; off-diagonal cells reveal
       which language pairs the model confuses most.

  3. top_features.png
       Horizontal bar chart of the highest-coefficient character n-grams
       for each language class (from the LinearSVC weight vectors).
       Shows which phonetic patterns the model learned to associate with
       each donor language, the linguistically interesting output.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd


OUTPUT_DIR = "output/plots"

_JAPANESE_FONT_CANDIDATES = [
    "Meiryo",  # Windows, bundled since Vista
    "Yu Gothic",  # Windows 8.1+
    "MS Gothic",  # Windows, older fallback
    "Hiragino Sans",  # macOS
    "Hiragino Kaku Gothic Pro",  # macOS, older
    "Noto Sans CJK JP",  # Linux / cross-platform
    "IPAGothic",  # Linux
    "IPAPGothic",  # Linux alternate
]


def _configure_japanese_font() -> None:
    """
    Detect and activate a Japanese-capable font for matplotlib.

    Scans the system font list once and sets rcParams['font.family'] to the
    first candidate found. Prints a one-line status so it is visible in the
    console without being noisy. Called automatically when this module loads.
    """
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font_name in _JAPANESE_FONT_CANDIDATES:
        if font_name in available_fonts:
            matplotlib.rcParams["font.family"] = font_name
            print(f'  [font] Using "{font_name}" for katakana rendering.')
            return

    # No Japanese font found, charts will still save but glyphs may appear as boxes
    print(
        "  [font] Warning: no Japanese font found on this system.\n"
        "         Katakana labels in top_features.png may render as boxes.\n"
        "         Install Meiryo, Yu Gothic, or Noto Sans CJK JP and rerun."
    )


# Run font detection once at import time so every chart function picks it up
_configure_japanese_font()


def save_class_distribution(df: pd.DataFrame):
    """
    Plot and save a horizontal bar chart of loanword sample counts by origin language.

    Horizontal layout keeps full language names (e.g. 'Ancient Greek') readable
    on the left axis without overlapping, which would happen with vertical bars
    once names exceed about six characters.

    Args:
        df : Cleaned DataFrame with a 'language' column.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sort ascending so the longest bar sits at the top of the chart
    counts = df["language"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        counts.index, counts.values, color="steelblue", edgecolor="white", height=0.65
    )

    # Annotate each bar with its exact sample count, placed just past the bar tip
    max_count = counts.values.max()
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_width() + max_count * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            ha="left",
            va="center",
            fontsize=9,
            color="#333333",
        )

    ax.set_title("Gairaigo Sample Count by Origin Language", fontsize=14, pad=14)
    ax.set_xlabel("Number of Loanword Entries", fontsize=11)
    ax.set_ylabel("Origin Language", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Give the count labels room on the right so they are not clipped
    ax.set_xlim(right=max_count * 1.12)

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()
    print("  [saved] class_distribution.png")


def save_confusion_matrix(cm: np.ndarray, class_names: list):
    """
    Plot and save a labeled heatmap of the confusion matrix.

    Args:
        cm          : Confusion matrix array from sklearn.metrics.confusion_matrix.
        class_names : Ordered list of class label strings.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Gairaigo Origin Classifier", fontsize=13, pad=14)
    ax.set_xlabel("Predicted Language", fontsize=11)
    ax.set_ylabel("True Language", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  [saved] confusion_matrix.png")


def save_top_features(model, vectorizer, label_encoder, top_n: int = 10):
    """
    Plot the top-weighted character n-grams for each language class.

    LinearSVC assigns a coefficient to every feature for every class.
    Higher coefficients mean the n-gram is a stronger positive signal for
    that class. Visualizing these reveals which phonetic sub-patterns
    the model learned to associate with each donor language.

    Args:
        model         : Fitted LinearSVC model.
        vectorizer    : Fitted TfidfVectorizer used during featurization.
        label_encoder : LabelEncoder used during preprocessing.
        top_n         : Number of top features to show per class.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feature_names = np.array(vectorizer.get_feature_names_out())
    class_names = label_encoder.classes_
    n_classes = len(class_names)

    # LinearSVC stores one coefficient row per class (one-vs-rest strategy)
    coefficients = model.coef_

    cols = 3
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.8))
    axes = axes.flatten()

    for i, (class_name, coef_row) in enumerate(zip(class_names, coefficients)):
        top_indices = np.argsort(coef_row)[-top_n:]
        top_features = feature_names[top_indices]
        top_values = coef_row[top_indices]

        axes[i].barh(top_features, top_values, color="steelblue", edgecolor="white")
        axes[i].set_title(class_name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Coefficient Weight", fontsize=8)
        axes[i].tick_params(axis="y", labelsize=8)
        axes[i].spines[["top", "right"]].set_visible(False)

    # Hide unused subplot panels when n_classes is not a multiple of cols
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Top {top_n} Character N-Gram Features per Language Class\n"
        f"(LinearSVC Coefficient Weights)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "top_features.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  [saved] top_features.png")
