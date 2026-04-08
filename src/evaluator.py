"""
Computes and reports classification metrics for the trained LinearSVC model.

Metrics reported:
  - Overall accuracy   : fraction of test samples correctly classified.
  - Per-class precision: of all samples predicted as class C, how many truly are C?
  - Per-class recall   : of all true samples of class C, how many were predicted C?
  - Per-class F1-score : harmonic mean of precision and recall per class.
  - Confusion matrix   : raw counts of (true label, predicted label) pairs;
                         passed to the visualizer for heatmap generation.
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(model, X_test, y_test, label_encoder) -> dict:
    """
    Run the trained model on the test set and compute all evaluation metrics.

    Args:
        model         : Fitted LinearSVC classifier.
        X_test        : Sparse test feature matrix.
        y_test        : True integer labels for the test set.
        label_encoder : LabelEncoder fitted during preprocessing; used to
                        convert integer predictions back to language strings.

    Returns:
        Dictionary with the following keys:
          'accuracy'        : float, overall test accuracy.
          'report'          : str, full per-class metrics table.
          'confusion_matrix': ndarray of shape (n_classes, n_classes).
          'class_names'     : list of string class names in label order.
          'y_pred'          : ndarray of integer predictions.
          'y_test'          : ndarray of true integer labels (same object passed in).
    """
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_

    accuracy = accuracy_score(y_test, y_pred)

    # zero_division=0 silences warnings for classes that never appear in predictions
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    _print_results(accuracy, report)

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
        "y_pred": y_pred,
        "y_test": y_test,
    }


def _print_results(accuracy: float, report: str):
    """Print a formatted summary of the evaluation results to stdout."""
    divider = "=" * 52
    print(f"\n{divider}")
    print("  Classification Results")
    print(divider)
    print(f"  Overall Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print("\n  Per-Class Metrics:\n")
    print(report)
