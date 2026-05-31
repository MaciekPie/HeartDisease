import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def evaluate_models(models: dict, X_test, y_test, results_dir: str = "results") -> pd.DataFrame:
    """
    Evaluates all models on the test set.

    Saves:
      - results/metrics.csv          — metrics table for all models
      - results/confusion_<name>.png — confusion matrix for each model
      - results/models_comparison.png — bar chart comparing all models

    Returns:
        pd.DataFrame with metrics, sorted by macro F1 descending.
    """
    os.makedirs(results_dir, exist_ok=True)
    rows = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc       = balanced_accuracy_score(y_test, y_pred) # number of classes is not balanced
        f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        print(f"  Balanced accuracy: {acc:.4f}")
        print(f"  F1 (macro):        {f1:.4f}")
        print(f"  Precision (macro): {precision:.4f}")
        print(f"  Recall (macro):    {recall:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        rows.append({
            "Model":             name,
            "Balanced Accuracy": round(acc, 4),
            "F1 (macro)":        round(f1, 4),
            "Precision (macro)": round(precision, 4),
            "Recall (macro)":    round(recall, 4),
        })

        _save_confusion_matrix(y_test, y_pred, name, results_dir)

    df = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False)
    metrics_path = os.path.join(results_dir, "metrics.csv")
    df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved → {metrics_path}")

    _save_comparison_chart(df, results_dir)

    print(f"\n✓ Best model (by macro F1): {df.iloc[0]['Model']}")
    return df


def _save_confusion_matrix(y_test, y_pred, name: str, results_dir: str):
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14, pad=12)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(results_dir, f"confusion_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {path}")


def _save_comparison_chart(df: pd.DataFrame, results_dir: str):
    metrics = [col for col in df.columns if col != "Model"]
    fig, ax = plt.subplots(figsize=(10, 5))
    df.set_index("Model")[metrics].plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Model comparison — metrics on test set", fontsize=13)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(results_dir, "models_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Comparison chart → {path}")
