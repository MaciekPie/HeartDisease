import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import wilcoxon


def run_experiment1(X, y, models):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Model Comparison")
    print("="*60)

    # 5x2 cross-validation: 5 splits repeated 2 times = 10 results per model
    # This gives more stable results than a single train/test split
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    all_scores = {}  # store all 10 Balanced Accuracy scores per model (needed for Wilcoxon test)
    summary = []

    for name, model in models.items():
        print(f"Testing {name}...")

        scores = cross_validate(
            model, X, y,
            cv=cv,
            scoring=["balanced_accuracy", "f1_macro", "precision", "recall"]
        )

        ba_scores = scores["test_balanced_accuracy"]   # 10 values
        f1_scores = scores["test_f1_macro"]            # 10 values

        precision_scores = scores["test_precision"]   # 10 values
        recall_scores = scores["test_recall"]            # 10 values

        all_scores[name] = ba_scores  # save for Wilcoxon test later

        summary.append({
            "Model":              name,
            "Balanced Accuracy":  round(ba_scores.mean(), 4),
            "BA std":             round(ba_scores.std(), 4),
            "F1 (macro)":         round(f1_scores.mean(), 4),
            "F1 std":             round(f1_scores.std(), 4),
            "Precision":         round(precision_scores.mean(), 4),
            "Precision std":             round(precision_scores.std(), 4),
            "Recall":         round(recall_scores.mean(), 4),
            "Recall std":             round(recall_scores.std(), 4),
        })

    # Sort by Balanced Accuracy
    df_results = pd.DataFrame(summary).sort_values("Balanced Accuracy", ascending=False)

    print("\nResults:")
    print(df_results.to_string(index=False))

    # Save results to CSV
    df_results.to_csv("results/exp1_results.csv", index=False)
    print("\nResults saved → results/exp1_results.csv")

    # Bar chart comparing models
    _plot_comparison(df_results)

    # Wilcoxon test — check if differences between models are statistically significant
    _wilcoxon_test(all_scores)

    # Confusion matrices — one simple 80/20 split, just for the picture
    _plot_confusion_matrices(X, y, models, "exp1_confusion_matrices.png", "Experiment 1")

    return df_results, all_scores


def _plot_comparison(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(df))
    width = 0.35

    ax.bar(
        [i - width/2 for i in x],
        df["Balanced Accuracy"],
        width,
        label="Balanced Accuracy",
        color="steelblue"
    )
    ax.bar(
        [i + width/2 for i in x],
        df["F1 (macro)"],
        width,
        label="F1 (macro)",
        color="orange"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Experiment 1: Model Comparison (5x2 CV)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/exp1_comparison.png", dpi=150)
    plt.close()
    print("Chart saved → results/exp1_comparison.png")


def _wilcoxon_test(all_scores):
    print("\nWilcoxon test (alpha = 0.05):")
    print("p < 0.05 means the difference is statistically significant\n")

    model_names = list(all_scores.keys())
    p_table = pd.DataFrame(index=model_names, columns=model_names)

    for m1 in model_names:
        for m2 in model_names:
            if m1 == m2:
                p_table.loc[m1, m2] = "-"
            else:
                try:
                    _, p = wilcoxon(all_scores[m1], all_scores[m2])
                    p_table.loc[m1, m2] = round(p, 4)
                except Exception:
                    # wilcoxon fails if both arrays are identical
                    p_table.loc[m1, m2] = "1.0"

    print(p_table.to_string())
    p_table.to_csv("results/exp1_wilcoxon.csv")
    print("\np-values saved → results/exp1_wilcoxon.csv")


def _plot_confusion_matrices(X, y, models, filename, title_prefix):
    # Cross-validation gives many different splits, so there is no single
    # "test set" to draw one confusion matrix from.
    # Instead, we do one simple 80/20 split here ONLY for this picture.
    print(f"\nGenerating confusion matrices (separate 80/20 split, for the picture only)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        # Train this model on the 80% split and predict on the 20% split
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        ax = axes[i]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Healthy", "Disease"])
        ax.set_yticklabels(["Healthy", "Disease"])

        # write the numbers inside each box
        for row in range(2):
            for col in range(2):
                ax.text(col, row, cm[row, col], ha="center", va="center")

    fig.suptitle(f"{title_prefix}: Confusion Matrices (80/20 split)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"Confusion matrices saved → results/{filename}")
