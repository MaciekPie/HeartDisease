import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from scipy.stats import wilcoxon


def run_experiment1(X, y, models):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Model Comparison")
    print("="*60)

    # 5x2 cross-validation: 5 splits repeated 2 times = 10 results per model
    # This gives more stable results than a single train/test split
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    all_scores = {}  # store all 10 scores per model (needed for Wilcoxon test)
    summary = []

    for name, model in models.items():
        print(f"Testing {name}...")

        scores = cross_validate(
            model, X, y,
            cv=cv,
            scoring=["balanced_accuracy", "f1_macro"]
        )

        ba_scores = scores["test_balanced_accuracy"]   # 10 values
        f1_scores = scores["test_f1_macro"]            # 10 values

        all_scores[name] = ba_scores  # save for Wilcoxon test later

        summary.append({
            "Model":              name,
            "Balanced Accuracy":  round(ba_scores.mean(), 4),
            "BA std":             round(ba_scores.std(), 4),
            "F1 (macro)":         round(f1_scores.mean(), 4),
            "F1 std":             round(f1_scores.std(), 4),
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

