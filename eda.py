"""
eda.py — Exploratory Data Analysis (EDA)
Run: python eda.py --data <path_to_csv>

Saves plots to the results/eda/ folder.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import create_target, CONDITION_COLS, DROP_COLS

OUTPUT_DIR = "results/eda"


def run_eda(path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(path)
    df = create_target(df)

    print(f"Data shape: {df.shape}")
    print(f"\nClass distribution (target):\n{df['target'].value_counts()}\n")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

    # 1. Class distribution
    _plot_class_distribution(df)

    # 2. Histograms of numeric features
    _plot_feature_histograms(df)

    # 3. Boxplots — features vs class
    _plot_boxplots(df)

    # 4. Correlation heatmap
    _plot_correlation_heatmap(df)

    print(f"\nAll EDA plots saved to: {OUTPUT_DIR}/")


def _plot_class_distribution(df: pd.DataFrame):
    counts = df["target"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Class distribution (target)", fontsize=14)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.bar_label(ax.containers[0], fontsize=9)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    _save(fig, "class_distribution.png")


def _plot_feature_histograms(df: pd.DataFrame):
    cols_to_drop = CONDITION_COLS + [c for c in DROP_COLS if c in df.columns] + ["target"]
    numeric = df.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include="number")

    n = len(numeric.columns)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(numeric.columns):
        axes[i].hist(numeric[col].dropna(), bins=30, color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Histograms of numeric features", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "feature_histograms.png")


def _plot_boxplots(df: pd.DataFrame):
    """Boxplots for the top 8 numeric features vs class."""
    cols_to_drop = CONDITION_COLS + [c for c in DROP_COLS if c in df.columns]
    numeric_cols = (
        df.drop(columns=cols_to_drop, errors="ignore")
        .select_dtypes(include="number")
        .columns
        .tolist()
    )
    top_cols = numeric_cols[:8]  # first 8 to keep the chart readable

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(top_cols):
        df.boxplot(column=col, by="target", ax=axes[i])
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        plt.sca(axes[i])
        plt.xticks(rotation=45, ha="right", fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Boxplots of features vs class", fontsize=13)
    plt.tight_layout()
    _save(fig, "boxplots_by_class.png")


def _plot_correlation_heatmap(df: pd.DataFrame):
    cols_to_drop = CONDITION_COLS + [c for c in DROP_COLS if c in df.columns] + ["target"]
    numeric = df.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include="number")

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(max(10, len(corr) // 2), max(8, len(corr) // 2)))
    sns.heatmap(
        corr,
        annot=len(corr) <= 15,   # show annotations only when few features
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        linewidths=0.3,
    )
    ax.set_title("Feature correlation heatmap", fontsize=14)
    plt.tight_layout()
    _save(fig, "correlation_heatmap.png")


def _save(fig, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV file")
    args = parser.parse_args()
    run_eda(args.data)
