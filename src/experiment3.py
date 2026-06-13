import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split


def run_experiment3(X, y, models):
    print("\n" + "="*60)
    print("EXPERIMENT 3: Effect of Training Data Size")
    print("="*60)

    # We will test these fractions of the full dataset
    subset_sizes = [0.2, 0.3, 0.5, 0.7, 1.0]

    # Same cross-validation as in Experiment 1
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    # results[model_name] = list of BA scores, one per subset size
    results = {name: [] for name in models}

    for size in subset_sizes:
        percent = int(size * 100)
        print(f"\n--- Subset size: {percent}% ({int(len(X) * size)} samples) ---")

        # Take a stratified sample of the data
        if size < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X, y,
                train_size=size,
                random_state=42,
                stratify=y   # keep class proportions
            )
        else:
            # Use the full dataset
            X_sub, y_sub = X, y

        for name, model in models.items():
            scores = cross_validate(
                model, X_sub, y_sub,
                cv=cv,
                scoring="balanced_accuracy"
            )
            ba = scores["test_score"].mean()
            results[name].append(round(ba, 4))
            print(f"  {name}: BA = {ba:.4f}")

    # Save results table
    size_labels = [f"{int(s*100)}%" for s in subset_sizes]
    df = pd.DataFrame(results, index=size_labels)
    df.index.name = "Subset size"
    df.to_csv("results/exp3_results.csv")
    print("\nResults saved → results/exp3_results.csv")
    print(df.to_string())

    # Plot learning curves
    _plot_learning_curves(results, size_labels)


def _plot_learning_curves(results, size_labels):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, scores in results.items():
        ax.plot(size_labels, scores, marker="o", label=name)

    ax.set_xlabel("Training data size")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Experiment 3: Learning Curves")
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig("results/exp3_learning_curves.png", dpi=150)
    plt.close()
    print("Learning curves saved → results/exp3_learning_curves.png")
