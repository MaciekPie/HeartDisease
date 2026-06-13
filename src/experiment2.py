import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def run_experiment2(X, y):
    print("\n" + "="*60)
    print("EXPERIMENT 2: Effect of Model Parameters")
    print("="*60)

    # Same cross-validation as in Experiment 1
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    # Test KNN with different numbers of neighbors
    _test_knn(X, y, cv)

    # Test Random Forest with different numbers of trees
    _test_random_forest(X, y, cv)


def _test_knn(X, y, cv):
    print("\n--- KNN: testing different n_neighbors ---")

    # Try different values of n_neighbors
    neighbors_to_test = [1, 3, 5, 7, 10, 15]
    results = []

    for n in neighbors_to_test:
        model = KNeighborsClassifier(n_neighbors=n)
        scores = cross_validate(model, X, y, cv=cv, scoring="balanced_accuracy")
        ba = scores["test_score"]

        results.append({
            "n_neighbors":       n,
            "Balanced Accuracy": round(ba.mean(), 4),
            "std":               round(ba.std(), 4),
        })
        print(f"  n_neighbors={n}: BA = {ba.mean():.4f} (±{ba.std():.4f})")

    df = pd.DataFrame(results)
    df.to_csv("results/exp2_knn.csv", index=False)

    # Line plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["n_neighbors"], df["Balanced Accuracy"], marker="o", color="steelblue")
    ax.fill_between(
        df["n_neighbors"],
        df["Balanced Accuracy"] - df["std"],
        df["Balanced Accuracy"] + df["std"],
        alpha=0.2, color="steelblue"
    )
    ax.set_xlabel("n_neighbors")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Experiment 2: KNN — Effect of n_neighbors")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/exp2_knn.png", dpi=150)
    plt.close()
    print("Chart saved → results/exp2_knn.png")


def _test_random_forest(X, y, cv):
    print("\n--- Random Forest: testing different n_estimators (number of trees) ---")

    trees_to_test = [10, 50, 100, 200, 500]
    results = []

    for n in trees_to_test:
        model = RandomForestClassifier(
            n_estimators=n,
            class_weight="balanced",
            random_state=42
        )
        scores = cross_validate(model, X, y, cv=cv, scoring="balanced_accuracy")
        ba = scores["test_score"]

        results.append({
            "n_estimators":      n,
            "Balanced Accuracy": round(ba.mean(), 4),
            "std":               round(ba.std(), 4),
        })
        print(f"  n_estimators={n}: BA = {ba.mean():.4f} (±{ba.std():.4f})")

    df = pd.DataFrame(results)
    df.to_csv("results/exp2_random_forest.csv", index=False)

    # Line plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["n_estimators"], df["Balanced Accuracy"], marker="o", color="green")
    ax.fill_between(
        df["n_estimators"],
        df["Balanced Accuracy"] - df["std"],
        df["Balanced Accuracy"] + df["std"],
        alpha=0.2, color="green"
    )
    ax.set_xlabel("n_estimators (number of trees)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Experiment 2: Random Forest — Effect of n_estimators")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/exp2_random_forest.png", dpi=150)
    plt.close()
    print("Chart saved → results/exp2_random_forest.png")

