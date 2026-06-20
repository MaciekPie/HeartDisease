import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


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
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["balanced_accuracy", "f1_macro"]
        )
        ba = scores["test_balanced_accuracy"]
        f1 = scores["test_f1_macro"]

        results.append({
            "n_neighbors":       n,
            "Balanced Accuracy": round(ba.mean(), 4),
            "BA std":            round(ba.std(), 4),
            "F1 (macro)":        round(f1.mean(), 4),
            "F1 std":            round(f1.std(), 4),
        })
        print(f"  n_neighbors={n}: BA = {ba.mean():.4f}, F1 = {f1.mean():.4f}")

    df = pd.DataFrame(results)
    df.to_csv("results/exp2_knn.csv", index=False)

    # Line plot: Balanced Accuracy and F1 vs n_neighbors
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["n_neighbors"], df["Balanced Accuracy"], marker="o", label="Balanced Accuracy", color="steelblue")
    ax.plot(df["n_neighbors"], df["F1 (macro)"], marker="s", label="F1 (macro)", color="orange")
    ax.set_xlabel("n_neighbors")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 2: KNN — Effect of n_neighbors")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/exp2_knn.png", dpi=150)
    plt.close()
    print("Chart saved → results/exp2_knn.png")

    # Confusion matrix for the best n_neighbors value found
    best_n = df.sort_values("Balanced Accuracy", ascending=False).iloc[0]["n_neighbors"]
    best_n = int(best_n)
    print(f"  Best n_neighbors = {best_n} → drawing its confusion matrix")
    best_model = KNeighborsClassifier(n_neighbors=best_n)
    _plot_single_confusion_matrix(
        X, y, best_model,
        title=f"KNN (n_neighbors={best_n})",
        filename="exp2_knn_confusion.png"
    )


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
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["balanced_accuracy", "f1_macro"]
        )
        ba = scores["test_balanced_accuracy"]
        f1 = scores["test_f1_macro"]

        results.append({
            "n_estimators":      n,
            "Balanced Accuracy": round(ba.mean(), 4),
            "BA std":            round(ba.std(), 4),
            "F1 (macro)":        round(f1.mean(), 4),
            "F1 std":            round(f1.std(), 4),
        })
        print(f"  n_estimators={n}: BA = {ba.mean():.4f}, F1 = {f1.mean():.4f}")

    df = pd.DataFrame(results)
    df.to_csv("results/exp2_random_forest.csv", index=False)

    # Line plot: Balanced Accuracy and F1 vs n_estimators
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["n_estimators"], df["Balanced Accuracy"], marker="o", label="Balanced Accuracy", color="green")
    ax.plot(df["n_estimators"], df["F1 (macro)"], marker="s", label="F1 (macro)", color="orange")
    ax.set_xlabel("n_estimators (number of trees)")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 2: Random Forest — Effect of n_estimators")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/exp2_random_forest.png", dpi=150)
    plt.close()
    print("Chart saved → results/exp2_random_forest.png")

    # Confusion matrix for the best n_estimators value found
    best_n = df.sort_values("Balanced Accuracy", ascending=False).iloc[0]["n_estimators"]
    best_n = int(best_n)
    print(f"  Best n_estimators = {best_n} → drawing its confusion matrix")
    best_model = RandomForestClassifier(n_estimators=best_n, class_weight="balanced", random_state=42)
    _plot_single_confusion_matrix(
        X, y, best_model,
        title=f"Random Forest (n_estimators={best_n})",
        filename="exp2_random_forest_confusion.png"
    )


def _plot_single_confusion_matrix(X, y, model, title, filename):
    # One simple 80/20 split, just for this picture
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Disease"])
    ax.set_yticklabels(["Healthy", "Disease"])

    for row in range(2):
        for col in range(2):
            ax.text(col, row, cm[row, col], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → results/{filename}")
