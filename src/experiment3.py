import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix


def run_experiment3(X, y, models):
    print("---------------------------------------------------------")
    print("EXPERIMENT 3: Effect of Training Data Size")
    print("---------------------------------------------------------")

    # Procenty danych używane do eksperymentu
    subset_sizes = [0.2, 0.3, 0.5, 0.7, 1.0]

    # Taka sama walidacja krzyżowa jak w eksperymencie 1
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    # results_ba[model_name]  = Słownik przechowujący wyniki Balanced Accuracy dla każdego modelu i każdej wielkości zbioru danych
    # results_f1[model_name]  = Słownik przechowujący wyniki F1 (macro) dla każdego modelu i każdej wielkości zbioru danych
    results_ba = {name: [] for name in models}
    results_f1 = {name: [] for name in models}

    for size in subset_sizes:
        percent = int(size * 100)
        print(f"\n--- Subset size: {percent}% ({int(len(X) * size)} samples) ---")

        # Pobranie próbki z zachowaniem proporcji klas
        if size < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X, y,
                train_size=size,
                random_state=42,
                stratify=y   # Zachowanie proporcji klas
            )
        else:
            # Wykorzystanie całego zbioru danych
            X_sub, y_sub = X, y

        for name, model in models.items():
            scores = cross_validate(
                model, X_sub, y_sub,
                cv=cv,
                scoring=["balanced_accuracy", "f1_macro"]
            )
            ba = scores["test_balanced_accuracy"].mean()
            f1 = scores["test_f1_macro"].mean()

            results_ba[name].append(round(ba, 4))
            results_f1[name].append(round(f1, 4))
            print(f"  {name}: BA = {ba:.4f}, F1 = {f1:.4f}")

    # Zapisz wyniki
    size_labels = [f"{int(s*100)}%" for s in subset_sizes]

    df_ba = pd.DataFrame(results_ba, index=size_labels)
    df_ba.index.name = "Subset size"
    df_ba.to_csv("results/exp3_results_balanced_accuracy.csv")

    df_f1 = pd.DataFrame(results_f1, index=size_labels)
    df_f1.index.name = "Subset size"
    df_f1.to_csv("results/exp3_results_f1.csv")

    print("\nBalanced Accuracy results:")
    print(df_ba.to_string())
    print("\nF1 (macro) results:")
    print(df_f1.to_string())
    print("\nResults saved → results/exp3_results_balanced_accuracy.csv, results/exp3_results_f1.csv")

    # Generowanie krzywych uczenia
    _plot_learning_curves(results_ba, size_labels)

    # Macierze pomyłek wygenerowane dla pełnego zbioru danych (100% dostępnych próbek)
    _plot_confusion_matrices_full(X, y, models)


def _plot_learning_curves(results_ba, size_labels):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, scores in results_ba.items():
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


def _plot_confusion_matrices_full(X, y, models):
    # Macierze pomyłek wyznaczane dla pełnego zbioru danych, przy użyciu pojedynczego podziału 80/20 wyłącznie do wizualizacji
    print("\nGenerating confusion matrices on full dataset (80/20 split, for the picture only)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
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

        for row in range(2):
            for col in range(2):
                ax.text(col, row, cm[row, col], ha="center", va="center")

    fig.suptitle("Experiment 3: Confusion Matrices (100% data, 80/20 split)", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/exp3_confusion_matrices.png", dpi=150)
    plt.close()
    print("Confusion matrices saved → results/exp3_confusion_matrices.png")
