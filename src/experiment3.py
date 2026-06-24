import numpy as np
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

    # Taka sama walidacja krzyżowa jak w eksperymencie 1 i 2
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    model_names = list(models.keys())
    n_models    = len(model_names)
    n_sizes     = len(subset_sizes)

    # Macierze wyników zaalokowane z góry za pomocą np.zeros
    # Wiersz = wielkość podzbioru, kolumna = model
    ba_matrix = np.zeros((n_sizes, n_models))   # Balanced Accuracy
    f1_matrix = np.zeros((n_sizes, n_models))   # F1 (macro)

    for size_idx, size in enumerate(subset_sizes):
        percent = int(size * 100)
        print(f"\n--- Subset size: {percent}% ({int(len(X) * size)} samples) ---")

        # Pobranie próbki z zachowaniem proporcji klas (stratify)
        if size < 1.0:
            # train_test_split z parametrem train_size daje nam żądany ułamek danych
            X_sub, _, y_sub, _ = train_test_split(
                X, y,
                train_size=size,
                random_state=42,
                stratify=y   # gwarancja zachowania proporcji klas w próbce
            )
        else:
            # Wykorzystanie całego zbioru danych
            X_sub, y_sub = X, y

        for model_idx, name in enumerate(model_names):
            model = models[name]

            scores = cross_validate(
                model, X_sub, y_sub,
                cv=cv,
                scoring=["balanced_accuracy", "f1_macro"]
            )

            # Zapisz wyniki do odpowiedniej komórki macierzy
            ba = scores["test_balanced_accuracy"].mean()
            f1 = scores["test_f1_macro"].mean()

            ba_matrix[size_idx, model_idx] = round(ba, 4)
            f1_matrix[size_idx, model_idx] = round(f1, 4)

            print(f"  {name}: BA = {ba:.4f}, F1 = {f1:.4f}")

    # Zapisz wyniki
    # Etykiety wierszy dla DataFrame (np. "20%", "30%", ...)
    size_labels = [f"{int(s*100)}%" for s in subset_sizes]

    # Zamień macierze numpy na DataFrame i zapisz do CSV
    df_ba = pd.DataFrame(ba_matrix, index=size_labels, columns=model_names)
    df_ba.index.name = "Subset size"
    df_ba.to_csv("results/exp3_results_balanced_accuracy.csv")

    df_f1 = pd.DataFrame(f1_matrix, index=size_labels, columns=model_names)
    df_f1.index.name = "Subset size"
    df_f1.to_csv("results/exp3_results_f1.csv")

    print("\nBalanced Accuracy results:")
    print(df_ba.to_string())
    print("\nF1 (macro) results:")
    print(df_f1.to_string())
    print("\nResults saved → results/exp3_results_balanced_accuracy.csv, results/exp3_results_f1.csv")

    # Generowanie krzywych uczenia
    _plot_learning_curves(ba_matrix, size_labels, model_names)

    # Macierze pomyłek wygenerowane dla pełnego zbioru danych (100% dostępnych próbek)
    _plot_confusion_matrices_full(X, y, models)


def _plot_learning_curves(ba_matrix, size_labels, model_names):
    """
    Krzywe uczenia pokazują jak Balanced Accuracy zmienia się
    wraz z wielkością zbioru treningowego.
    Płaski koniec krzywej sugeruje, że dodatkowe dane nie poprawią modelu.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Każda kolumna macierzy to wyniki jednego modelu dla kolejnych rozmiarów
    for col_idx, name in enumerate(model_names):
        scores = ba_matrix[:, col_idx]   # wyciągnij kolumnę modelu
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
    print("Learning curves saved - results/exp3_learning_curves.png")


def _plot_confusion_matrices_full(X, y, models):
    """
    Macierze pomyłek dla pełnego zbioru danych (100%).
    Podział 80/20 służy wyłącznie do wizualizacji — nie do oceny modeli.
    """
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

        # Wypisz liczby wewnątrz każdego pola macierzy
        for row in range(2):
            for col in range(2):
                ax.text(col, row, cm[row, col], ha="center", va="center")

    fig.suptitle("Experiment 3: Confusion Matrices (100% data, 80/20 split)", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/exp3_confusion_matrices.png", dpi=150)
    plt.close()
    print("Confusion matrices saved - results/exp3_confusion_matrices.png")
