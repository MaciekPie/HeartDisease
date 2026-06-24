import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def run_experiment2(X, y):
    print("---------------------------------------------------------")
    print("EXPERIMENT 2: Effect of Model Parameters")
    print("---------------------------------------------------------")

    # Taka sama walidacja krzyżowa jak w eksperymencie 1 (5 podziałów, 2 powtórzenia)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    # Badanie wpływu liczby sąsiadów w KNN
    _test_knn(X, y, cv)

    # Badanie wpływu liczby drzew w Random Forest
    _test_random_forest(X, y, cv)


def _test_knn(X, y, cv):
    """
    KNN (K-Nearest Neighbors) klasyfikuje punkt na podstawie k najbliższych sąsiadów.
    Testujemy różne wartości k (n_neighbors) aby znaleźć optymalną.
    Za mało sąsiadów - model przeuczony (reaguje na szum).
    Za dużo sąsiadów - model zbyt ogólny (ignoruje lokalne wzorce).
    """
    print("\n--- KNN: testing different n_neighbors ---")

    # Test różnych wartości liczby sąsiadów
    neighbors_to_test = [1, 3, 5, 7, 9, 15]
    n = len(neighbors_to_test)

    # Tablice wyników zaalokowane z góry za pomocą np.zeros
    ba_means = np.zeros(n)   # średnie Balanced Accuracy dla każdego k
    ba_stds  = np.zeros(n)   # odchylenia standardowe BA
    f1_means = np.zeros(n)   # średnie F1 dla każdego k
    f1_stds  = np.zeros(n)   # odchylenia standardowe F1

    for i, n in enumerate(neighbors_to_test):
        model = KNeighborsClassifier(n_neighbors=n)
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["balanced_accuracy", "f1_macro"]
        )
        ba = scores["test_balanced_accuracy"]
        f1 = scores["test_f1_macro"]

        # Zapisz wyniki do odpowiednich komórek tablic
        ba_means[i] = round(scores["test_balanced_accuracy"].mean(), 4)
        ba_stds[i]  = round(scores["test_balanced_accuracy"].std(), 4)
        f1_means[i] = round(scores["test_f1_macro"].mean(), 4)
        f1_stds[i]  = round(scores["test_f1_macro"].std(), 4)

        print(f"  n_neighbors={n}: BA = {ba_means[i]:.4f}, F1 = {f1_means[i]:.4f}")

    # Zbuduj DataFrame z wyników i zapisz do CSV
    df = pd.DataFrame({
        "n_neighbors":       neighbors_to_test,
        "Balanced Accuracy": ba_means,
        "BA std":            ba_stds,
        "F1 (macro)":        f1_means,
        "F1 std":            f1_stds,
    })
    df.to_csv("results/exp2_knn.csv", index=False)

    # Wykres zależności jakości klasyfikacji od liczby sąsiadów
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(neighbors_to_test, ba_means, marker="o", label="Balanced Accuracy", color="steelblue")
    ax.plot(neighbors_to_test, f1_means, marker="s", label="F1 (macro)", color="orange")
    ax.set_xlabel("n_neighbors")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 2: KNN — Effect of n_neighbors")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/exp2_knn.png", dpi=150)
    plt.close()
    print("Chart saved - results/exp2_knn.png")

    # Wygenerowanie macierzy pomyłek dla najlepszego znalezionego parametru
    best_idx = int(np.argmax(ba_means))  # indeks największej wartości BA
    best_n = neighbors_to_test[best_idx]  # odpowiadająca wartość k
    print(f"  Best n_neighbors = {best_n} - drawing its confusion matrix")
    best_model = KNeighborsClassifier(n_neighbors=best_n)

    # Wywołanie funkcji pomocniczej rysującej macierz pomyłek dla jednego modelu
    # Przekazujemy: dane X i y, model, tytuł wykresu i nazwę pliku do zapisu
    _plot_single_confusion_matrix(
        X, y, best_model,
        title=f"KNN (n_neighbors={best_n})",
        filename="exp2_knn_confusion.png"
    )


def _test_random_forest(X, y, cv):
    """
    Random Forest to zbiór drzew decyzyjnych (las).
    Każde drzewo uczy się na losowej próbce danych i cech.
    Testujemy różne wartości n_estimators (liczba drzew w lesie).
    Więcej drzew - lepsze wyniki, ale dłuższy czas uczenia.
    """
    print("\n--- Random Forest: testing different n_estimators (number of trees) ---")

    trees_to_test = [10, 50, 100, 200, 500]
    n = len(trees_to_test)

    # Tablice na wyniki — zaalokowane z góry jako np.zeros
    ba_means = np.zeros(n)
    ba_stds  = np.zeros(n)
    f1_means = np.zeros(n)
    f1_stds  = np.zeros(n)

    for i, n_trees in enumerate(trees_to_test):
        model = RandomForestClassifier(
            n_estimators=n_trees,
            class_weight="balanced",
            random_state=42
        )
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["balanced_accuracy", "f1_macro"]
        )

        ba_means[i] = round(scores["test_balanced_accuracy"].mean(), 4)
        ba_stds[i]  = round(scores["test_balanced_accuracy"].std(), 4)
        f1_means[i] = round(scores["test_f1_macro"].mean(), 4)
        f1_stds[i]  = round(scores["test_f1_macro"].std(), 4)

        print(f"  n_estimators={n_trees}: BA = {ba_means[i]:.4f}, F1 = {f1_means[i]:.4f}")

    # Zbuduj DataFrame z wyników i zapisz do CSV
    df = pd.DataFrame({
        "n_estimators":      trees_to_test,
        "Balanced Accuracy": ba_means,
        "BA std":            ba_stds,
        "F1 (macro)":        f1_means,
        "F1 std":            f1_stds,
    })
    df.to_csv("results/exp2_random_forest.csv", index=False)

    # Wykres zależności wartości Balanced Accuracy i F1 od liczby drzew w modelu
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trees_to_test, ba_means, marker="o", label="Balanced Accuracy", color="green")
    ax.plot(trees_to_test, f1_means, marker="s", label="F1 (macro)", color="orange")
    ax.set_xlabel("n_estimators (number of trees)")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 2: Random Forest — Effect of n_estimators")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/exp2_random_forest.png", dpi=150)
    plt.close()
    print("Chart saved - results/exp2_random_forest.png")

    # Wygenerowanie macierzy pomyłek dla najlepszego znalezionego parametru n_estimators
    best_idx    = int(np.argmax(ba_means))
    best_n      = trees_to_test[best_idx]
    print(f"  Best n_estimators = {best_n} → drawing its confusion matrix")
    best_model = RandomForestClassifier(n_estimators=best_n, class_weight="balanced", random_state=42)
    
    # Wywołanie funkcji pomocniczej rysującej macierz pomyłek dla jednego modelu
    # Przekazujemy: dane X i y, model, tytuł wykresu i nazwę pliku do zapisu
    _plot_single_confusion_matrix(
        X, y, best_model,
        title=f"Random Forest (n_estimators={best_n})",
        filename="exp2_random_forest_confusion.png"
    )


def _plot_single_confusion_matrix(X, y, model, title, filename):
    """
    Rysuje macierz pomyłek dla jednego modelu.
    Używamy prostego podziału 80/20 wyłącznie do celów wizualizacji
    (wyniki liczbowe eksperymentu pochodzą z walidacji krzyżowej).
    """
    # Prosty podział danych 80/20 wykorzystywany wyłącznie do celów wizualizacji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Trenuj model i wykonaj predykcję na zbiorze testowym
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Oblicz macierz pomyłek (tablica 2×2)
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

    # Wypisz liczby wewnątrz każdego pola macierzy
    for row in range(2):
        for col in range(2):
            ax.text(col, row, cm[row, col], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved - results/{filename}")
