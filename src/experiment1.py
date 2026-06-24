import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import wilcoxon


def run_experiment1(X, y, models):
    print("---------------------------------------------------------")
    print("EXPERIMENT 1: Model Comparison")
    print("---------------------------------------------------------")

    # Walidacja krzyżowa 5x2:
    # Dane dzielimy na 5 części (foldów), cały proces powtarzamy 2 razy.
    # Daje to 10 wyników na model → bardziej wiarygodna ocena niż jeden podział.
    # Stratified = zachowuje proporcje klas w każdym foldzie.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    # Słownik: nazwa modelu - tablica 10 wyników Balanced Accuracy
    # Potrzebny później do testu Wilcoxona (porównanie parami)
    all_scores = {}

    # Lista słowników z wynikami — na końcu zamienimy ją w tabelę DataFrame
    summary = []

    for name, model in models.items():
        print(f"Testing model: {name}...")

        # cross_validate uruchamia model na każdym z 10 podziałów
        # i zwraca słownik z tablicami wyników dla każdej metryki
        scores = cross_validate(
            model, X, y,
            cv=cv,
            scoring=["balanced_accuracy", "f1_macro", "precision", "recall"]
        )

        # Pobranie wyników z kolejnych iteracji walidacji krzyżowej
        ba_scores = scores["test_balanced_accuracy"] # 10 wartości
        f1_scores = scores["test_f1_macro"]          # 10 wartości
        precision_scores = scores["test_precision"] # 10 wartości
        recall_scores = scores["test_recall"]       # 10 wartości

        # Zapis wyników do późniejszego porównania statystycznego (Wilcoxon)
        all_scores[name] = ba_scores

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

    # Posortowanie modeli od najlepszego do najgorszego według Balanced Accuracy
    df_results = pd.DataFrame(summary).sort_values("Balanced Accuracy", ascending=False)

    print("\nResults:")
    print(df_results.to_string(index=False))

    # Zapis wyników do pliku CSV
    df_results.to_csv("results/exp1_results.csv", index=False)
    print("\nResults saved → results/exp1_results.csv")

    # Wykres słupkowy porównujący modele
    _plot_comparison(df_results)

    # Test Wilcoxona sprawdzający, czy różnice między modelami są istotne statystycznie
    _wilcoxon_test(all_scores)

    # Macierze pomyłek generowane na jednym podziale 80/20 (do wizualizacji wyników)
    _plot_confusion_matrices(X, y, models, "exp1_confusion_matrices.png", "Experiment 1")

    return df_results, all_scores


def _plot_comparison(df):
    """Wykres słupkowy porównujący Balanced Accuracy i F1 dla każdego modelu."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Pozycje słupków na osi X (0, 1, 2, ...)
    x = range(len(df))
    width = 0.35    # szerokość pojedynczego słupka

    # Słupki Balanced Accuracy — przesunięte w lewo o połowę szerokości
    ax.bar(
        [i - width/2 for i in x],
        df["Balanced Accuracy"],
        width,
        label="Balanced Accuracy",
        color="steelblue"
    )

    # Słupki F1 — przesunięte w prawo o połowę szerokości
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
    print("Chart saved - results/exp1_comparison.png")


def _wilcoxon_test(all_scores):
    """
    Test Wilcoxona sprawdza, czy różnica między dwoma modelami jest statystycznie istotna.
    Porównujemy każdą parę modeli na podstawie ich 10 wyników BA z walidacji krzyżowej.
    p < 0.05 → różnica jest istotna statystycznie (nie jest przypadkowa).
    """
    print("\nWilcoxon test (alpha = 0.05):")
    print("p < 0.05 means the difference is statistically significant\n")

    model_names = list(all_scores.keys())

    # Tworzymy pustą tabelę n×n na wartości p
    # (każda komórka to wynik testu dla pary modeli)
    p_table = pd.DataFrame(index=model_names, columns=model_names)

    for m1 in model_names:
        for m2 in model_names:
            if m1 == m2:
                # Model porównany sam ze sobą — brak sensu testowania
                p_table.loc[m1, m2] = "-"
            else:
                try:
                    # wilcoxon() przyjmuje dwie tablice wyników i zwraca (statystykę, p-value)
                    _, p = wilcoxon(all_scores[m1], all_scores[m2])
                    p_table.loc[m1, m2] = round(p, 4)
                except Exception:
                    # Błąd może wystąpić gdy oba modele mają identyczne wyniki
                    p_table.loc[m1, m2] = "1.0"

    print(p_table.to_string())
    p_table.to_csv("results/exp1_wilcoxon.csv")
    print("\np-values saved - results/exp1_wilcoxon.csv")


def _plot_confusion_matrices(X, y, models, filename, title_prefix):
    """
    Rysuje macierze pomyłek dla każdego modelu.

    Walidacja krzyżowa tworzy wiele różnych podziałów train/test,
    więc nie ma jednego "zbioru testowego". Dla wizualizacji robimy
    osobny, jednorazowy podział 80/20.

    Macierz pomyłek pokazuje:
      - TP (prawy dolny): poprawnie wykryte choroby
      - TN (lewy górny):  poprawnie wykryte osoby zdrowe
      - FP (prawy górny): zdrowi sklasyfikowani jako chorzy
      - FN (lewy dolny):  chorzy sklasyfikowani jako zdrowi
    """
    print(f"\nGenerating confusion matrices (separate 80/20 split, for the picture only)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Siatka 2×3 subwykresów — po jednym na model
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        # Trenowanie modelu na 80% danych i wykonanie predykcji na pozostałych 20%
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # confusion_matrix zwraca macierz 2×2
        cm = confusion_matrix(y_test, y_pred)

        ax = axes[i]
        ax.imshow(cm, cmap="Blues") # wizualizacja jako mapa ciepła
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Healthy", "Disease"])
        ax.set_yticklabels(["Healthy", "Disease"])

        # Wyświetlenie wartości wewnątrz pól macierzy
        for row in range(2):
            for col in range(2):
                ax.text(col, row, cm[row, col], ha="center", va="center")

    fig.suptitle(f"{title_prefix}: Confusion Matrices (80/20 split)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"Confusion matrices saved - results/{filename}")
