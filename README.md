# Heart Disease Classification

Projekt porównujący metody klasyfikacji binarnej do przewidywania choroby serca na podstawie danych klinicznych pacjentów.

**Metody Sztucznej Inteligencji | Wrocław 2026**
**Autorzy:** Aleksandra Suwaj, Natalia Ponomarenkow, Maciej Pieczykolan

---

## Opis problemu

Na podstawie 11 cech klinicznych pacjenta (wiek, wyniki EKG, ciśnienie, tętno itp.) model przewiduje czy pacjent ma niewydolność serca (0 = zdrowy, 1 = chory).

**Dane:** [Kaggle — Heart Failure Prediction Dataset (fedesoriano, 2021)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

| Właściwość | Wartość |
|---|---|
| Liczba rekordów | 920 pacjentów (725 mężczyzn, 195 kobiet) |
| Liczba cech | 11 cech wejściowych |
| Rozkład klas | 508 chorych (55.2%) vs 410 zdrowych (44.8%) |
| Braki danych | Brak |

---

## Struktura projektu

```
HeartDiseaseClassification/
├── main.py                  # Uruchamia wszystkie eksperymenty
├── src/
│   ├── preprocessing.py     # Wczytanie i przygotowanie danych
│   ├── models.py            # Definicje 6 modeli ML
│   ├── experiment1.py       # Porównanie modeli + test Wilcoxona
│   ├── experiment2.py       # Wpływ parametrów modeli
│   └── experiment3.py       # Wpływ rozmiaru zbioru treningowego
└── results/                 # Wyniki i wykresy (generowane automatycznie)
```

---

## Instalacja

```bash
pip install -r requirements.txt
```

**Wymagania:** Python 3.10+

> **Uwaga:** Wymaga `numpy<2.0`. W przypadku błędu kompatybilności NumPy uruchom:
> ```bash
> pip install "numpy<2.0" numexpr bottleneck --upgrade
> ```

---

## Uruchomienie

```bash
python main.py --data <ścieżka_do_pliku.csv>
```

Przykład:
```bash
python main.py --data data/heart.csv
```

Wszystkie wyniki trafiają automatycznie do folderu `results/`.

---

## Modele

| Model | Opis |
|---|---|
| Logistic Regression | Model liniowy, baseline |
| KNN | Klasyfikacja przez podobieństwo |
| Decision Tree | Drzewo decyzyjne |
| Random Forest | Las lososwy |
| SVM | Maszyna wektorów nośnych (kernel RBF) |
| Gradient Boosting | Boosting gradientowy |

Modele obsługujące `class_weight` mają ustawione `balanced`.

---

## Eksperymenty

### Eksperyment 1 — Porównanie modeli
Wszystkie 6 modeli oceniane za pomocą 5x2 walidacji krzyżowej (RepeatedStratifiedKFold) — metryki: Balanced Accuracy, F1 (macro), Precision, Recall.
Test Wilcoxona sprawdza czy różnice między modelami są statystycznie istotne (α = 0.05).

Wyniki:
- `results/exp1_results.csv` — tabela metryk wszystkich modeli
- `results/exp1_comparison.png` — wykres porównawczy (Balanced Accuracy + F1)
- `results/exp1_wilcoxon.csv` — tabela p-wartości dla każdej pary modeli
- `results/exp1_confusion_matrices.png` — macierze pomyłek (80/20 split, jedna iteracja, wyłącznie ilustracyjnie)

### Eksperyment 2 — Wpływ parametrów
- KNN: testowane różne wartości `n_neighbors` (1, 3, 5, 7, 10, 15)
- Random Forest: testowane różne wartości `n_estimators` (10, 50, 100, 200, 500)

Wyniki:
- `results/exp2_knn.csv`, `results/exp2_knn.png`
- `results/exp2_random_forest.csv`, `results/exp2_random_forest.png`
- `results/exp2_knn_confusion.png` — macierz pomyłek dla najlepszego `n_neighbors`
- `results/exp2_random_forest_confusion.png` — macierz pomyłek dla najlepszego `n_estimators`

### Eksperyment 3 — Wpływ rozmiaru zbioru
Wszystkie modele trenowane na podzbiorach: 20%, 30%, 50%, 70%, 100% danych.
Wynik to krzywa uczenia — jak wyniki zmieniają się wraz z ilością danych.

Wyniki:
- `results/exp3_results_balanced_accuracy.csv`, `results/exp3_results_f1.csv` — tabele metryk dla każdej wielkości podzbioru
- `results/exp3_learning_curves.png` — krzywe uczenia (Balanced Accuracy vs wielkość zbioru)
- `results/exp3_confusion_matrices.png` — macierze pomyłek na pełnym zbiorze (80/20 split, ilustracyjnie)

---

## Metodologia

- **Walidacja krzyżowa:** 5x2 (RepeatedStratifiedKFold, n_splits=5, n_repeats=2, random_state=42) — 10 wyników na model, te same podziały danych dla wszystkich modeli (umożliwia parowany test Wilcoxona)
- **Główne metryki:** Balanced Accuracy (odporna na niezbalansowanie klas) i Macro F1-Score
- **Preprocessing:** one-hot encoding zmiennych kategorycznych, StandardScaler
