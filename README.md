# BloodTest Classification

Projekt porównujący metody klasyfikacji wieloklasowej na danych medycznych (wyniki badań krwi).

**Metody Sztucznej Inteligencji | Wrocław 2026**

---

## Opis problemu

Na podstawie wyników badań krwi pacjenta model klasyfikuje jego stan zdrowia do jednej z 11 klas:
`healthy`, `cardiovascular`, `diabetes`, `metabolic_syndrome`, `anemia`, `thyroid`,
`vitamin_d_deficiency`, `vitamin_b12_deficiency`, `liver`, `kidney`, `inflammation`.

Projekt obsługuje **3 zestawy danych** — przełączanie odbywa się przy minimalnych zmianach w kodzie (patrz sekcja poniżej).

---

## Obsługiwane zestawy danych

| # | Zestaw danych | Zadanie | Rozmiar | Link |
|---|---|---|---|---|
| 1 | Global Blood Test Health Insights | Wieloklasowa (11 stanów) | ~130 wierszy | [Kaggle](https://www.kaggle.com/datasets/kantesti/global-blood-test-health-insights-2025-2026) |
| 2 | Heart Failure Prediction | Binarna (choroba serca tak/nie) | 920 wierszy | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| 3 | Credit Card Fraud Detection | Binarna (oszustwo tak/nie) | 284k wierszy | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## Struktura projektu

```
BloodTestClassification/
├── main.py              # Główny pipeline (preprocessing → trening → ewaluacja)
├── eda.py               # Eksploracyjna analiza danych
├── src/
│   ├── preprocessing.py # Wczytanie, czyszczenie, podział train/val/test, skalowanie + sekcje datasetów
│   ├── models.py        # Definicje 6 modeli ML
│   ├── train.py         # Trening + k-fold cross-validation
│   └── evaluate.py      # Ewaluacja, confusion matrix, wykresy
├── app/
│   └── app.py           # Aplikacja Streamlit + sekcje datasetów
├── models/              # Zapisany model, scaler, feature_names (po uruchomieniu main.py)
├── results/             # Metryki CSV, wykresy (po uruchomieniu main.py)
│   └── eda/             # Wykresy EDA (po uruchomieniu eda.py)
└── requirements.txt
```

---

## Instalacja

```bash
pip install -r requirements.txt
```

> **Uwaga:** Wymaga `numpy<2.0`. W przypadku błędu kompatybilności NumPy uruchom:
> ```bash
> pip install "numpy<2.0" numexpr bottleneck --upgrade
> ```

---

## Przełączanie zestawów danych

Aby przełączyć się na inny dataset, edytuj **2 pliki**:

### 1. `src/preprocessing.py`
Zakomentuj aktywną sekcję i odkomentuj wybraną:
```python
# ── DATASET 1: Blood Test Classification [ACTIVE] ────────────────
CONDITION_COLS = [ ... ]   # zakomentuj tę sekcję

# ── DATASET 2: Heart Disease Prediction [INACTIVE] ───────────────
# CONDITION_COLS = []      # odkomentuj tę sekcję
# DROP_COLS = []
# def create_target(df):
#     ...
```

### 2. `app/app.py`
Tak samo — zakomentuj blok Dataset 1, odkomentuj wybrany dataset.

### 3. `main.py`
Zmień jedną linię:
```python
N_SPLITS = 3   # Dataset 1 (mały zbiór danych)
N_SPLITS = 5   # Dataset 2 lub 3
```

`eda.py`, `train.py`, `evaluate.py`, `models.py` — **bez zmian**.

---

## Uruchomienie

### 1. EDA (opcjonalne, ale zalecane)
```bash
python eda.py --data <ścieżka_do_pliku.csv>
# Wykresy → results/eda/
```

### 2. Pełny pipeline ML
```bash
python main.py --data <ścieżka_do_pliku.csv>
# Wyniki → results/metrics.csv, results/confusion_*.png
# Model  → models/best_model.pkl
```

### 3. Aplikacja
```bash
streamlit run app/app.py
```

---

## Modele

| Model | Opis |
|---|---|
| Logistic Regression | Model liniowy, baseline |
| K-Nearest Neighbors | Klasyfikacja przez podobieństwo |
| Decision Tree | Drzewo decyzyjne |
| Random Forest | Ensemble drzew decyzyjnych |
| SVM | Maszyna wektorów nośnych (kernel RBF) |
| Gradient Boosting | Boosting gradientowy (XGBoost-style) |

Wszystkie modele obsługujące `class_weight` mają ustawione `balanced` ze względu na możliwe niezbalansowanie klas.

---

## Metodologia

- **Podział danych:** 60% trening / 20% walidacja / 20% test (stratyfikowany tam gdzie możliwe)
- **Walidacja krzyżowa:** Stratyfikowany k-fold na zbiorze treningowym (k=3 dla Dataset 1, k=5 dla Dataset 2 i 3)
- **Metryka wyboru modelu:** macro F1-score (odporna na niezbalansowanie klas)
- **Preprocessing:** usunięcie zbędnych kolumn, one-hot encoding, StandardScaler (dopasowany tylko na train)

---

## Wyniki

Po uruchomieniu `main.py` wyniki ewaluacji dostępne są w:

- `results/metrics.csv` — tabela z accuracy, F1, precision, recall dla każdego modelu
- `results/models_comparison.png` — wykres porównawczy
- `results/confusion_<ModelName>.png` — macierz pomyłek dla każdego modelu
- `results/cv_results.csv` — wyniki cross-validation
