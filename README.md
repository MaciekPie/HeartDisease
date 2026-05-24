# BloodTest Classification

Projekt porównujący metody klasyfikacji wieloklasowej na danych medycznych (wyniki badań krwi).

**Metody Sztucznej Inteligencji | Wrocław 2026**

---

## Opis problemu

Na podstawie wyników badań krwi pacjenta model klasyfikuje jego stan zdrowia do jednej z 11 klas:
`healthy`, `cardiovascular`, `diabetes`, `metabolic_syndrome`, `anemia`, `thyroid`,
`vitamin_d_deficiency`, `vitamin_b12_deficiency`, `liver`, `kidney`, `inflammation`.

**Dane:** [Kaggle – Global Blood Test Health Insights 2025–2026](https://www.kaggle.com/datasets/kantesti/global-blood-test-health-insights-2025-2026)

---

## Struktura projektu

```
BloodTestClassification/
├── main.py              # Główny pipeline (preprocessing → trening → ewaluacja)
├── eda.py               # Eksploracyjna analiza danych
├── src/
│   ├── preprocessing.py # Wczytanie, czyszczenie, podział train/val/test, skalowanie
│   ├── models.py        # Definicje 6 modeli ML
│   ├── train.py         # Trening + k-fold cross-validation
│   └── evaluate.py      # Ewaluacja, confusion matrix, wykresy
├── app/
│   └── app.py           # Aplikacja Streamlit
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

- **Podział danych:** 60% trening / 20% walidacja / 20% test (stratyfikowany)
- **Walidacja krzyżowa:** 5-fold StratifiedKFold na zbiorze treningowym
- **Metryka wyboru modelu:** macro F1-score (odporna na niezbalansowanie klas)
- **Preprocessing:** usunięcie kolumn geograficznych, one-hot encoding, StandardScaler

---

## Wyniki

Po uruchomieniu `main.py` wyniki ewaluacji dostępne są w:
- `results/metrics.csv` — tabela z accuracy, F1, precision, recall dla każdego modelu
- `results/models_comparison.png` — wykres porównawczy
- `results/confusion_<ModelName>.png` — macierz pomyłek dla każdego modelu
- `results/cv_results.csv` — wyniki cross-validation
