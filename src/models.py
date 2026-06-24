from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def get_models():
    # Słownik modeli: klucz = nazwa (do wykresów), wartość = obiekt modelu
    # class_weight="balanced" sprawia, że model zwraca większą uwagę
    # na rzadszą klasę — zapobiega ignorowaniu mniejszościowej grupy
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, # maksymalna liczba iteracji optymalizacji
            class_weight="balanced",
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5 # domyślna liczba sąsiadów
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, # liczba drzew w lesie
            class_weight="balanced",
            random_state=42
        ),
        "SVM": SVC(
            class_weight="balanced",
            probability=True, # wymagane do obliczania prawdopodobieństw
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
    }
    return models
