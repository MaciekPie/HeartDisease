from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models() -> dict:
    """
    Returns a dictionary of 6 tabular classification models.
    Models that support class_weight are set to 'balanced'
    due to potential class imbalance in medical/financial data.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,   # required for predict_proba in the app
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
        ),
    }
