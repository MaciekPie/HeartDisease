from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def get_models():
    # 6 different models to compare
    # class_weight="balanced" tells the model that both classes are equally important
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ),
        "SVM": SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
    }
    return models
