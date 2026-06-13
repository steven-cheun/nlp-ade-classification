from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")),
    ])


def run_grid_search(X_train, y_train):
    pipeline = build_pipeline()
    param_grid = {
        "tfidf__max_features": [20000, 50000, 100000],
        "clf__C": [0.1, 1, 10],
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)


def load_pipeline(path):
    return joblib.load(path)
