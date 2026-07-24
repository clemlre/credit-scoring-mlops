"""Lance les deux baselines de l'Étape 2 (LogReg + RF) avec tracking MLFlow.

Équivalent au notebook 02_mlflow_setup.ipynb mais exécutable depuis le shell.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import training
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

training.setup_mlflow()
X, y = training.load_training_data()
print(f"Données : {X.shape}, y_mean={y.mean():.4f}")

logreg = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)),
])
print("\n[1/2] Logistic Regression")
training.cv_run(
    logreg, X, y,
    run_name="logreg_baseline", n_splits=3,
    extra_params={"model": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000},
    extra_tags={"step": "2", "family": "linear"},
)

rf = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42,
    )),
])
print("\n[2/2] Random Forest")
training.cv_run(
    rf, X, y,
    run_name="rf_baseline", n_splits=3,
    extra_params={"model": "RandomForest", "n_estimators": 100, "max_depth": 10, "class_weight": "balanced"},
    extra_tags={"step": "2", "family": "tree-ensemble"},
)

print("\nFini.")