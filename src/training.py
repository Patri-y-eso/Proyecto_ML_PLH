"""
training.py
-----------
Entrena los modelos con los datos de data/processed, guarda los conjuntos de
train/test usados y los modelos entrenados.

Entrada:
    data/processed/games_processed.csv
    data/processed/feature_list.json
Salida:
    data/train/train.csv, data/test/test.csv
    models/trained_model_<n>.pkl, models/models_index.json
    models/final_model.pkl, models/final_model_meta.json

Uso (desde la raíz del proyecto):
    python src/training.py                        # los 6 modelos
    python src/training.py --modelos xgboost lightgbm
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TRAIN_DIR, TEST_DIR = ROOT / "data" / "train", ROOT / "data" / "test"
MODELS = ROOT / "models"
RS = 42

LOG_CANDIDATES = ["price", "achievements", "n_languages", "n_screenshots", "n_movies",
                  "len_short_desc", "len_detailed_desc", "dev_prev_games", "dev_prev_hits", "pub_prev_games"]


def make_preprocessor(features):
    log_cols = [c for c in LOG_CANDIDATES if c in features]
    other_cols = [c for c in features if c not in log_cols]
    log_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                         ("sc", StandardScaler())])
    other_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    return ColumnTransformer([("log", log_pipe, log_cols), ("other", other_pipe, other_cols)],
                             verbose_feature_names_out=False)


def get_models(spw):
    """Modelo + rejilla de hiperparámetros. El orden define el n de trained_model_n.pkl."""
    return {
        "logreg": (LogisticRegression(max_iter=3000, class_weight="balanced"),
                   {"clf__C": [0.01, 0.1, 1, 10]}),
        "knn": (KNeighborsClassifier(),
                {"clf__n_neighbors": [25, 75, 150], "clf__weights": ["uniform", "distance"]}),
        "decision_tree": (DecisionTreeClassifier(class_weight="balanced", random_state=RS),
                          {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [20, 100]}),
        "random_forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=RS),
                          {"clf__max_depth": [10, 20, None], "clf__min_samples_leaf": [5, 20]}),
        "xgboost": (XGBClassifier(n_estimators=400, scale_pos_weight=spw, eval_metric="logloss",
                                  tree_method="hist", random_state=RS),
                    {"clf__max_depth": [4, 6], "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8, 1.0]}),
        "lightgbm": (LGBMClassifier(n_estimators=400, class_weight="balanced", random_state=RS, verbose=-1),
                     {"clf__num_leaves": [31, 63], "clf__learning_rate": [0.05, 0.1]}),
    }


def best_f1_threshold(model, X, y, cv):
    """Umbral que maximiza el F1 con predicciones out-of-fold del train."""
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def main(seleccion):
    for d in (TRAIN_DIR, TEST_DIR, MODELS):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROC / "games_processed.csv")
    with open(PROC / "feature_list.json", encoding="utf-8") as f:
        meta = json.load(f)
    features, target = meta["features"], meta["target"]

    train, test = train_test_split(df, test_size=0.2, stratify=df[target], random_state=RS)
    train.to_csv(TRAIN_DIR / "train.csv", index=False)
    test.to_csv(TEST_DIR / "test.csv", index=False)
    X_train, y_train = train[features], train[target]
    print(f"Train {X_train.shape} | Test {test.shape} | % éxito {y_train.mean()*100:.2f}")

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    modelos = get_models(spw)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)

    indice, resultados = {}, {}
    for i, (name, (est, grid)) in enumerate(modelos.items(), 1):
        if seleccion and name not in seleccion:
            continue
        t0 = time.time()
        pipe = Pipeline([("prep", make_preprocessor(features)), ("clf", est)])
        gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)

        path = MODELS / f"trained_model_{i}.pkl"
        joblib.dump(gs.best_estimator_, path)
        indice[path.name] = {"modelo": name, "cv_PR_AUC": round(gs.best_score_, 4), "best_params": gs.best_params_}
        resultados[name] = (gs.best_score_, gs.best_estimator_, gs.best_params_)
        print(f"  {name:14s} CV PR-AUC={gs.best_score_:.3f} ({time.time()-t0:.0f}s) {gs.best_params_}")

    with open(MODELS / "models_index.json", "w", encoding="utf-8") as f:
        json.dump(indice, f, indent=2, default=str)

    final_name = max(resultados, key=lambda k: resultados[k][0])
    cv_score, final_model, params = resultados[final_name]
    thr = best_f1_threshold(final_model, X_train, y_train, cv)

    joblib.dump(final_model, MODELS / "final_model.pkl")
    with open(MODELS / "final_model_meta.json", "w", encoding="utf-8") as f:
        json.dump({"modelo": final_name, "umbral": thr, "features": features,
                   "cv_PR_AUC": round(cv_score, 4), "best_params": {k: str(v) for k, v in params.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"\nModelo final: {final_name} (CV PR-AUC {cv_score:.3f}, umbral {thr:.3f}) -> models/final_model.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrena los modelos y guarda el final")
    ap.add_argument("--modelos", nargs="*", default=None,
                    help="subconjunto: logreg knn decision_tree random_forest xgboost lightgbm")
    main(ap.parse_args().modelos)
