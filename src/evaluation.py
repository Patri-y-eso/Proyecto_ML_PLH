"""
evaluation.py
-------------
Evalúa el modelo final con los datos de data/test y genera las métricas.

Entrada:
    models/final_model.pkl, models/final_model_meta.json
    data/test/test.csv
Salida:
    models/evaluation_report.json  (métricas + tabla de deciles)
    (por pantalla) métricas, classification report y matriz de confusión

Uso (desde la raíz del proyecto):
    python src/evaluation.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TEST_DIR = ROOT / "data" / "test"


def precision_at_k(y_true, proba, frac):
    k = max(1, int(len(proba) * frac))
    top = np.argsort(proba)[::-1][:k]
    return float(np.asarray(y_true)[top].mean())


def compute_metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "PR_AUC": average_precision_score(y, proba),
        "ROC_AUC": roc_auc_score(y, proba),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "prec@top1%": precision_at_k(y, proba, 0.01),
        "prec@top5%": precision_at_k(y, proba, 0.05),
    }


def decile_table(y, proba):
    d = pd.DataFrame({"proba": proba, "y": np.asarray(y)})
    d["decil"] = pd.qcut(d["proba"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
    t = d.groupby("decil").agg(proba_min=("proba", "min"), pct_exito=("y", "mean"), n_exitos=("y", "sum"))
    t["lift"] = t["pct_exito"] / d["y"].mean()
    t["pct_exitos_acumulado"] = t["n_exitos"].cumsum() / t["n_exitos"].sum()
    return t


def main():
    model = joblib.load(MODELS / "final_model.pkl")
    with open(MODELS / "final_model_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    features, thr = meta["features"], meta["umbral"]

    test = pd.read_csv(TEST_DIR / "test.csv")
    X, y = test[features], test["exitoso"]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)

    metrics = compute_metrics(y, proba, thr)
    deciles = decile_table(y, proba)

    print(f"Modelo: {meta['modelo']} | umbral: {thr:.3f} | test: {len(test):,} juegos "
          f"({y.mean()*100:.2f}% exitosos)\n")
    print(pd.Series(metrics).round(3).to_string(), "\n")
    print(classification_report(y, pred, target_names=["No exitoso", "Exitoso"]))
    print("Matriz de confusión [filas = real, columnas = predicho]:")
    print(pd.DataFrame(confusion_matrix(y, pred), index=["No exitoso", "Exitoso"],
                       columns=["No exitoso", "Exitoso"]), "\n")
    print("Deciles:")
    print(deciles.round(3).to_string())

    report = {"modelo": meta["modelo"], "umbral": thr, "tasa_base": round(float(y.mean()), 4),
              "metricas": {k: round(float(v), 4) for k, v in metrics.items()},
              "deciles": deciles.round(4).reset_index().to_dict(orient="records")}
    with open(MODELS / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nGuardado models/evaluation_report.json")


if __name__ == "__main__":
    main()
