"""
data_processing.py
------------------
Procesa los datos de data/raw y guarda el resultado en data/processed.

Entrada:
    data/raw/<histórico Kaggle>.csv
    data/raw/upcoming_enriched.json   (generado en notebooks/01_Fuentes.ipynb)
Salida:
    data/processed/games_processed.csv
    data/processed/upcoming_processed.csv
    data/processed/feature_list.json

Uso (desde la raíz del proyecto):
    python src/data_processing.py --hist games.csv
"""
import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# ---------------------------------------------------------------- parámetros
FECHA_INICIO = pd.Timestamp("2015-01-01")
FECHA_CORTE = pd.Timestamp("2025-11-25")   # juegos con >= 12 meses a la venta
MIN_RESENAS = 500
MIN_POSITIVAS = 0.80
TOP_GENRES, TOP_CATS, TOP_TAGS = 20, 25, 40
N_TAGS_JUEGO = 5
CATS_EXCLUIR = {"Steam Trading Cards"}     # se añade tras el lanzamiento (leakage)

BASE_FEATURES = [
    "price", "is_free", "windows", "mac", "linux", "n_platforms",
    "achievements", "n_languages", "has_spanish", "has_chinese",
    "n_screenshots", "n_movies", "len_short_desc", "len_detailed_desc",
    "release_month", "release_weekday", "n_genres", "n_categories", "self_published",
    "dev_prev_games", "dev_prev_hits", "pub_prev_games",
]
META = ["appid", "name", "release_date"]


# ---------------------------------------------------------------- parseo
def parse_list(x):
    """Texto JSON / repr de Python / 'a, b, c' -> lista o dict."""
    if isinstance(x, (list, dict)):
        return x
    if x is None or pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(s)
        except Exception:
            pass
    return [t.strip() for t in s.split(",") if t.strip()]


def to_names(x, top=None):
    """Lista de nombres; si es un dict de votos (tags) ordena por votos."""
    v = parse_list(x)
    if isinstance(v, dict):
        v = [k for k, _ in sorted(v.items(),
                                  key=lambda kv: -kv[1] if isinstance(kv[1], (int, float)) else 0)]
    names = []
    for e in v if isinstance(v, list) else []:
        n = (e.get("description") or e.get("name")) if isinstance(e, dict) else e
        if n:
            names.append(str(n).strip())
    return names[:top] if top else names


def parse_languages(x):
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if x is None or pd.isna(x):
        return []
    s = re.sub(r"<[^>]+>", "", str(x))
    s = s.split("languages with full audio support")[0].replace("*", "")
    v = parse_list(s)
    return [str(i).strip() for i in v if str(i).strip()] if isinstance(v, list) else []


def count_items(x):
    v = parse_list(x)
    return len(v) if isinstance(v, (list, dict)) else 0


def text_len(x):
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return 0
    return len(re.sub(r"<[^>]+>", " ", str(x)).split())


def to_bool_int(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).astype(int)


def fecha_exacta(s):
    """Solo fechas con día ('Nov 24, 2026'); 'Q1 2027' o '2027' -> NaT."""
    s = s.astype("string")
    return pd.to_datetime(s.where(s.str.contains(",", na=False)), errors="coerce", format="mixed")


def slug(s):
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


# ---------------------------------------------------------------- limpieza + target
def clean_hist(hist_raw: pd.DataFrame) -> pd.DataFrame:
    hist = hist_raw.rename(columns={"app_id": "appid"}).copy()
    hist = hist.drop_duplicates("appid")
    hist = hist[hist["price_status"] != "unavailable"]
    hist["release_date"] = pd.to_datetime(hist["release_date"], errors="coerce", format="mixed")
    hist = hist.dropna(subset=["release_date", "name"])
    for c in ["positive", "negative"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0)

    hist["n_reviews"] = hist["positive"] + hist["negative"]
    hist["pct_positive"] = np.where(hist["n_reviews"] > 0, hist["positive"] / hist["n_reviews"], np.nan)
    hist["exitoso"] = ((hist["n_reviews"] >= MIN_RESENAS) &
                       (hist["pct_positive"] >= MIN_POSITIVAS)).astype(int)
    return hist


# ---------------------------------------------------------------- features
def build_base(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    h = fuente == "hist"
    out = pd.DataFrame(index=df.index)
    out["appid"] = df["appid"].astype("int64")
    out["name"] = df["name"]

    if h:
        out["release_date"] = df["release_date"]
        out["is_free"] = (df["price_status"] == "free").astype(int)
        out["n_screenshots"] = df["screenshots"].map(count_items)
        out["n_movies"] = df["movies"].map(count_items)
    else:
        fecha = fecha_exacta(df["release_date_store"])
        if "release_date_api" in df:
            fecha = fecha.fillna(fecha_exacta(df["release_date_api"]))
        out["release_date"] = fecha
        out["is_free"] = to_bool_int(df["is_free"])
        out["n_screenshots"] = pd.to_numeric(df["n_screenshots"], errors="coerce").fillna(0)
        out["n_movies"] = pd.to_numeric(df["n_movies"], errors="coerce").fillna(0)

    out["price"] = pd.to_numeric(df["price"], errors="coerce")
    out.loc[out["is_free"] == 1, "price"] = 0.0
    for p in ["windows", "mac", "linux"]:
        out[p] = to_bool_int(df[p])
    out["n_platforms"] = out[["windows", "mac", "linux"]].sum(axis=1)

    out["achievements"] = pd.to_numeric(df["achievements"], errors="coerce").fillna(0)
    if not h:
    # Steam no publica los logros hasta cerca del lanzamiento: 0 = desconocido
        out["achievements"] = out["achievements"].replace(0, np.nan)

    langs = df["supported_languages"].map(parse_languages)
    out["n_languages"] = langs.map(len)
    out["has_spanish"] = langs.map(lambda l: int(any("Spanish" in x for x in l)))
    out["has_chinese"] = langs.map(lambda l: int(any("Chinese" in x for x in l)))

    out["len_short_desc"] = df["short_description"].map(text_len)
    out["len_detailed_desc"] = df["detailed_description"].map(text_len)
    out["release_month"] = out["release_date"].dt.month
    out["release_weekday"] = out["release_date"].dt.weekday

    out["genres_l"] = df["genres"].map(to_names)
    out["categories_l"] = df["categories"].map(lambda x: [c for c in to_names(x) if c not in CATS_EXCLUIR])
    out["tags_l"] = df["tags"].map(lambda x: to_names(x, top=N_TAGS_JUEGO))
    out["developers_l"] = df["developers"].map(to_names)
    out["publishers_l"] = df["publishers"].map(to_names)

    out["n_genres"] = out["genres_l"].map(len)
    out["n_categories"] = out["categories_l"].map(len)
    out["self_published"] = [int(bool(set(d) & set(p))) for d, p in zip(out["developers_l"], out["publishers_l"])]
    out["dev0"] = out["developers_l"].map(lambda l: l[0] if l else "UNKNOWN")
    out["pub0"] = out["publishers_l"].map(lambda l: l[0] if l else "UNKNOWN")
    return out


def add_experience(hist_f: pd.DataFrame, up_f: pd.DataFrame):
    """Juegos y éxitos previos del desarrollador / publisher (solo juegos anteriores)."""
    hist_f = hist_f.sort_values("release_date").copy()
    g = hist_f.groupby("dev0")
    hist_f["dev_prev_games"] = g.cumcount()
    hist_f["dev_prev_hits"] = g["exitoso"].cumsum() - hist_f["exitoso"]
    hist_f["pub_prev_games"] = hist_f.groupby("pub0").cumcount()

    dev_tot = hist_f.groupby("dev0").agg(games=("appid", "size"), hits=("exitoso", "sum"))
    pub_tot = hist_f.groupby("pub0").size()
    up_f = up_f.copy()
    up_f["dev_prev_games"] = up_f["dev0"].map(dev_tot["games"]).fillna(0).astype(int)
    up_f["dev_prev_hits"] = up_f["dev0"].map(dev_tot["hits"]).fillna(0).astype(int)
    up_f["pub_prev_games"] = up_f["pub0"].map(pub_tot).fillna(0).astype(int)

    for df_ in (hist_f, up_f):
        df_.loc[df_["dev0"] == "UNKNOWN", ["dev_prev_games", "dev_prev_hits"]] = 0
        df_.loc[df_["pub0"] == "UNKNOWN", "pub_prev_games"] = 0
    return hist_f, up_f


def top_vocab(series, n):
    return series.explode().dropna().value_counts().head(n).index.tolist()


def one_hot(df, vocab):
    src = {"genre": "genres_l", "cat": "categories_l", "tag": "tags_l"}
    cols = {}
    for pref, words in vocab.items():
        sets = df[src[pref]].map(set)
        for v in words:
            cols[f"{pref}_{slug(v)}"] = sets.map(lambda s, v=v: int(v in s))
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


# ---------------------------------------------------------------- main
def main(hist_name: str, up_name: str):
    PROC.mkdir(parents=True, exist_ok=True)
    print("Cargando datos...")
    hist = clean_hist(pd.read_csv(RAW / hist_name, low_memory=False))
    up_raw = pd.read_json(RAW / up_name)

    print("Construyendo features...")
    hist_f = build_base(hist, "hist")
    hist_f["exitoso"] = hist["exitoso"]
    hist_f["n_reviews"] = hist["n_reviews"]
    up_f = build_base(up_raw, "up")
    hist_f, up_f = add_experience(hist_f, up_f)

    mask = hist_f["release_date"].between(FECHA_INICIO, FECHA_CORTE) & (hist_f["tags_l"].map(len) > 0)
    train_df = hist_f[mask].copy()

    vocab = {"genre": top_vocab(train_df["genres_l"], TOP_GENRES),
             "cat": top_vocab(train_df["categories_l"], TOP_CATS),
             "tag": top_vocab(train_df["tags_l"], TOP_TAGS)}
    train_df = one_hot(train_df, vocab)
    up_f = one_hot(up_f, vocab)

    ohe = [c for c in train_df if c.startswith(("genre_", "cat_", "tag_"))]
    features = BASE_FEATURES + ohe

    train_df[META + features + ["exitoso", "n_reviews"]].to_csv(PROC / "games_processed.csv", index=False)
    up_f[META + features].to_csv(PROC / "upcoming_processed.csv", index=False)
    with open(PROC / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump({"features": features, "target": "exitoso", "vocab": vocab,
                   "params": {"MIN_RESENAS": MIN_RESENAS, "MIN_POSITIVAS": MIN_POSITIVAS,
                              "FECHA_INICIO": str(FECHA_INICIO.date()), "FECHA_CORTE": str(FECHA_CORTE.date())}},
                  f, ensure_ascii=False, indent=2)

    print(f"OK -> {len(features)} features | train: {len(train_df):,} "
          f"({train_df['exitoso'].mean()*100:.2f}% exitosos) | próximos: {len(up_f)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Procesa data/raw -> data/processed")
    ap.add_argument("--hist", default="games.csv", help="csv histórico de Kaggle dentro de data/raw")
    ap.add_argument("--upcoming", default="upcoming_enriched.json", help="json de próximos dentro de data/raw")
    args = ap.parse_args()
    main(args.hist, args.upcoming)
