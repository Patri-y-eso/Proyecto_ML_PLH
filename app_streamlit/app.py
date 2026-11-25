"""
App de Streamlit: predicción del éxito de próximos lanzamientos de Steam.

Uso (desde la raíz del proyecto):
    streamlit run app_streamlit/app.py

Necesita haber ejecutado antes:
    src/data_processing.py, src/training.py, src/evaluation.py
    y notebooks/04_Prediccion_Proximos.ipynb (para data/processed/upcoming_predictions.csv)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_processing import slug  # noqa: E402  (mismo nombrado de columnas que en el entrenamiento)

MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="Steam Hit Predictor", page_icon="🎮", layout="wide")


# ------------------------------------------------------------------ carga
@st.cache_resource
def load_model():
    model = joblib.load(MODELS / "final_model.pkl")
    with open(MODELS / "final_model_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(MODELS / "evaluation_report.json", encoding="utf-8") as f:
        report = json.load(f)
    with open(PROC / "feature_list.json", encoding="utf-8") as f:
        flist = json.load(f)
    return model, meta, report, flist


@st.cache_data
def load_upcoming():
    path = PROC / "upcoming_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


model, meta, report, flist = load_model()
FEATURES, THR, VOCAB = meta["features"], meta["umbral"], flist["vocab"]
DECILES = pd.DataFrame(report["deciles"]).set_index("decil")
BASE = report["tasa_base"] * 100


def decil_de(p):
    for d, row in DECILES.iterrows():
        if p >= row["proba_min"]:
            return int(d)
    return 10


def nivel(d):
    return ("🟢 Alto", "#2a9d8f") if d <= 2 else (("🟡 Medio", "#e9c46a") if d <= 5 else ("🔴 Bajo", "#e76f51"))


def mostrar_resultado(proba):
    d = decil_de(proba)
    pct = DECILES.loc[d, "pct_exito"] * 100
    etiqueta, color = nivel(d)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<h2 style='color:{color};margin:0'>{etiqueta}</h2>Potencial", unsafe_allow_html=True)
    # El signo del texto decide la flecha: "+" verde hacia arriba, "-" roja hacia abajo
    c2.metric("Éxito en juegos similares", f"{pct:.0f} %",
              f"{pct - BASE:+.0f} pp vs media ({BASE:.0f} %) · x{pct / BASE:.1f}")
    c3.metric("Decil del ranking", f"{d} de 10", "1 = mejor", delta_color="off")
    st.caption("“Éxito en juegos similares” = % de juegos históricos con una puntuación del modelo parecida "
               "que alcanzaron ≥ 500 reseñas y ≥ 80 % positivas.")


# ------------------------------------------------------------------ cabecera
st.title("🎮 Steam Hit Predictor")
st.markdown("¿Tendrá éxito un juego **antes de que salga**? Modelo entrenado con ~78.000 juegos de Steam "
            "usando solo información disponible antes del lanzamiento.")

tab1, tab2, tab3 = st.tabs(["📅 Próximos lanzamientos", "🧪 Simulador", "📊 Sobre el modelo"])

# ------------------------------------------------------------------ tab 1
with tab1:
    up = load_upcoming()
    if up is None:
        st.warning("No encuentro data/processed/upcoming_predictions.csv. Ejecuta antes el notebook 04.")
    else:
        up = up.sort_values("proba", ascending=False).reset_index(drop=True)
        up["Potencial"] = up["decil"].map(lambda d: nivel(d)[0])
        vista = up[["name", "release_date", "price", "Potencial", "pct_exito_esperado", "vs_media"]].rename(
            columns={"name": "Juego", "release_date": "Lanzamiento", "price": "Precio ($)",
                     "pct_exito_esperado": "% éxito juegos similares", "vs_media": "x vs media"})
        vista["Precio ($)"] = vista["Precio ($)"].map(lambda p: "sin anunciar" if pd.isna(p) else f"{p:.2f}")
        vista.index += 1
        st.dataframe(vista, height=420)

        st.subheader("Detalle de un juego")
        juego = st.selectbox("Elige un juego", up["name"])
        fila = up[up["name"] == juego].iloc[0]
        precio_conocido = pd.notna(fila["price"])
        precio = st.slider("Precio de lanzamiento (USD)", 0.0, 70.0,
                           float(fila["price"]) if precio_conocido else 19.99, 0.5,
                           help="Steam no publica el precio de muchos juegos hasta el día del lanzamiento: "
                                "prueba distintos precios.")
        if not precio_conocido:
            st.info("Este juego aún no tiene precio anunciado: mueve el slider para ver cómo afecta.")
        X = fila[FEATURES].to_frame().T.astype(float)
        X["price"], X["is_free"] = precio, int(precio == 0)
        mostrar_resultado(model.predict_proba(X)[0, 1])

        c1, c2, c3 = st.columns(3)
        c1.metric("Juegos previos del estudio", int(fila["dev_prev_games"]))
        c2.metric("Éxitos previos del estudio", int(fila["dev_prev_hits"]))
        c3.metric("Idiomas", int(fila["n_languages"]))

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown("Describe un juego (real o imaginario) y mira su potencial.")
    with st.form("simulador"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Precio y plataformas**")
            price = st.number_input("Precio (USD)", 0.0, 100.0, 14.99, 0.5)
            windows = st.checkbox("Windows", True)
            mac = st.checkbox("Mac")
            linux = st.checkbox("Linux")
            month = st.selectbox("Mes de lanzamiento", range(1, 13), index=9)
            weekday = st.selectbox("Día de la semana", range(7), index=3,
                                   format_func=lambda d: ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"][d])
        with c2:
            st.markdown("**Contenido de la ficha**")
            n_languages = st.slider("Nº de idiomas", 1, 30, 5)
            has_spanish = st.checkbox("Incluye español", True)
            has_chinese = st.checkbox("Incluye chino", True)
            achievements = st.slider("Nº de logros", 0, 200, 20)
            n_screenshots = st.slider("Nº de capturas", 0, 40, 10)
            n_movies = st.slider("Nº de vídeos", 0, 10, 2)
            len_short = st.slider("Palabras descripción corta", 0, 80, 35)
            len_detail = st.slider("Palabras descripción larga", 0, 2000, 300)
        with c3:
            st.markdown("**Estudio**")
            dev_prev_games = st.number_input("Juegos previos del desarrollador", 0, 200, 0)
            dev_prev_hits = st.number_input("Éxitos previos del desarrollador", 0, 50, 0)
            pub_prev_games = st.number_input("Juegos previos del publisher", 0, 1000, 0)
            self_published = st.checkbox("Autopublicado", True)

        genres = st.multiselect("Géneros", VOCAB["genre"], default=["Indie", "Action"][: len(VOCAB["genre"])])
        cats = st.multiselect("Características (categorías de Steam)", VOCAB["cat"])
        tags = st.multiselect("Tags (máx. 5)", VOCAB["tag"], max_selections=5)
        enviado = st.form_submit_button("Predecir 🎯", type="primary")

    if enviado:
        row = dict.fromkeys(FEATURES, 0.0)
        row.update({
            "price": price, "is_free": int(price == 0), "windows": int(windows), "mac": int(mac),
            "linux": int(linux), "n_platforms": int(windows) + int(mac) + int(linux),
            "achievements": achievements, "n_languages": n_languages, "has_spanish": int(has_spanish),
            "has_chinese": int(has_chinese), "n_screenshots": n_screenshots, "n_movies": n_movies,
            "len_short_desc": len_short, "len_detailed_desc": len_detail, "release_month": month,
            "release_weekday": weekday, "n_genres": len(genres), "n_categories": len(cats),
            "self_published": int(self_published), "dev_prev_games": dev_prev_games,
            "dev_prev_hits": min(dev_prev_hits, dev_prev_games), "pub_prev_games": pub_prev_games,
        })
        for pref, sel in (("genre", genres), ("cat", cats), ("tag", tags)):
            for v in sel:
                col = f"{pref}_{slug(v)}"
                if col in row:
                    row[col] = 1
        mostrar_resultado(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])

# ------------------------------------------------------------------ tab 3
with tab3:
    m = report["metricas"]
    st.markdown(f"**Modelo:** `{meta['modelo']}` · **Éxito** = ≥ 500 reseñas y ≥ 80 % positivas · "
                f"Tasa base: **{BASE:.1f} %**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC", f"{m['PR_AUC']:.2f}", f"azar ≈ {BASE / 100:.2f}", delta_color="off")
    c2.metric("ROC-AUC", f"{m['ROC_AUC']:.2f}")
    c3.metric("Precisión top 1 %", f"{m['prec@top1%'] * 100:.0f} %")
    c4.metric("Precisión top 5 %", f"{m['prec@top5%'] * 100:.0f} %")

    st.subheader("% de éxito real por decil de puntuación (test)")
    chart = (DECILES["pct_exito"] * 100).rename("% éxito real")
    chart.index = [f"D{d:02d}" for d in chart.index]   # D01..D10: así se ordenan bien en el eje
    st.bar_chart(chart, color="#2a9d8f")
    top20 = DECILES["pct_exitos_acumulado"].iloc[1] * 100
    st.markdown(f"Revisando solo el **20 %** de juegos mejor puntuados se encuentra el **{top20:.0f} %** de los éxitos.")
