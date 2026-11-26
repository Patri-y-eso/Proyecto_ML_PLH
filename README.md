# 🎮 Steam Hit Predictor

**¿Se puede saber si un videojuego de Steam va a tener éxito _antes_ de que salga a la venta?**

Proyecto final de Machine Learning (Bootcamp de Data Science). El modelo predice el potencial de éxito de un juego usando **solo información disponible antes de su lanzamiento**: precio, géneros, idiomas, ficha de la tienda, trayectoria del estudio, etc. Después se aplica a **36 juegos que se lanzan entre noviembre y diciembre de 2026**.

![App de Streamlit](docs/img/app.png)

---

## 1. El problema

En Steam se publican cada año más juegos: unos 2.500 en 2015, cerca de 25.000 en 2025 y casi 29.000 en 2026. Con tanta competencia, solo **1 de cada 14 juegos** triunfa.

Detectar pronto qué juegos tienen potencial es útil para:

- **Publishers e inversores**: decidir en qué juegos invertir o cuáles fichar.
- **Plataformas y medios**: elegir qué lanzamientos destacar.
- **Estudios**: saber qué aspectos de su juego y de su ficha pesan más en el éxito.

**Tipo de problema:** clasificación binaria supervisada con clases desbalanceadas.

**Definición de éxito:** un juego es *exitoso* si alcanza **≥ 500 reseñas** (≈ 15.000-30.000 copias, según la regla de Boxleiter) **y ≥ 80 % de reseñas positivas** (la etiqueta "Muy positivas" de Steam).

---

## 2. Los datos

| Fuente | Contenido | Uso |
|---|---|---|
| [Kaggle – Steam Games Dataset (daily updates)](https://www.kaggle.com/datasets/hubertsidorowicz/steam-games-dataset-daily-updates) | 150.863 juegos × 42 columnas | Entrenamiento |
| [API Steam Upcoming Games (cleanedweb)](https://cleanedweb.com/catalog/steam-upcoming-games-api/) | 36 próximos lanzamientos | Predicción |
| API pública de la tienda de Steam (`store.steampowered.com/api/appdetails`) | Ficha completa de los 36 próximos | Enriquecimiento |

**Decisiones clave:**

- **Sin fugas de información.** Las reseñas, los propietarios, las horas de juego, Metacritic, los DLCs y la categoría *Steam Trading Cards* solo existen después del lanzamiento, así que se excluyen como variables.
- **Ventana de entrenamiento: 2015 – noviembre de 2025.** Solo entran juegos con al menos 12 meses a la venta, que ya han tenido tiempo de acumular reseñas.
- **Solo juegos con tags.** El 0,0 % de los juegos sin tags es exitoso, porque los tags los votan los jugadores.
- **Resultado:** **78.593 juegos**, de los que un **7,16 % son exitosos**, descritos con **107 variables**:
  - precio y si es gratuito
  - plataformas
  - nº de idiomas, si incluye español o chino
  - logros
  - nº de capturas y vídeos
  - longitud de las descripciones
  - mes y día de la semana del lanzamiento
  - autopublicación
  - experiencia del desarrollador y del publisher
  - one-hot de los 20 géneros, 25 categorías y 40 tags más frecuentes

> 📦 **Versión ligera del histórico.** El CSV completo de Kaggle pesa ~1 GB y no se sube al repositorio. En su lugar se incluye `data/raw/steam_games_lite.csv.gz`, generado en `01_Fuentes.ipynb`. Conserva las 150.863 filas y solo las columnas que usa el proyecto, y sustituye las columnas pesadas (descripciones en HTML y listas de capturas y vídeos) por sus recuentos ya calculados. Con él se obtienen **exactamente los mismos datos procesados** que con el CSV completo. Para regenerarlo, descarga el CSV de Kaggle en `data/raw/steam_games.csv` y ejecuta la sección 6 del notebook `01`.

---

## 3. Métrica

Con solo un 7 % de positivos, el *accuracy* no sirve: un modelo que diga siempre "no exitoso" acierta el 92,8 % de las veces.

- **Métrica principal: PR-AUC (Average Precision).** Mide cómo de bien el modelo encuentra los éxitos sin llenarse de falsas alarmas. El azar puntúa 0,07.
- **Métricas complementarias:** ROC-AUC, precision, recall, F1 y una métrica de negocio, **precision@top-k**: de los juegos mejor puntuados, qué % triunfa de verdad.

---

## 4. Resultados

Se entrenaron 6 modelos supervisados con `Pipeline` + `GridSearchCV`, con validación cruzada estratificada de 3 folds:

| Modelo | CV PR-AUC | Test PR-AUC | ROC-AUC |
|---|---|---|---|
| **XGBoost** ✅ | **0,479** | **0,470** | **0,907** |
| LightGBM | 0,471 | 0,475 | 0,907 |
| Random Forest | 0,449 | 0,445 | 0,901 |
| Regresión logística | 0,437 | 0,429 | 0,895 |
| KNN | 0,370 | 0,365 | 0,859 |
| Árbol de decisión | 0,348 | 0,354 | 0,862 |
| *Baseline aleatorio* | – | *0,071* | *0,497* |

**Modelo final: XGBoost.** Su PR-AUC es unas **6,6 veces la del azar**, y no hay sobreajuste: CV y test dan prácticamente lo mismo.

### En lenguaje de negocio

- De los juegos que el modelo pone en el **10 % superior**, el **42 % triunfa**, frente al 7 % al azar (**x5,9**).
- Revisando solo el **20 % mejor puntuado** se encuentra el **81 % de todos los éxitos**.
- Con el umbral elegido (0,774): de 1.469 juegos marcados como prometedores, **641 triunfan (44 %)**.

### Qué variables pesan más (permutation importance + SHAP)

| A favor del éxito ⬆️ | En contra ⬇️ |
|---|---|
| Estudio con éxitos previos | Muchos juegos previos sin ningún éxito |
| Precio medio-alto (refleja ambición del proyecto) | Precio muy bajo |
| Muchos idiomas, sobre todo chino | Pocas capturas en la ficha |
| Logros, tráilers, Steam Cloud | Tags saturados: Casual, Arcade, 2D Platformer |
| Multiplayer, Story Rich, Free to Play | Early Access |

### Robustez

Sin las variables de historial del estudio (posible fuga leve), la PR-AUC solo baja a **0,43**. Con estudios debutantes, la PR-AUC es **0,39**, unas 7 veces su tasa base. Rendimiento honesto del modelo: **PR-AUC entre 0,43 y 0,47**.

### Predicción de los 36 próximos lanzamientos

- **20 juegos con potencial alto, 15 medio y 1 bajo.**
- Se esperan **≈ 6,4 éxitos (≈ 18 %)**, más del doble de la media de Steam.
- **Favoritos** (decil 1, 42 % de éxito en juegos similares): *Arctic Drive, Reshine, No Asteroids, Warhammer 40,000: Dawn of War IV, Drunken Goddess Reflux, Blind Descent, Dumpster Gang, Me and You* y *Shark Mart*.
- ¿Por qué tantos con potencial alto? Los 36 son lanzamientos **destacados** de Steam (9 idiomas de mediana frente a 1 en el histórico). Los 712 juegos históricos con un perfil parecido también caen en un 77 % en los deciles altos, y el 29 % triunfó de verdad.

---

## 5. Estructura del repositorio

```
Proyecto_ML_PLH/
├── data/
│   ├── raw/          # steam_games_lite.csv.gz + API de próximos + fichas de la tienda de Steam
│   ├── processed/    # games_processed.csv, upcoming_processed.csv, feature_list.json,
│   │                 # upcoming_predictions.csv
│   ├── train/        # train.csv
│   └── test/         # test.csv
├── notebooks/
│   ├── 01_Fuentes.ipynb                  # Adquisición y unión de fuentes
│   ├── 02_LimpiezaEDA.ipynb              # Limpieza, target, feature engineering y EDA
│   ├── 03_Entrenamiento_Evaluacion.ipynb # 6 modelos, GridSearch, evaluación, SHAP, robustez
│   └── 04_Prediccion_Proximos.ipynb      # Predicción de los 36 próximos lanzamientos
├── src/
│   ├── data_processing.py   # data/raw -> data/processed
│   ├── training.py          # entrena los 6 modelos y guarda el final
│   └── evaluation.py        # métricas sobre data/test
├── models/
│   ├── trained_model_1.pkl ... trained_model_6.pkl
│   ├── final_model.pkl            # XGBoost (Pipeline completo)
│   ├── final_model_meta.json      # features y umbral
│   ├── models_index.json          # qué modelo es cada .pkl
│   └── evaluation_report.json     # métricas y deciles
├── app_streamlit/
│   ├── app.py
│   └── requirements.txt
├── docs/
│   ├── presentacion.pptx
│   └── memoria.md
└── README.md
```

| Fichero | Modelo |
|---|---|
| `trained_model_1.pkl` | Regresión logística |
| `trained_model_2.pkl` | KNN |
| `trained_model_3.pkl` | Árbol de decisión |
| `trained_model_4.pkl` | Random Forest |
| `trained_model_5.pkl` | XGBoost |
| `trained_model_6.pkl` | LightGBM |
| `final_model.pkl` | XGBoost (modelo elegido) |

---

## 6. Cómo ejecutar el proyecto

### Requisitos

Python 3.12.

```bash
pip install -r app_streamlit/requirements.txt
```

### Opción A · Scripts (reproduce todo desde la terminal)

Desde la **raíz del proyecto**:

```bash
# 1. Procesar los datos: data/raw -> data/processed
#    (por defecto usa data/raw/steam_games_lite.csv.gz)
python src/data_processing.py

# 2. Entrenar los 6 modelos y guardar el final (~5 min)
python src/training.py

# 3. Evaluar el modelo final sobre el test
python src/evaluation.py
```

Los scripts dan exactamente las mismas cifras que los notebooks (semilla fija: `random_state = 42`).

### Opción B · Notebooks

Ejecutar en orden los notebooks `01` → `02` → `03` → `04`.

> El notebook `01` descarga las fichas de la tienda de Steam (~1,6 s por juego). Su resultado ya está guardado en `data/raw/upcoming_enriched.json`, así que no hace falta volver a ejecutarlo.

### Predicción de los próximos lanzamientos

Ejecutar `notebooks/04_Prediccion_Proximos.ipynb`, que genera `data/processed/upcoming_predictions.csv`.

### App de Streamlit

```bash
streamlit run app_streamlit/app.py
```

La app tiene tres pestañas:

- **📅 Próximos lanzamientos**: ranking de los 36 juegos y detalle de cada uno, con un slider para probar el precio si aún no está anunciado.
- **🧪 Simulador**: describe cualquier juego y obtén su potencial.
- **📊 Sobre el modelo**: métricas y % de éxito real por decil.

---

## 7. Limitaciones

- **Precio y logros no publicados.** 27 de los 36 próximos no tienen precio anunciado y 29 no tienen logros publicados. Se imputan con la mediana del entrenamiento. El análisis de sensibilidad muestra que los favoritos se mantienen en el decil 1 a cualquier precio.
- **El mercado cambia.** La tasa de éxito en Steam ha caído del ~40 % (2012-2013) al ~3-4 % (2023-2024). Los porcentajes absolutos pueden ser optimistas para 2026; el ranking relativo sigue siendo válido.
- **Puntuaciones no calibradas.** Por el tratamiento del desbalance (`scale_pos_weight`), la salida del modelo no es una probabilidad real. Se traduce a "% de éxito en juegos similares" usando los deciles del test.
- **Fuga leve.** El éxito de los juegos previos de un estudio se mide con las reseñas actuales. Su efecto está acotado (PR-AUC 0,43 sin esas variables).
- **Muestra de predicción pequeña y seleccionada.** Son 36 juegos destacados, no una muestra al azar.

## 8. Próximos pasos

- **Validación real.** Los 36 juegos salen entre el 24/11 y el 15/12/2026. En 6-12 meses se pueden comprobar sus reseñas y medir el acierto del modelo fuera de muestra.
- **Más datos de prelanzamiento.** Wishlists, seguidores de la página de la tienda, demos en Steam Next Fest, visualizaciones de tráilers.
- **NLP sobre las descripciones** (TF-IDF o embeddings) y análisis de las imágenes de cabecera.
- **Calibrar las probabilidades** (Platt / isotónica) y **reentrenar periódicamente** para adaptarse al mercado.
- **Desplegar la app** en Streamlit Community Cloud.

---

## Autor

**[Tu nombre]** · Bootcamp de Data Science · 2026
[GitHub](https://github.com/tu-usuario) · [LinkedIn](https://linkedin.com/in/tu-usuario)
