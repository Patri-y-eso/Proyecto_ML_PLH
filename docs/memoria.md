# Memoria del proyecto · Steam Hit Predictor

**Predicción del éxito de videojuegos de Steam antes de su lanzamiento**

Proyecto final de Machine Learning · Bootcamp de Data Science · 2026

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Contexto y objetivos de negocio](#2-contexto-y-objetivos-de-negocio)
3. [Adquisición de datos](#3-adquisición-de-datos)
4. [Limpieza y definición del target](#4-limpieza-y-definición-del-target)
5. [Feature engineering](#5-feature-engineering)
6. [Análisis exploratorio (EDA)](#6-análisis-exploratorio-eda)
7. [Modelado](#7-modelado)
8. [Evaluación](#8-evaluación)
9. [Interpretación del modelo](#9-interpretación-del-modelo)
10. [Análisis de robustez](#10-análisis-de-robustez)
11. [Predicción de los próximos lanzamientos](#11-predicción-de-los-próximos-lanzamientos)
12. [Puesta en producción: scripts y app](#12-puesta-en-producción-scripts-y-app)
13. [Valor de negocio, viabilidad y ROI](#13-valor-de-negocio-viabilidad-y-roi)
14. [Limitaciones y consideraciones éticas](#14-limitaciones-y-consideraciones-éticas)
15. [Próximos pasos](#15-próximos-pasos)
16. [Conclusiones](#16-conclusiones)

---

## 1. Resumen ejecutivo

Se ha desarrollado un modelo de clasificación que estima si un videojuego de Steam tendrá éxito **antes de su lanzamiento**, usando únicamente información pública disponible en ese momento.

- **Datos:** 150.863 juegos de Steam (Kaggle). Tras la limpieza quedan 78.593 para entrenar, con 107 variables.
- **Definición de éxito:** ≥ 500 reseñas y ≥ 80 % positivas. Lo cumple el 7,16 % de los juegos.
- **Modelo final:** XGBoost, elegido entre 6 modelos supervisados con `Pipeline` y `GridSearchCV`.
- **Rendimiento:** PR-AUC 0,47 (el azar da 0,07) y ROC-AUC 0,91.
- **En términos de negocio:** el 10 % de juegos mejor puntuados tiene un 42 % de éxitos (5,9 veces el azar), y el 20 % mejor puntuado concentra el 81 % de todos los éxitos.
- **Aplicación:** se han evaluado 36 juegos que salen entre noviembre y diciembre de 2026. El modelo identifica 9 favoritos y espera unos 6,4 éxitos entre los 36.
- **Entrega:** notebooks documentados, scripts reproducibles y una app de Streamlit para usuarios no técnicos.

---

## 2. Contexto y objetivos de negocio

### 2.1 El problema

Steam, la mayor tienda de videojuegos de PC, publica cada año más juegos: 2.513 en 2015, 24.688 en 2025 y 28.835 en 2026 (unos 80 al día). Desde 2017 (Steam Direct) cualquier desarrollador puede publicar pagando una pequeña cuota, y la competencia se ha disparado. El resultado es que **solo uno de cada catorce juegos** alcanza un éxito comercial y de crítica claro, y la tasa de éxito no deja de bajar.

### 2.2 ¿A quién le sirve el modelo?

| Usuario | Decisión que apoya |
|---|---|
| **Publishers e inversores** | En qué proyectos invertir o qué estudios fichar entre cientos de propuestas |
| **Plataformas y medios especializados** | Qué próximos lanzamientos destacar o cubrir |
| **Estudios independientes** | Qué aspectos de su juego y de su ficha en la tienda se asocian al éxito |

### 2.3 Objetivos

1. **Priorizar:** ordenar los próximos lanzamientos por potencial para concentrar la atención (y el dinero) donde hay más probabilidad de éxito.
2. **Reducir riesgo:** descartar con seguridad los proyectos con muy pocas opciones.
3. **Explicar:** entender qué factores se asocian al éxito, no solo predecirlo.

---

## 3. Adquisición de datos

*Notebook: `01_Fuentes.ipynb`*

Se combinaron tres fuentes:

| Fuente | Registros | Papel |
|---|---|---|
| **Kaggle – Steam Games Dataset (daily updates)** | 150.863 juegos × 42 columnas | Histórico para aprender |
| **API Steam Upcoming Games (cleanedweb)** | 36 próximos lanzamientos | Juegos a predecir |
| **API pública de la tienda de Steam (`appdetails`)** | 36 fichas completas | Completar los datos de los próximos |

**Problema resuelto:** la API de próximos lanzamientos solo aporta título, fecha, plataformas y tags. No trae desarrollador, géneros, idiomas ni capturas, precisamente las variables con las que aprende el modelo. Se usó el identificador de cada juego (`appid`) para descargar su ficha completa de la tienda de Steam:

- Precios en dólares (`cc=us`) y textos en inglés (`l=english`), para que fueran comparables con Kaggle.
- Pausas de 1,6 s entre peticiones y reintentos automáticos ante el error 429 (límite de peticiones).
- Una caché en disco para poder reanudar la descarga si se interrumpía.

Las 36 fichas se descargaron correctamente y todas eran de tipo *game*. Ninguno de los 36 juegos aparecía en el histórico.

**Decisión fundamental:** las columnas del histórico se clasificaron en tres grupos:

- **Post-lanzamiento:** reseñas, propietarios, horas de juego, Metacritic, DLCs. Solo sirven para construir el target y **nunca se usan como variables**, porque un juego no lanzado no las tiene. Usarlas sería una fuga de información.
- **Sin valor predictivo:** enlaces, emails de soporte y similares. Se descartan.
- **Candidatas:** información conocida antes del lanzamiento y presente en ambas fuentes.

---

## 4. Limpieza y definición del target

*Notebook: `02_LimpiezaEDA.ipynb` · Script: `src/data_processing.py`*

### 4.1 Limpieza

| Paso | Efecto |
|---|---|
| Eliminar juegos `unavailable` (retirados de la tienda) | −1.429 |
| Eliminar duplicados y juegos sin fecha o nombre válidos | −33 |
| **Total tras la limpieza básica** | **149.401** |
| Ventana de entrenamiento: lanzados entre 2015 y noviembre de 2025 | Solo juegos con ≥ 12 meses a la venta |
| Solo juegos con al menos un tag | Ver justificación abajo |
| **Juegos para entrenar** | **78.593** |

- **¿Por qué desde 2015?** Antes, Steam era una tienda seleccionada a mano por Valve (y después filtrada por Greenlight). La tasa de éxito rondaba el 30-42 % y el mercado no se parecía al actual.
- **¿Por qué excluir los juegos sin tags?** Los tags los votan los jugadores, así que un juego sin tags es un juego que casi nadie jugó: el **0,0 %** de ellos es exitoso. Mantenerlos enseñaría al modelo un atajo que no existe en los próximos lanzamientos, que siempre tienen tags.

### 4.2 Definición de éxito

> **exitoso = 1** si el juego tiene **≥ 500 reseñas** y **≥ 80 % de reseñas positivas**

- **Popularidad:** según la regla de Boxleiter, cada reseña equivale a unas 30-60 ventas, así que 500 reseñas son unas 15.000-30.000 copias.
- **Calidad:** el 80 % coincide con la etiqueta "Muy positivas" de Steam.

Se compararon 15 combinaciones de umbrales. Esta da un 4,9 % de éxitos en la ventana temporal (7,16 % tras excluir los juegos sin tags): un éxito real y con suficientes ejemplos para aprender. Umbrales más bajos (100 reseñas) incluyen juegos demasiado pequeños; más altos (5.000 reseñas) dejan solo un 1,4 % de positivos.

---

## 5. Feature engineering

Se construyeron **107 variables** con la misma función para el histórico y para los próximos lanzamientos, lo que garantiza que significan lo mismo en ambos casos.

| Grupo | Variables | Qué capturan |
|---|---|---|
| Precio | `price`, `is_free` | Ambición del proyecto y modelo de negocio |
| Plataformas | `windows`, `mac`, `linux`, `n_platforms` | Alcance técnico |
| Localización | `n_languages`, `has_spanish`, `has_chinese` | Inversión en traducción y acceso a mercados |
| Ficha de la tienda | `n_screenshots`, `n_movies`, longitud de las descripciones, `achievements` | Esfuerzo y cuidado en la presentación |
| Calendario | `release_month`, `release_weekday` | Estacionalidad |
| Contenido | One-hot de 20 géneros, 25 categorías y 40 tags | Tipo de juego |
| Estudio | `self_published`, `dev_prev_games`, `dev_prev_hits`, `pub_prev_games` | Experiencia y trayectoria |

**Decisiones relevantes:**

- **Unificación de formatos.** El histórico guarda géneros y categorías como texto JSON, y la tienda los devuelve como listas o HTML. Se escribieron funciones de *parseo* comunes para ambas fuentes.
- **Experiencia del estudio calculada solo con el pasado.** Para cada juego se cuentan los juegos (y los éxitos) que su desarrollador había publicado **antes** de esa fecha. Es como un currículum escrito el día del lanzamiento.
- **Vocabulario del one-hot calculado solo con los datos de entrenamiento.** Los próximos lanzamientos se adaptan a ese vocabulario.
- **Tags limitados a los 5 más votados por juego**, para igualar lo que ofrece la API de próximos lanzamientos.
- **Excluidos:** la categoría *Steam Trading Cards*, que Valve añade después del lanzamiento a juegos que ya venden, y el año de lanzamiento, porque los próximos son de 2026 y el modelo tendría que extrapolar.
- **Logros desconocidos.** En los próximos lanzamientos, un 0 en logros significa "aún no publicados" (29 de 36 casos). Se trata como valor desconocido y se imputa la mediana.

---

## 6. Análisis exploratorio (EDA)

| Hallazgo | Dato |
|---|---|
| **Clases desbalanceadas** | 93 % no exitosos frente a 7 % exitosos |
| **La tasa de éxito cae con el tiempo** | ~40 % en 2012-2013 → ~3-4 % en 2023-2024 |
| **Idiomas: la relación más clara** | 3 % de éxito con 1 idioma → 23 % con 11-20 idiomas |
| **Cuidado de la ficha** | 1 % de éxito con 0-3 capturas → 13 % con 16-25 |
| **Precio** | Mediana de ~15 $ en los exitosos frente a ~5 $ en los no exitosos |
| **Géneros** | Free to Play (~11 %), RPG (~10 %) y Simulation (~9 %) arriba; Education (~2 %), Violent y Gore (~1,3 %) abajo |
| **Tags** | Multiplayer, Story Rich y Anime (18-20 %) frente a Arcade, 2D Platformer y Hidden Object (~2,5 %) |
| **Experiencia del estudio** | Relación en "U invertida": 6 % en debutantes, ~10 % con 1-10 juegos previos, 6 % con más de 10 |

**Lecturas principales:**

- **Ninguna variable tiene una correlación mayor de 0,26 con el éxito** (Spearman). No hay una variable mágica y el modelo debe combinar muchas señales. También es un buen indicio de que no hay fugas evidentes.
- **Se detectó *data drift*.** Los 36 próximos lanzamientos son producciones más grandes que la media del histórico: mediana de 9 idiomas frente a 1, más capturas, más vídeos y descripciones más largas.
- **El precio es una correlación, no una causa.** Refleja la ambición y el presupuesto del proyecto. Subir el precio no hace mejor a un juego.

---

## 7. Modelado

*Notebook: `03_Entrenamiento_Evaluacion.ipynb` · Script: `src/training.py`*

### 7.1 Diseño

- **Split 80/20 estratificado:** 62.874 juegos para entrenar y 15.719 para test, ambos con un 7,16 % de éxitos. El test no se usó para ninguna decisión.
- **Pipeline:** imputación por mediana, `log1p` en las variables asimétricas, `StandardScaler` y clasificador. Al ir todo dentro del Pipeline no hay fugas entre folds y el modelo guardado acepta los datos sin preprocesar.
- **GridSearchCV:** validación cruzada estratificada de 3 folds, optimizando la PR-AUC.
- **Desbalance:** `class_weight="balanced"` (o `scale_pos_weight` en XGBoost), en lugar de sobremuestreo sintético (SMOTE), que con tantas variables binarias genera ejemplos poco realistas.

### 7.2 Modelos probados

| Modelo | Mejores hiperparámetros | CV PR-AUC | Tiempo |
|---|---|---|---|
| Regresión logística | C = 0,1 | 0,437 | 14 s |
| KNN | k = 150, peso por distancia | 0,370 | 57 s |
| Árbol de decisión | profundidad 8, mín. 100 por hoja | 0,348 | 15 s |
| Random Forest | sin límite de profundidad, mín. 5 por hoja | 0,449 | 126 s |
| **XGBoost** | **learning rate 0,05, profundidad 4, subsample 0,8** | **0,479** | 47 s |
| LightGBM | learning rate 0,05, 31 hojas | 0,471 | 24 s |

Como referencia: un clasificador aleatorio (baseline) obtiene una PR-AUC de 0,071, y una regresión logística sin ajustar, 0,440.

**Modelo elegido: XGBoost**, por tener la mejor PR-AUC en validación cruzada. LightGBM queda prácticamente empatado.

---

## 8. Evaluación

### 8.1 Elección de la métrica

Con un 7 % de positivos, un modelo que dice siempre "no exitoso" tiene un *accuracy* del 92,8 % y no sirve para nada. Se eligió la **PR-AUC** como métrica principal porque se centra en la clase minoritaria y su referencia de azar es la tasa de positivos (0,07).

### 8.2 Resultados en test

| Métrica | Valor |
|---|---|
| **PR-AUC** | **0,470** (CV: 0,479, sin sobreajuste) |
| ROC-AUC | 0,907 |
| Precision / Recall / F1 (umbral 0,774) | 0,436 / 0,569 / 0,494 |
| Precisión en el top 1 % / top 5 % | 71 % / 55 % |

El **umbral de decisión (0,774)** se eligió maximizando el F1 con predicciones *out-of-fold* del train, sin tocar el test. Con él se obtiene esta matriz de confusión:

| | Predice "no exitoso" | Predice "exitoso" |
|---|---|---|
| **No exitoso real** | 13.765 | 828 |
| **Exitoso real** | 485 | 641 |

De 1.469 juegos marcados como prometedores, **641 triunfan (44 %, frente al 7 % al azar)**, y se detecta el **57 % de todos los éxitos** revisando solo el 9 % del catálogo.

### 8.3 Lectura por deciles

| Decil | % éxito real | Lift | % de éxitos acumulado |
|---|---|---|---|
| 1 | 42,1 % | x5,9 | 58,8 % |
| 2 | 16,0 % | x2,2 | **81,1 %** |
| 3 | 7,0 % | x1,0 | 90,9 % |
| 4 | 3,1 % | x0,4 | 95,1 % |
| 5 | 1,5 % | x0,2 | 97,2 % |
| 6-10 | < 1 % | < x0,11 | 100 % |

La escalera es perfecta: a mayor puntuación, más éxitos reales. Del decil 6 al 10 casi no hay éxitos, así que el modelo descarta con seguridad la mitad del catálogo.

---

## 9. Interpretación del modelo

Se usaron tres técnicas complementarias: importancia interna (*gain*), *permutation importance* y **SHAP**.

| Factor | Efecto | Interpretación |
|---|---|---|
| Éxitos previos del estudio | ⬆️ Muy fuerte | La señal más potente |
| Muchos juegos previos **sin** éxitos | ⬇️ | Estudios que publican mucho y con poca calidad |
| Precio medio-alto | ⬆️ | Proyectos más ambiciosos |
| Logros, tráilers, Steam Cloud | ⬆️ | Juegos pulidos y fichas cuidadas |
| Muchos idiomas, chino | ⬆️ | Alcance internacional |
| Pocas capturas | ⬇️ | Ficha descuidada |
| Free to Play, Multiplayer | ⬆️ | Gran base de jugadores y comunidad |
| Casual, Action, Arcade, Indie, Early Access | ⬇️ | Géneros saturados o juegos inacabados |

**Hallazgo destacado:** el modelo ha aprendido por sí solo que lo que importa de un estudio no es *cuántos* juegos ha publicado, sino su **tasa de acierto**. Es una interacción no lineal que la regresión logística no capta y XGBoost sí, y explica buena parte de su ventaja.

---

## 10. Análisis de robustez

Se sospechaba una **fuga leve** en `dev_prev_hits`. El éxito de los juegos anteriores de un estudio se mide con las reseñas actuales, y un éxito nuevo puede impulsar las ventas del catálogo antiguo. Para medirlo, se reentrenó el modelo anulando esas variables:

| Escenario | PR-AUC | ROC-AUC |
|---|---|---|
| Todas las variables | 0,470 | 0,907 |
| Sin `dev_prev_hits` | 0,430 | 0,891 |
| Sin ningún historial de desarrollador o publisher | 0,428 | 0,891 |

**Conclusión:** el modelo es robusto. La mayor parte de la señal viene de la ficha del juego. El rendimiento honesto se sitúa en una **PR-AUC de 0,43-0,47**. Para los juegos futuros la variable es legítima, porque usa el historial que existe hoy.

Con **estudios debutantes** (60 % del test, 5,6 % de éxitos), la PR-AUC es **0,389**, unas 7 veces su tasa base. Es decir, el modelo funciona también sin historial.

---

## 11. Predicción de los próximos lanzamientos

*Notebook: `04_Prediccion_Proximos.ipynb`*

### 11.1 Calibración

Al entrenar con `scale_pos_weight`, las puntuaciones del modelo no son probabilidades reales. Se tradujeron a **"% de éxito en juegos similares"**: a cada juego nuevo se le asigna el decil del test en el que caería y la tasa real de éxito de ese decil.

### 11.2 Resultados

- **20 juegos con potencial alto (deciles 1-2), 15 medio y 1 bajo.** Nueve superan el umbral de decisión.
- **Éxitos esperados: ≈ 6,4 de 36 (≈ 18 %),** más del doble de la tasa base de Steam.
- **Favoritos (decil 1):** *Arctic Drive, Reshine, No Asteroids, Warhammer 40,000: Dawn of War IV, Drunken Goddess Reflux, Blind Descent, Dumpster Gang, Me and You* y *Shark Mart*.
- **Prueba de sentido común:**
  - *Dawn of War IV* (gran franquicia, estudio con 2 éxitos previos) aparece en el decil 1.
  - *Winx Club – Moonlight Quest* (estudio con 10 juegos previos sin ningún éxito) cae al decil 5.
- **Explicación individual (SHAP) de *Arctic Drive*:**
  - Suman: precio de 19,99 $, Steam Cloud, logros, tráilers, chino y género Simulation.
  - Restan: Early Access y que el estudio no tenga éxitos previos.

### 11.3 ¿Por qué tantos juegos con potencial alto?

En el histórico, el 20 % de los juegos está en los deciles 1-2; entre los 36 próximos, el 56 %. No es un error del modelo, sino un **sesgo de selección de la muestra**: la API recoge los próximos lanzamientos destacados de Steam. Para comprobarlo se buscaron en el test los **712 juegos históricos con un perfil parecido** (≥ 5 idiomas, ≥ 10 capturas, ≥ 1 vídeo). El modelo coloca al 77 % de ellos en los deciles 1-2, y el **29,1 % triunfó de verdad** (cuatro veces la media).

### 11.4 Sensibilidad al precio

27 de los 36 juegos no tienen precio anunciado, y el modelo imputa la mediana del entrenamiento (~5 $), que resulta ser uno de los escenarios más pesimistas. Se simularon precios de 0 $ a 59,99 $:

- Los **6 favoritos sin precio siguen en el decil 1 a cualquier precio**.
- Con precios de 19,99 $ o más, la mayoría de los juegos sube.

El análisis mide la incertidumbre de la predicción; no es una recomendación de precio.

---

## 12. Puesta en producción: scripts y app

- **Scripts reproducibles (`src/`).** `data_processing.py`, `training.py` y `evaluation.py` reproducen el proyecto desde la terminal con **cifras idénticas a los notebooks** (semilla fija), lo que demuestra que el proceso es reproducible.
- **App de Streamlit (`app_streamlit/app.py`).** No reentrena: carga el Pipeline guardado y predice al instante. Tiene tres pestañas:
  1. **Próximos lanzamientos:** ranking y detalle de cada juego, con un slider de precio.
  2. **Simulador:** cualquier juego, real o imaginario. Por ejemplo, el mismo juego pasa del decil 5 al 1 al subir el precio, los idiomas, las capturas y añadir modo multijugador.
  3. **Sobre el modelo:** métricas y gráfico de deciles.

---

## 13. Valor de negocio, viabilidad y ROI

### 13.1 Valor

El modelo no sustituye el criterio de un experto, pero **multiplica su eficiencia**. Un equipo que revisara solo el 20 % de los proyectos mejor puntuados encontraría el 81 % de los éxitos, con una tasa de acierto casi seis veces mayor que eligiendo al azar en el decil superior.

### 13.2 Escenario ilustrativo

*Escenario hipotético, construido con las tasas del test.* Un publisher recibe 1.000 propuestas y solo puede analizar a fondo 100:

| Estrategia | Éxitos esperados entre las 100 analizadas |
|---|---|
| Elegir al azar | ~7 |
| Elegir el decil 1 del modelo | **~42** |

Con el mismo esfuerzo de análisis, el número de éxitos en la cartera se multiplica por seis.

### 13.3 Viabilidad y costes

- **Datos:** públicos y gratuitos (Kaggle, API de la tienda de Steam). La API de próximos lanzamientos tiene un nivel gratuito de 5.000 registros.
- **Cómputo:** el entrenamiento completo tarda unos 5 minutos en un portátil, y la predicción, milisegundos.
- **Despliegue:** la app puede publicarse gratis en Streamlit Community Cloud. Mantenerla actualizada solo exige programar los tres scripts periódicamente.

El coste de implantación es muy bajo frente al valor de una sola buena decisión de inversión.

---

## 14. Limitaciones y consideraciones éticas

### 14.1 Limitaciones

1. **Información incompleta antes del lanzamiento.** Steam no publica el precio ni los logros hasta cerca del lanzamiento, así que se imputan. La sensibilidad al precio está analizada.
2. **Cambio del mercado.** La tasa de éxito ha caído con los años, y los porcentajes absolutos pueden ser optimistas para 2026. El ranking relativo sigue siendo válido.
3. **Puntuaciones no calibradas.** Se resuelve traduciéndolas a deciles del test.
4. **Fuga leve en el historial del estudio.** Está acotada con el análisis de robustez (PR-AUC 0,43 sin esas variables).
5. **Posibles variables post-lanzamiento no detectadas.** Por ejemplo, *Steam Workshop*, que algunos estudios añaden tras lanzar. Conviene comprobarlo con el mismo análisis de robustez.
6. **Muestra de predicción pequeña y seleccionada:** 36 juegos destacados, no una muestra aleatoria.
7. **Variables que faltan.** Las más predictivas del prelanzamiento, como las *wishlists* o los seguidores, no son públicas.

### 14.2 Consideraciones éticas

- **Sesgo contra estudios nuevos y pequeños.** El historial del estudio es la señal más fuerte. Si la industria usara modelos así para decidir a quién financiar, los debutantes lo tendrían más difícil. El modelo funciona también con debutantes (PR-AUC 0,39), pero debe usarse como **apoyo** a la decisión, no como filtro automático.
- **Profecía autocumplida.** Si una plataforma destaca solo los juegos con mejor puntuación, estos recibirán más visibilidad y tendrán más éxito, lo que reforzaría el modelo. Conviene combinar la predicción con descubrimiento editorial.
- **Correlación no es causalidad.** El modelo no debe usarse para recomendar acciones como "sube el precio". Las variables reflejan características de los proyectos exitosos, no palancas garantizadas.
- **Transparencia.** El uso de SHAP permite explicar cada predicción, lo que facilita detectar y corregir comportamientos indeseados.

---

## 15. Próximos pasos

1. **Validación real fuera de muestra.** Los 36 juegos salen entre el 24/11 y el 15/12/2026. En 6-12 meses se podrán consultar sus reseñas y medir el acierto real del modelo.
2. **Nuevas fuentes de datos:** wishlists y seguidores (vía SteamDB o acuerdos con publishers), participación en Steam Next Fest, visualizaciones de tráilers y menciones en redes sociales.
3. **Procesamiento de lenguaje natural** sobre las descripciones (TF-IDF o embeddings) y visión por computador sobre las imágenes de cabecera.
4. **Calibración de probabilidades** (Platt o isotónica) para ofrecer probabilidades reales.
5. **Ponderación temporal o reentrenamiento periódico**, para adaptarse a un mercado cada vez más competitivo.
6. **Ampliar la predicción** a todos los próximos lanzamientos (la API permite miles de registros) y desplegar la app públicamente.

---

## 16. Conclusiones

1. **Es posible anticipar el éxito de un videojuego antes de su lanzamiento** con información pública. El modelo final (XGBoost) alcanza una PR-AUC de 0,47, unas 6,6 veces la del azar, sin sobreajuste y con un rendimiento robusto (0,43 en el escenario más conservador).
2. **El valor está en priorizar.** El 20 % de juegos mejor puntuados concentra el 81 % de los éxitos, lo que permite a un publisher, inversor o plataforma centrar sus recursos donde hay más probabilidades.
3. **El éxito tiene patrones reconocibles:** estudios con trayectoria de éxitos, proyectos ambiciosos (precio, localización, ficha cuidada, logros) y géneros menos saturados. Ninguna variable decide sola: el éxito es una combinación de factores.
4. **El rigor metodológico ha sido clave:** excluir toda información posterior al lanzamiento, medir la fuga residual, tratar correctamente el desbalance, elegir el modelo y el umbral sin tocar el test, detectar el sesgo de selección de los próximos lanzamientos y corregir los datos que faltan (logros).
5. **Entre los 36 próximos lanzamientos el modelo espera unos 6 éxitos**, con 9 favoritos claros. Su validación en los próximos meses será la prueba definitiva del proyecto.
