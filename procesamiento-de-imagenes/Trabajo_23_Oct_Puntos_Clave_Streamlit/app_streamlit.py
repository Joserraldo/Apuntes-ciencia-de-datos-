# -*- coding: utf-8 -*-
"""
Detector de Puntos Clave en Perros — App Streamlit
==================================================

Trabajo — viernes 23 de octubre de 2026
Asignatura: Ciencia de Datos · Procesamiento de Imágenes (avanzado)

Qué hace
--------
1. Carga el modelo Keras (MobileNetV2 + regresión) entrenado en el cuaderno
   `Avanzada_Cuaderno_9D_Deteccion_puntos_clave2.ipynb`.
2. Predice 24 puntos anatómicos del perro: 48 coordenadas + 24 visibilidades.
3. Dibuja los puntos **y un contorno azul** que bordea al perro: la envolvente
   suavizada de los puntos visibles.
4. Muestra las coordenadas en píxeles y normalizadas, con descarga en CSV/PNG.

Toda la lógica de inferencia y dibujo vive en `puntos_clave.py`, reutilizable
desde el cuaderno y desde los scripts de evaluación.

Ejecutar:
    streamlit run app_streamlit.py
"""

import io
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from puntos_clave import (COLORES, IMG_SIZE, MODEL_PATH, NOMBRES, NUM_PUNTOS,
                          SUPERVISADOS, cargar_modelo, dibujar, predecir)

MUESTRAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muestras")

# --------------------------------------------------------------------------- #
# Modelo (caché de Streamlit)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def modelo_cacheado():
    """Carga el modelo .keras una sola vez por sesión."""
    return cargar_modelo(MODEL_PATH)


# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Detector de puntos clave en perros",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .hero {
        background: linear-gradient(120deg, rgba(56,189,248,.16), rgba(167,139,250,.16) 60%, rgba(56,189,248,.10));
        border: 1px solid rgba(148,163,184,.28);
        border-radius: 18px;
        padding: 22px 26px 18px 26px;
        margin-bottom: 14px;
      }
      .hero h1 { font-size: 1.85rem; margin: 0 0 6px 0; letter-spacing: -.02em; }
      .hero p { margin: 0; opacity: .85; font-size: .95rem; line-height: 1.55; }
      .chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
      .chip {
        font-size: .8rem; padding: 5px 12px; border-radius: 999px;
        border: 1px solid rgba(148,163,184,.38); background: rgba(148,163,184,.12);
        white-space: nowrap;
      }
      .leyenda { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px 18px; margin-top: 8px; }
      .leyenda div { font-size: .85rem; opacity: .92; display: flex; align-items: center; gap: 8px; }
      .punto { width: 11px; height: 11px; border-radius: 50%; flex: 0 0 auto; box-shadow: 0 0 0 2px rgba(255,255,255,.18); }
      .punto.hueco { box-shadow: inset 0 0 0 2px rgba(203,213,225,.9); }
      div[data-testid="stMetricValue"] { font-size: 1.35rem; }
      div[data-testid="stMetricLabel"] p { opacity: .75; font-size: .82rem; }
      .pie { font-size: .86rem; opacity: .8; line-height: 1.5; }
      table.ficha { width: 100%; border-collapse: collapse; font-size: .9rem; }
      table.ficha td { padding: 6px 10px; border-bottom: 1px solid rgba(148,163,184,.18); }
      table.ficha td:first-child { color: #7dd3fc; white-space: nowrap; width: 130px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🐕 Detector de puntos clave en perros</h1>
      <p>Aprendizaje por transferencia con <b>MobileNetV2</b> + una cabeza de regresión que localiza
      <b>24 puntos anatómicos</b> sobre la imagen de un perro. El contorno azul es la envolvente
      suavizada de los puntos visibles: la silueta que bordea al animal.</p>
      <div class="chips">
        <span class="chip">📅 Trabajo — viernes 23 de octubre</span>
        <span class="chip">📚 StanfordExtra · 24 keypoints</span>
        <span class="chip">🧠 MobileNetV2 + 2 SeparableConv2D</span>
        <span class="chip">🎯 72 salidas · 48 coordenadas + 24 visibilidades</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("📥 Imagen de entrada")
    fuente = st.radio("Fuente", ["Usar imagen de muestra", "Subir imagen"],
                      label_visibility="collapsed")

    imagen = None
    nombre_imagen = ""
    if fuente == "Subir imagen":
        up = st.file_uploader("Sube una foto de un perro (JPG/PNG)",
                              type=["jpg", "jpeg", "png"])
        if up is not None:
            imagen = Image.open(io.BytesIO(up.read())).convert("RGB")
            nombre_imagen = up.name
    else:
        muestras = []
        if os.path.isdir(MUESTRAS_DIR):
            muestras = sorted(f for f in os.listdir(MUESTRAS_DIR)
                              if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if muestras:
            sel = st.selectbox("Imagen de muestra", muestras, index=0)
            imagen = Image.open(os.path.join(MUESTRAS_DIR, sel)).convert("RGB")
            nombre_imagen = sel
        else:
            st.warning("No hay imágenes en `muestras/`.")

    st.markdown("---")
    st.subheader("🎨 Visualización")
    ver_contorno = st.toggle("Contorno del perro", value=True)
    ver_numeros = st.toggle("Numerar los puntos", value=False)
    grosor = st.slider("Grosor del contorno", 1, 6, 3)
    tam = st.slider("Tamaño de los puntos", 0.5, 2.0, 1.0, step=0.1)

    st.markdown("---")
    st.caption(
        "**Modelo:** MobileNetV2 (ImageNet) + regresión, entrenado con la muestra "
        "pública de StanfordExtra (25 imágenes)."
    )
    with st.expander("ℹ️ Sobre la precisión"):
        st.markdown(
            "Es una **demostración del pipeline** completo: transferencia de aprendizaje → "
            "regresión de coordenadas → despliegue. Con 25 imágenes de muestra los puntos son "
            "aproximados; entrenar con el dataset completo (12.000 imágenes) o con más épocas "
            "mejora el resultado."
        )

# --------------------------------------------------------------------------- #
# Cuerpo
# --------------------------------------------------------------------------- #
if imagen is None:
    st.info("👈 Elige una imagen de muestra o sube una foto para empezar.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"No se encontró el modelo en `{MODEL_PATH}`.")
    st.stop()

try:
    modelo = modelo_cacheado()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo cargar el modelo: `{e}`")
    st.stop()

with st.spinner("Localizando los puntos anatómicos…"):
    kps, vis = predecir(modelo, imagen)
    anotada = dibujar(imagen, kps, vis=vis, contorno=ver_contorno,
                      etiquetas=ver_numeros, grosor=grosor, tam_punto=tam)

w_img, h_img = imagen.size

col_a, col_b = st.columns(2, gap="large")
with col_a:
    with st.container(border=True):
        st.markdown("**🖼️ Imagen original**")
        st.image(imagen, width="stretch")
        st.caption(f"`{nombre_imagen}` · {w_img} × {h_img} px")
with col_b:
    with st.container(border=True):
        st.markdown("**🎯 Predicción · 24 puntos clave**")
        st.image(anotada, width="stretch")
        st.caption("Contorno = envolvente suavizada de los puntos visibles. "
                   "Los anillos tenues son puntos que el modelo no localiza.")

# ------------------------------- métricas ---------------------------------- #
visibles = np.array([vis[i] >= 0.5 for i in SUPERVISADOS])
coords_vis = kps[SUPERVISADOS][visibles]
dists = (np.linalg.norm(coords_vis - coords_vis.mean(axis=0), axis=1) * 100
         if len(coords_vis) else np.array([0.0]))

with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Puntos localizados", f"{int(visibles.sum())} de {len(SUPERVISADOS)}",
              help="De los 20 puntos que StanfordExtra anota; los otros 4 no tienen etiqueta en el dataset.")
    m2.metric("Coordenadas predichas", f"{NUM_PUNTOS * 2}", help="24 puntos × (x, y)")
    m3.metric("Resolución de análisis", f"{IMG_SIZE}×{IMG_SIZE}", help="Entrada de MobileNetV2")
    m4.metric("Dispersión del esqueleto", f"{dists.mean():.1f} %",
              help="Qué tanto se separan los puntos entre sí. Cerca de 0 % = todos en el mismo píxel.")

tabla = pd.DataFrame({
    "Punto clave": NOMBRES,
    "visible": ["✅" if (i in SUPERVISADOS and vis[i] >= 0.5) else "—" for i in range(NUM_PUNTOS)],
    "confianza": np.round(vis, 2),
    "x (px)": np.round(kps[:, 0] * w_img, 1),
    "y (px)": np.round(kps[:, 1] * h_img, 1),
    "x (norm)": np.round(kps[:, 0], 4),
    "y (norm)": np.round(kps[:, 1], 4),
})

# -------------------------------- pestañas --------------------------------- #
tab_tabla, tab_como, tab_leyenda, tab_desc = st.tabs(
    ["📋 Coordenadas", "⚙️ Cómo funciona", "🎨 Leyenda de puntos", "⬇️ Descargas"]
)

with tab_tabla:
    st.markdown("#### Coordenadas de los 24 puntos predichos")
    st.caption(
        "`x`/`y` en píxeles sobre la imagen original y normalizadas en [0, 1]. "
        "`visible` = el modelo sí localiza ese punto en esta foto; `confianza` es su probabilidad."
    )
    st.dataframe(tabla, width="stretch", hide_index=True)

with tab_como:
    izq, der = st.columns([3, 2], gap="large")
    with izq:
        st.markdown("#### Del píxel al punto clave")
        st.markdown(
            """
**1. Preprocesado.** La imagen se redimensiona a `224×224` (igual que en el
entrenamiento) y se mantiene en `uint8`: el modelo aplica por dentro el
`preprocess_input` propio de MobileNetV2.

**2. Extracción de características.** El backbone `MobileNetV2` (preentrenado en
ImageNet) convierte la imagen en un mapa de características de `7×7×1280`. No se
entrena desde cero: se reutiliza lo que la red ya sabe de las imágenes.

**3. Regresión de coordenadas + visibilidad.** Dos `SeparableConv2D` reducen
`7×7 → 3×3 → 1×1` y devuelven **72 números**: 48 coordenadas (24 puntos × x/y) y
**24 visibilidades**. La activación `sigmoid` garantiza salidas en `[0, 1]`.

La visibilidad es clave en este dataset: hay puntos que la muestra de StanfordExtra
**nunca anota** (ojos, cruz y garganta, que solo existen en el subconjunto
Animal-Pose) y otros que simplemente quedan ocultos en una foto concreta. Sin esa
señal, la red los empujaba a la esquina `(0, 0)` y el contorno salía torcido.

**4. Despliegue.** Multiplicando por el ancho y el alto originales se recuperan las
coordenadas en píxeles; con ellas se dibujan los puntos y la envolvente suavizada
(convex hull + suavizado de Chaikin) que forma el contorno.

> El **borde** es una estimación geométrica a partir de los puntos, no una máscara
> de segmentación: el modelo solo predice keypoints y el contorno se deriva de ellos
> para visualizar la silueta.
            """
        )
    with der:
        st.markdown("#### Ficha técnica")
        st.markdown(
            """
<table class="ficha">
<tr><td>Entrada</td><td>224 × 224 × 3 · RGB · uint8</td></tr>
<tr><td>Backbone</td><td>MobileNetV2 · ImageNet</td></tr>
<tr><td>Cabeza</td><td>Dropout 0.3 → SeparableConv2D 72 (5×5, ReLU) → SeparableConv2D 72 (3×3, Sigmoid)</td></tr>
<tr><td>Salida</td><td>1 × 1 × 72 → 48 coordenadas + 24 visibilidades</td></tr>
<tr><td>Pérdida</td><td>MSE enmascarado + entropía cruzada binaria</td></tr>
<tr><td>Datos</td><td>StanfordExtra · muestra de 25 imágenes</td></tr>
<tr><td>Formato</td><td>Keras 3 (.keras)</td></tr>
</table>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.metric("Parámetros totales", "2.354.256", help="Backbone congelado + cabeza de regresión")

with tab_leyenda:
    st.markdown("#### Los 24 puntos clave (StanfordExtra)")
    st.caption("El color identifica el grupo anatómico; es el mismo del dataset original. "
               "Los 4 últimos no vienen anotados en la muestra del dataset.")
    filas = []
    for i in range(NUM_PUNTOS):
        clase = "punto" if i in SUPERVISADOS else "punto hueco"
        estilo = f"background:{COLORES[i]}" if i in SUPERVISADOS else "background:transparent"
        filas.append(
            f'<div><span class="{clase}" style="{estilo}"></span>'
            f"<span>{i + 1:02d} · {NOMBRES[i]}</span></div>"
        )
    st.markdown(f'<div class="leyenda">{"".join(filas)}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "- 🔴 **Rojos** · pata delantera izquierda  \n"
        "- 🟡 **Amarillos** · pata trasera izquierda  \n"
        "- 🟢 **Verdes** · pata delantera derecha  \n"
        "- 🔵 **Azules** · pata trasera derecha  \n"
        "- 🟣 **Morados** · cola  \n"
        "- 🩷 **Rosa / azul** · orejas  \n"
        "- 🟠 **Naranjas** · nariz y barbilla"
    )

with tab_desc:
    izq, der = st.columns(2, gap="large")
    buf = io.BytesIO()
    anotada.save(buf, format="PNG")
    with izq:
        st.markdown("#### Imagen anotada")
        st.caption("PNG a resolución original, con puntos, contorno y numeración opcional.")
        st.download_button("⬇️ Descargar imagen anotada (PNG)", buf.getvalue(),
                           file_name="prediccion_puntos_clave.png", mime="image/png",
                           width="stretch")
    with der:
        st.markdown("#### Coordenadas")
        st.caption("CSV con los 24 puntos: píxeles, normalizado y confianza de visibilidad.")
        st.download_button("⬇️ Descargar coordenadas (CSV)",
                           tabla.to_csv(index=False).encode("utf-8"),
                           file_name="coordenadas_puntos_clave.csv", mime="text/csv",
                           width="stretch")

st.markdown("---")
st.markdown(
    '<div class="pie">📌 <b>Proyecto académico</b> — Ciencia de Datos · Procesamiento de Imágenes. '
    "Dataset: <a href='https://github.com/benjiebob/StanfordExtra'>StanfordExtra</a> (Biggs et al., ECCV 2020). "
    "Modelo entrenado y evaluado en el cuaderno "
    "<code>Avanzada_Cuaderno_9D_Deteccion_puntos_clave2.ipynb</code>.</div>",
    unsafe_allow_html=True,
)
