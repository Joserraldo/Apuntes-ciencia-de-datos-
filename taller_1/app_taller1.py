# -*- coding: utf-8 -*-
"""
App Streamlit — Taller 1: Preprocesamiento de Datos y Clasificación con Redes Neuronales
José Alejandro Téllez Prada — U00060479
Ejecutar:  streamlit run app_taller1.py
"""

import json
import ast

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

URL_CSV = "https://raw.githubusercontent.com/adiacla/bigdata/refs/heads/master/pacientes.csv"

st.set_page_config(page_title="Taller 1 — Redes Neuronales", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .title-app {
        background: linear-gradient(90deg, #7d1a1a, #a82020);
        padding: 1.2rem 1.5rem; border-radius: 12px; color: white;
        margin-bottom: 0.5rem;
    }
    .title-app h1 {margin: 0; font-size: 1.7rem;}
    .title-app p {margin: 0.2rem 0 0 0; opacity: 0.85; font-size: 0.95rem;}
    .badge {
        display: inline-block; background: #e8e8e8; border-radius: 6px;
        padding: 2px 8px; font-size: 0.8rem; margin-right: 6px; color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-app">
        <h1>🧠 Red Neuronal Artificial — Clasificación de Problemas Cardíacos</h1>
        <p>Taller 1 · Ciencia de Datos · José Alejandro Téllez Prada — U00060479</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Red neuronal (numpy) — idéntica a la del cuaderno
# ---------------------------------------------------------------------------
def init_red(arquitectura, semilla=42):
    np.random.seed(semilla)
    W, b = [], []
    for i in range(len(arquitectura) - 1):
        W.append(np.random.randn(arquitectura[i], arquitectura[i + 1]) * 0.5)
        b.append(np.zeros(arquitectura[i + 1]))
    return W, b


def propagar(X, W, b):
    a = X
    acts = [a]
    for i in range(len(W) - 1):
        a = np.maximum(0, a @ W[i] + b[i])
        acts.append(a)
    out = a @ W[-1] + b[-1]
    acts.append(out)
    return out, acts


def error_cuadratico(y, out):
    return np.mean((out - np.asarray(y).reshape(-1, 1)) ** 2)


def entrenar(X, y, arquitectura, lr, epocas=2000, semilla=42):
    y = np.asarray(y).reshape(-1, 1)
    W, b = init_red(arquitectura, semilla)
    perdidas = []
    n = len(y)
    for _ in range(epocas):
        out, acts = propagar(X, W, b)
        perdidas.append(error_cuadratico(y, out))
        grad = 2 * (out - y) / n
        dW, db = [None] * len(W), [None] * len(b)
        for i in range(len(W) - 1, -1, -1):
            dW[i] = acts[i].T @ grad
            db[i] = grad.sum(axis=0)
            if i > 0:
                grad = (grad @ W[i].T) * (acts[i] > 0)
        for i in range(len(W)):
            W[i] -= lr * dW[i]
            b[i] -= lr * db[i]
    return W, b, perdidas


def precision(y_real, out):
    pred = np.where(out >= 0, 1, -1).reshape(-1)
    return np.mean(pred == np.asarray(y_real).reshape(-1))


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv(URL_CSV, sep=",", encoding="utf-8-sig")
    df.dropna(inplace=True)
    df = df[["edad", "colesterol", "problema_cardiaco"]].copy()
    df.columns = ["x", "y", "label"]
    df["label"] = df["label"].map({1: 1, 0: -1})
    for col in ["x", "y"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std() * 2
    return df


@st.cache_data
def dividir(df):
    X = df[["x", "y"]].values
    y = df["label"].values
    return train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)


ARQUITECTURAS_PREDEF = ["[2, 2, 1]", "[2, 4, 1]", "[2, 8, 1]", "[2, 4, 4, 1]", "[2, 8, 8, 1]", "[2, 16, 8, 1]"]
LR_PREDEF = [0.003, 0.01, 0.03, 0.1, 0.3]

df = cargar_datos()
X_train, X_test, y_train, y_test = dividir(df)

tab_datos, tab_entrenar, tab_exp, tab_pesos = st.tabs(
    ["📊 Datos y JSON", "🧠 Entrenar", "🔬 Experimentos", "⚙️ Pesos y Sesgos"]
)

# ============================ TAB DATOS ============================
with tab_datos:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes", f"{len(df)}")
    c2.metric("Positivos (+1)", int((df["label"] == 1).sum()))
    c3.metric("Negativos (−1)", int((df["label"] == -1).sum()))
    c4.metric("Split 70/30", f"{len(X_train)} / {len(X_test)}")

    st.markdown("**Preprocesamiento aplicado:** renombrado `edad→x`, `colesterol→y`, `problema_cardiaco→label` · "
                "target `0→-1, 1→1` · estandarización Z-score × 2")
    st.dataframe(df.head(10), width="stretch")

    st.markdown("#### Verificación del preprocesamiento")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Medias (≈ 0):**")
        st.write(df[["x", "y"]].mean().round(4))
    with v2:
        st.markdown("**Desviaciones (= 2):**")
        st.write(df[["x", "y"]].std().round(4))

    st.markdown("#### JSON compatible con ScienxLab")
    registros = df.round(6).to_dict(orient="records")
    json_str = json.dumps(registros, ensure_ascii=False, indent=2)
    st.code(json_str[:800] + ("\n..." if len(json_str) > 800 else ""), language="json")
    st.download_button("⬇️ Descargar pacientes_preprocesado.json", json_str,
                       file_name="pacientes_preprocesado.json", mime="application/json")

# ============================ TAB ENTRENAR ============================
with tab_entrenar:
    st.markdown("**Condiciones fijas:** Classification · Squared error · ReLU · Regularización: None · 70/30")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        arq_sel = st.selectbox("Arquitectura", ARQUITECTURAS_PREDEF, index=3,
                               help="[entradas, ocultas..., salida]")
    with s2:
        lr_sel = st.select_slider("Learning rate", options=LR_PREDEF, value=0.003)
    with s3:
        epocas = st.slider("Épocas", 200, 5000, 2000, step=200)
    with s4:
        semilla = st.number_input("Semilla (seed)", 0, 999, 42)

    if st.button("🚀 Entrenar red neuronal", type="primary", width="stretch"):
        with st.spinner("Entrenando…"):
            arq = ast.literal_eval(arq_sel)
            W, b, perd = entrenar(X_train, y_train, arq, lr_sel, epocas, int(semilla))
            out_tr, _ = propagar(X_train, W, b)
            out_te, _ = propagar(X_test, W, b)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Acc train", f"{precision(y_train, out_tr):.3f}")
            m2.metric("Acc test", f"{precision(y_test, out_te):.3f}")
            m3.metric("Pérdida train", f"{perd[-1]:.4f}")
            m4.metric("Pérdida test", f"{error_cuadratico(y_test, out_te):.4f}")

            g1, g2 = st.columns(2)
            with g1:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(perd, color="#a82020", lw=1.5)
                ax.set_xlabel("Época"); ax.set_ylabel("Pérdida (squared error)")
                ax.set_title(f"Curva de pérdida — {arq_sel}, lr = {lr_sel}")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            with g2:
                x_min, x_max = df["x"].min() - 0.5, df["x"].max() + 0.5
                y_min, y_max = df["y"].min() - 0.5, df["y"].max() + 0.5
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
                out_grid, _ = propagar(np.c_[xx.ravel(), yy.ravel()], W, b)
                zz = out_grid.reshape(xx.shape)
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                ax2.contourf(xx, yy, zz, levels=[-np.inf, 0, np.inf], colors=("lightblue", "mistyrose"), alpha=0.7)
                ax2.contour(xx, yy, zz, levels=[0], colors="black", linewidths=2)
                for lab, color in [(1, "red"), (-1, "blue")]:
                    sub = df[df["label"] == lab]
                    ax2.scatter(sub["x"], sub["y"], c=color, s=18, alpha=0.8, label=f"label = {lab}")
                ax2.set_xlabel("x"); ax2.set_ylabel("y")
                ax2.set_title("Frontera de decisión")
                ax2.legend(); ax2.grid(alpha=0.3)
                st.pyplot(fig2)

            st.session_state["pesos_entrenados"] = (W, b, arq, lr_sel)
            st.success("Entrenamiento completado — pesos disponibles en la pestaña ⚙️ Pesos y Sesgos.")
    else:
        st.info("Configura los hiperparámetros y pulsa **Entrenar red neuronal**.")

# ============================ TAB EXPERIMENTOS ============================
with tab_exp:
    st.markdown("**Búsqueda exhaustiva:** todas las combinaciones de learning rate × arquitectura (30 modelos).")
    e1, e2 = st.columns(2)
    with e1:
        lrs_exp = st.multiselect("Learning rates a probar", [str(x) for x in LR_PREDEF],
                                 default=[str(x) for x in LR_PREDEF])
    with e2:
        arqs_exp = st.multiselect("Arquitecturas a probar", ARQUITECTURAS_PREDEF, default=ARQUITECTURAS_PREDEF)
    epocas_exp = st.slider("Épocas (experimentos)", 200, 2000, 1000, step=100)

    if st.button("🔬 Ejecutar experimentos", type="primary"):
        if not lrs_exp or not arqs_exp:
            st.warning("Selecciona al menos un learning rate y una arquitectura.")
        else:
            with st.spinner(f"Entrenando {len(lrs_exp) * len(arqs_exp)} modelos…"):
                filas = []
                for lr in [float(x) for x in lrs_exp]:
                    for arq in arqs_exp:
                        W, b, perd = entrenar(X_train, y_train, ast.literal_eval(arq), lr, epocas_exp)
                        out_tr, _ = propagar(X_train, W, b)
                        out_te, _ = propagar(X_test, W, b)
                        filas.append({
                            "learning_rate": lr, "arquitectura": arq,
                            "perdida_train": round(perd[-1], 5),
                            "perdida_test": round(error_cuadratico(y_test, out_te), 5),
                            "acc_train": round(precision(y_train, out_tr), 4),
                            "acc_test": round(precision(y_test, out_te), 4),
                        })
                tabla = pd.DataFrame(filas)
                tabla["ranking"] = tabla["acc_test"] * 1000 - tabla["perdida_test"]
                tabla = tabla.sort_values("ranking", ascending=False).reset_index(drop=True)

                st.markdown("#### Tabla completa (ordenada por ranking)")
                st.dataframe(tabla, width="stretch", height=320)

                mejor = tabla.iloc[0]
                st.markdown("#### 🏆 Mejor modelo")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Arquitectura", mejor["arquitectura"])
                m2.metric("Learning rate", f"{mejor['learning_rate']:.3f}")
                m3.metric("Acc test", f"{mejor['acc_test']:.4f}")
                m4.metric("Pérdida test", f"{mejor['perdida_test']:.4f}")

                st.markdown("#### Top 5")
                st.dataframe(tabla.head(5), width="stretch")
                st.download_button("⬇️ Descargar resultados CSV", tabla.to_csv(index=False),
                                   file_name="resultados_busqueda_exhaustiva.csv", mime="text/csv")
                st.session_state["mejor_grid"] = mejor

# ============================ TAB PESOS ============================
with tab_pesos:
    st.markdown("**Matriz de pesos (w) y sesgos (bias)** de la red entrenada en la pestaña Entrenar.")
    pesos = st.session_state.get("pesos_entrenados")
    if pesos is None:
        st.info("Primero entrena un modelo en la pestaña 🧠 Entrenar.")
    else:
        W, b, arq, lr = pesos
        st.caption(f"Arquitectura {arq} · learning rate {lr}")
        for i in range(len(W)):
            st.markdown(f"**Capa {i} ({W[i].shape[0]} → {W[i].shape[1]}):**")
            st.code(f"W = {np.round(W[i], 6).tolist()}\nb = {np.round(b[i], 6).tolist()}")