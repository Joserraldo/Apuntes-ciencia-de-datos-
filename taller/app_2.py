
import math
from pathlib import Path

import joblib
import streamlit as st


def forward(X1, X2):
    """Compute a forward pass of the network."""
    X1X2 = X1 * X2
    a1 = max(0, 1.4 + (0.48 * X1) + (-0.36 * X2) + (2.2 * X1X2))
    a2 = max(0, 0.49 + (-0.16 * X1) + (-0.0072 * X2) + (-0.43 * X1X2))
    a3 = max(0, 0.38 + (-1.4 * X1) + (-0.49 * X2) + (0.41 * X1X2))
    a4 = max(0, -0.89 + (-0.83 * X1) + (1.5 * X2) + (-0.48 * X1X2))
    a5 = max(0, -0.57 + (-2.4 * X1) + (-1.8 * X2) + (-0.55 * X1X2))
    a6 = max(0, -0.037 + (1.1 * X1) + (-0.84 * X2) + (0.67 * X1X2))
    a7 = max(0, 0.021 + (0.30 * X1) + (-0.40 * X2) + (-1.0 * X1X2))
    a8 = max(0, -0.038 + (0.13 * X1) + (-0.078 * X2) + (-0.36 * X1X2))
    a9 = max(0, 2.3 + (-0.35 * a1) + (0.47 * a2) + (0.93 * a3) + (-1.0 * a4) + (-1.8 * a5) + (-0.84 * a6) + (0.94 * a7) + (0.23 * a8))
    a10 = max(0, 0.66 + (-1.5 * a1) + (-0.39 * a2) + (0.19 * a3) + (-0.72 * a4) + (-1.5 * a5) + (0.24 * a6) + (0.87 * a7) + (0.20 * a8))
    a11 = max(0, -0.43 + (-0.33 * a1) + (0.33 * a2) + (0.14 * a3) + (-0.57 * a4) + (-0.28 * a5) + (0.58 * a6) + (1.1 * a7) + (0.25 * a8))
    a12 = max(0, 0.49 + (-0.00081 * a1) + (0.34 * a2) + (0.75 * a3) + (-0.56 * a4) + (1.4 * a5) + (-1.0 * a6) + (1.2 * a7) + (0.16 * a8))
    a13 = max(0, 0.58 + (-2.0 * a1) + (0.80 * a2) + (-0.078 * a3) + (0.48 * a4) + (-0.15 * a5) + (0.39 * a6) + (0.81 * a7) + (-0.10 * a8))
    a14 = max(0, -0.24 + (-0.63 * a9) + (-0.19 * a10) + (0.13 * a11) + (-0.77 * a12) + (0.62 * a13))
    a15 = max(0, -0.56 + (2.3 * a9) + (-0.43 * a10) + (1.2 * a11) + (0.43 * a12) + (-1.4 * a13))
    a16 = max(0, 1.3 + (1.0 * a9) + (1.6 * a10) + (-0.46 * a11) + (-1.5 * a12) + (0.95 * a13))
    a17 = max(0, 0.084 + (-0.39 * a9) + (0.15 * a10) + (-0.46 * a11) + (-0.50 * a12) + (0.30 * a13))
    a18 = max(0, -0.96 + (0.73 * a14) + (1.1 * a15) + (-2.0 * a16) + (-0.12 * a17))
    a19 = max(0, 0.082 + (0.48 * a14) + (-2.3 * a15) + (0.79 * a16) + (-0.43 * a17))
    return math.tanh(0.13 + (-1.4 * a18) + (2.2 * a19))


# ---------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------
st.set_page_config(page_title="Predicción de problemas cardiacos", page_icon="❤️", layout="centered")

IMG_CORAZON = "https://studikard.com/wp-content/uploads/2023/01/corazon-2.jpg"
IMG_ATAQUE = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRLyw7qdqyeeiNh28ej5_DfoLjyp4ML7r6R7fbQufmksw&s=10"
IMG_FELIZ = "https://img.magnific.com/vector-premium/dibujos-animados-vector-lindo-personaje-corazon-fuerte-diseno_52569-1817.jpg"
IMG_LOGO = "https://unab.edu.co/wp-content/uploads/2022/09/cropped-favicon-naranja.png"

st.title("❤️ Predicción de problemas cardiacos")
st.image(IMG_LOGO, width=72)
image_columns = st.columns(3)
for column, image, caption in zip(
    image_columns,
    (IMG_CORAZON, IMG_ATAQUE, IMG_FELIZ),
    ("Corazón", "Señales de alerta", "Corazón saludable"),
):
    column.image(image, caption=caption, width="stretch")

st.header("🎯 Objetivo")
st.write(
    "Esta aplicación demuestra un modelo experimental de ciencia de datos que recibe la edad y el "
    "nivel de colesterol de una persona, normaliza esos datos y los procesa con una red neuronal. "
    "El resultado orienta sobre una posible clasificación de riesgo; no reemplaza una valoración médica."
)

st.header("📋 Instrucciones")
st.write(
    """
     1. Ingrese una edad entre **20 y 100 años**.
     2. Ingrese el colesterol total entre **200 y 500 mg/dL**.
     3. Presione **Predecir**. La aplicación aplicará el modelo de estandarización guardado y multiplicará
         los valores normalizados por 2 antes de ejecutar la red neuronal.
     4. Revise la clasificación, el porcentaje estimado y las recomendaciones preventivas.
    """
)

# Escalador entrenado en procesamiento.py (mismo preprocesamiento del entrenamiento)
@st.cache_resource
def load_scaler():
    return joblib.load(Path(__file__).with_name("modelo_estandarizacion.joblib"))


scaler = load_scaler()

st.sidebar.header("🩺 Datos de entrada")
edad = st.sidebar.slider("Edad (años)", min_value=20, max_value=100, value=45, step=1)
colesterol = st.sidebar.slider("Colesterol (mg/dL)", min_value=200, max_value=500, value=250, step=1)

predecir = st.sidebar.button("Predecir", type="primary")

if predecir:
    # Mismo preprocesamiento que en el entrenamiento: estandarizar y multiplicar por 2
    X_scaled = scaler.transform([[edad, colesterol]]) * 2
    x1, x2 = X_scaled[0][0], X_scaled[0][1]

    salida = forward(x1, x2)
    clase = 1 if salida >= 0 else -1
    prob_prediccion = ((salida + 1) / 2) * 100 if clase == 1 else ((1 - salida) / 2) * 100

    st.header("🔎 Resultado de la Predicción")

    if clase == 1:
        st.image(IMG_ATAQUE, caption="Posible riesgo cardíaco", width="stretch")
        st.error(f"⚠️ Podría sufrir problemas del corazón (Clase: {clase})")
        st.error(f"Porcentaje estimado de la predicción: **{prob_prediccion:.2f}%**")

        st.subheader("💡 Recomendaciones preventivas")
        st.markdown(
            """
            - Reducir el consumo de grasas saturadas y colesterol dietético.
            - Realizar actividad física regular (mínimo 150 min/semana).
            - Controlar el peso corporal y evitar el sobrepeso.
            - Evitar el consumo de tabaco y limitar el alcohol.
            - Consultar a un profesional de la salud para una evaluación personalizada.
            - Mantener una dieta rica en fibra, frutas, verduras y pescado.
            """
        )
    else:
        st.image(IMG_FELIZ, caption="Corazón saludable", width="stretch")
        st.success(f"✅ No sufrirá problemas del corazón (Clase: {clase})")
        st.success(f"Porcentaje estimado de la predicción: **{prob_prediccion:.2f}%**")

        st.subheader("💡 Recomendaciones para mantener la salud cardíaca")
        st.markdown(
            """
            - Mantener una alimentación balanceada baja en grasas saturadas.
            - Continuar con actividad física regular.
            - Realizar chequeos médicos periódicos de colesterol y presión arterial, especialmente si tiene más de 40 años.
            - Evitar el sedentarismo y el consumo de tabaco.
            """
        )

st.divider()
st.caption("Ciencia de Datos UNAB 2026")
st.caption("José Alejandro Téllez Prada - UNAB | Trabajo experimental ™")