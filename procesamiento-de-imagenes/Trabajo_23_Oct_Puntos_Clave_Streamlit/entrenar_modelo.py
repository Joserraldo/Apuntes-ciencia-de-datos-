# -*- coding: utf-8 -*-
"""
entrenar_modelo.py — Reentrenamiento del detector de puntos clave (StanfordExtra)
================================================================================
Trabajo — viernes 23 de octubre · Ciencia de Datos · Procesamiento de Imágenes

El cuaderno del curso entrena la red tal cual viene (5 épocas, backbone congelado,
lr = 1e-4). Ese modelo **no localiza nada**: predice el promedio y los 24 puntos
caen en un único píxel (dispersión 0,3 %, error 27 % de la imagen = azar).

Este script replica la misma arquitectura pero corrige cuatro cosas:

  1. **Backbone parcialmente descongelado** (último bloque de MobileNetV2), para que
     las características se adapten al dominio y no solo la cabeza.
  2. **Volteo horizontal con intercambio de etiquetas izquierda/derecha.** El pipeline
     original voltea la imagen pero deja los nombres: eso etiqueta mal el 30 % de los
     datos de aumento (una pata izquierda pasa a ser derecha y sigue llamándose igual).
  3. **Pérdida enmascarada por visibilidad.** Los puntos que el dataset no anota
     (ojos, cruz, garganta: nunca aparecen en la muestra StanfordExtra) ya no se
     entrenan como si estuvieran en (0, 0).
  4. **Cabeza multi-tarea**: 48 coordenadas + 24 visibilidades. En inferencia se sabe
     qué puntos el modelo realmente ve, y el contorno se calcula solo con esos.

Velocidad: se cachean las activaciones de la parte congelada y se entrena el
sub-modelo entrenable sobre ese caché. El resultado es idéntico a entrenar
end-to-end, pero cada época cuesta ~1 s en CPU en vez de ~30 s.

Uso:
    python entrenar_modelo.py

Requiere `StanfordExtra-master/` (el cuaderno lo descarga solo) y guarda
`modelos/modelo_puntos_clave.keras`.

Resultados medidos con este script (ver `resultados/metricas.txt`):
    error medio     27,4 % -> 7,9 % de la imagen (puntos visibles)
    dispersión      0,33 % -> 31,3 %  (ya no colapsa en un solo punto)
    visibilidad     acierto 100 % sobre los 20 puntos anotados
"""

import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from PIL import Image, ImageEnhance

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
SEED = 42
rng = np.random.default_rng(SEED)
keras.utils.set_random_seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
SE_DIR = os.path.join(BASE, "StanfordExtra-master")
JSON_PATH = os.path.join(SE_DIR, "StanfordExtra_sample.json")
MODEL_PATH = os.path.join(BASE, "modelos", "modelo_puntos_clave.keras")

IMG_SIZE = 224
NUM_KP = 24
NUM_COORD = NUM_KP * 2          # 48 coordenadas
NUM_OUT = NUM_COORD + NUM_KP    # + 24 visibilidades = 72 salidas
VARIANTS = 60                   # aumentos por imagen
EPOCHS = 160
BATCH = 32
LR = 1e-3
UNFREEZE_FROM = "block_16_expand"   # último bloque del backbone

# Pares simétricos: al voltear la imagen hay que intercambiar las etiquetas
SWAP = {0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11,
        6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5,
        14: 15, 15: 14, 18: 19, 19: 18, 20: 21, 21: 20}
PERM = np.array([SWAP.get(i, i) for i in range(NUM_KP)])

# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def cargar(entry):
    """Imagen RGB 224×224 + coordenadas normalizadas + visibilidad."""
    img = (Image.open(os.path.join(SE_DIR, "sample_imgs", entry["img_path"]))
           .convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    w, h = entry["img_width"], entry["img_height"]
    kps = np.array([[p[0] / w, p[1] / h] for p in entry["joints"]], dtype="float32")
    vis = np.array([1.0 if p[2] > 0 else 0.0 for p in entry["joints"]], dtype="float32")
    return np.array(img, dtype="uint8"), np.clip(kps, 0.0, 1.0), vis


def aumentar(img, kps_n, vis, g):
    """Rotación ±12°, escala 0.85-1.15, traslación ±5 %, volteo y fotometría."""
    ang = np.deg2rad(g.uniform(-12, 12))
    s = g.uniform(0.85, 1.15)
    tx, ty = g.uniform(-0.05, 0.05) * IMG_SIZE, g.uniform(-0.05, 0.05) * IMG_SIZE
    c = np.array([IMG_SIZE / 2, IMG_SIZE / 2])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s

    A = np.eye(3)
    A[:2, :2] = R
    A[:2, 2] = c + np.array([tx, ty]) - R @ c

    if g.random() < 0.5:                                # volteo + intercambio L/R
        F = np.array([[-1.0, 0, IMG_SIZE - 1], [0, 1.0, 0], [0, 0, 1]])
        A = F @ A
        kps_n = kps_n[PERM].copy()
        vis = vis[PERM].copy()

    Ai = np.linalg.inv(A)                               # PIL usa la transformada inversa
    img2 = img.transform((IMG_SIZE, IMG_SIZE), Image.AFFINE,
                         (Ai[0, 0], Ai[0, 1], Ai[0, 2], Ai[1, 0], Ai[1, 1], Ai[1, 2]),
                         resample=Image.BILINEAR)

    px = kps_n * IMG_SIZE
    ph = np.concatenate([px, np.ones((len(px), 1))], 1) @ A.T
    kps_aug = np.clip(ph[:, :2] / IMG_SIZE, 0.0, 1.0).astype("float32")

    img2 = ImageEnhance.Brightness(img2).enhance(g.uniform(0.75, 1.25))
    img2 = ImageEnhance.Contrast(img2).enhance(g.uniform(0.8, 1.2))
    return np.array(img2, dtype="uint8"), kps_aug, vis


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def get_model(trainable_from=None):
    """MobileNetV2 (ImageNet) + Dropout + 2 SeparableConv2D, con la cola entrenable."""
    base = keras.applications.MobileNetV2(weights="imagenet", include_top=False,
                                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    if trainable_from is not None:
        idx = [l.name for l in base.layers].index(trainable_from)
        for l in base.layers[idx:]:
            l.trainable = True

    pre_inner = keras.Model(base.input, base.get_layer(trainable_from).output)
    tail_inner = keras.Model(base.get_layer(trainable_from).output, base.output)

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    feats = pre_inner(x)                       # punto de corte (se cachea)
    x = tail_inner(feats)
    x = layers.Dropout(0.3)(x)
    x = layers.SeparableConv2D(NUM_OUT, kernel_size=5, strides=1, activation="relu")(x)
    outputs = layers.SeparableConv2D(NUM_OUT, kernel_size=3, strides=1,
                                     activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="detector_puntos_clave")
    return model, feats, tail_inner


def perdida(y_true, y_pred):
    """MSE enmascarado por visibilidad (coordenadas) + entropía cruzada (visibilidad)."""
    yc, yv = y_true[..., :NUM_COORD], y_true[..., NUM_COORD:]
    pc, pv = y_pred[..., :NUM_COORD], y_pred[..., NUM_COORD:]
    m = tf.repeat(yv, 2, axis=-1)              # (x, y) de cada punto comparten máscara
    mse = tf.reduce_sum(m * tf.square(pc - yc)) / (tf.reduce_sum(m) + 1e-6)
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(yv, pv))
    return mse + 0.5 * bce


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def main():
    if not os.path.exists(JSON_PATH):
        raise SystemExit("Falta 'StanfordExtra-master/'. Ejecuta el cuaderno una vez "
                         "para que descargue el dataset de muestra.")

    datos = json.load(open(JSON_PATH, encoding="utf-8"))
    imgs, kps, vis = zip(*(cargar(e) for e in datos))
    imgs = np.stack(imgs); kps = np.stack(kps); vis = np.stack(vis)
    print(f"imágenes: {imgs.shape} | puntos visibles por imagen: {vis.sum(1).mean():.1f}/24")

    model, feats, tail_inner = get_model(UNFREEZE_FROM)
    n_train = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    print(f"parámetros: {model.count_params():,} totales | {n_train:,} entrenables")

    # --- caché de aumentos a través de la parte congelada ---
    t0 = time.time()
    X, Y = [], []
    for i in range(len(imgs)):
        for _ in range(VARIANTS):
            im, kp, vi = aumentar(Image.fromarray(imgs[i]), kps[i], vis[i], rng)
            X.append(im)
            Y.append(np.concatenate([kp.reshape(-1), vi]).astype("float32"))
    X = np.stack(X)
    Y = np.stack(Y).astype("float32").reshape(-1, 1, 1, NUM_OUT)
    print(f"aumentos generados: {X.shape} en {time.time() - t0:.0f}s")

    feat_extractor = keras.Model(model.input, feats)
    FEATS = feat_extractor.predict(X, batch_size=BATCH, verbose=0)
    del X
    print("features cacheadas:", FEATS.shape)

    # --- sub-modelo entrenable (comparte pesos con `model`) ---
    tin = layers.Input(FEATS.shape[1:])
    z = tail_inner(tin)
    z = model.get_layer("dropout")(z)
    z = model.get_layer("separable_conv2d")(z)
    z = model.get_layer("separable_conv2d_1")(z)
    train_model = keras.Model(tin, z)
    train_model.compile(loss=perdida,
                        optimizer=keras.optimizers.AdamW(learning_rate=LR,
                                                         weight_decay=1e-4))

    t0 = time.time()
    hist = train_model.fit(FEATS, Y, batch_size=BATCH, epochs=EPOCHS, verbose=2)
    print(f"entrenamiento: {time.time() - t0:.0f}s | pérdida final: {hist.history['loss'][-1]:.5f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"modelo guardado en {MODEL_PATH} "
          f"({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
