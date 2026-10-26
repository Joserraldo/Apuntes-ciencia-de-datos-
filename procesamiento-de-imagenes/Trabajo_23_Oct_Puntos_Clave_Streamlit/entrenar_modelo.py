# -*- coding: utf-8 -*-
"""
entrenar_modelo.py — Detector de puntos clave en perros con mapas de calor
==========================================================================
Trabajo — viernes 23 de octubre · Ciencia de Datos · Procesamiento de Imágenes

Este script es la versión "de producción" del cuaderno 9D: mismos datos
(StanfordExtra, 24 puntos) y mismo backbone (MobileNetV2), pero con la cabeza
corregida. Mide el resultado de forma honesta y guarda el modelo que usa la app.

Por qué no se usa la cabeza del cuaderno original
-------------------------------------------------
El cuaderno hace regresión directa de 48 números tras un pooling global (salida 1×1×48).
Con 25 imágenes eso no aprende a localizar: en imágenes que el modelo NO vio, el error
(≈ 20 % de la imagen) es igual al de predecir siempre la pose media. La regresión de
coordenadas necesita miles de ejemplos.

Esta versión predice **mapas de calor** (uno de 56×56 por punto) en vez de coordenadas:
mantiene la información espacial y cada punto se localiza donde la red "ve" esa parte
del cuerpo. Con las mismas 25 imágenes, el error en imágenes no vistas baja a ≈ 8 %.

Otras correcciones respecto al cuaderno
---------------------------------------
  1. Volteo horizontal con intercambio de etiquetas izquierda/derecha.
  2. Los puntos no anotados (ojos, cruz y garganta en la muestra) no se entrenan como
     si estuvieran en (0, 0): su mapa de calor es cero y no entran en el error.
  3. Si un aumento deja un punto fuera de la imagen, deja de contar como visible.
  4. BatchNorm congelado (modo inferencia) en el backbone: con lotes pequeños, los
     BatchNorm entrenables dan un desajuste entre entrenamiento e inferencia.
  5. Validación cruzada de 5 pliegues: el error que se reporta es SIEMPRE sobre
     imágenes que el modelo no vio durante ese entrenamiento.
  6. El modelo se guarda (`modelos/modelo_puntos_clave.keras`).

Uso
---
    python entrenar_modelo.py              # validación cruzada + modelo final (GPU: pocos minutos; CPU: ~1 h)
    python entrenar_modelo.py --rapido     # prueba de humo (2 épocas, sin validación cruzada)

Requiere `StanfordExtra-master/` (el cuaderno lo descarga solo).
Escribe `resultados/validacion_cruzada.json` y `modelos/modelo_puntos_clave.keras`.
"""

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:                                   # con PyTorch instalado se entrena en la GPU (CUDA)
    import torch  # noqa: F401
    os.environ.setdefault("KERAS_BACKEND", "torch")
except ImportError:                    # si no, TensorFlow (CPU en Windows nativo)
    pass

import keras
import numpy as np
from keras import layers, ops
from PIL import Image, ImageEnhance

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
SEED = 0
BASE = os.path.dirname(os.path.abspath(__file__))
SE_DIR = os.path.join(BASE, "StanfordExtra-master")
JSON_PATH = os.path.join(SE_DIR, "StanfordExtra_sample.json")
MODEL_PATH = os.path.join(BASE, "modelos", "modelo_puntos_clave.keras")
CV_PATH = os.path.join(BASE, "resultados", "validacion_cruzada.json")

IMG_SIZE = 224
HM = 56                         # lado del mapa de calor (stride 4)
NUM_KP = 24
VARIANTS = 40                   # aumentos por imagen
EPOCHS = 50
BATCH = 16
LR = 1e-3
SIGMA = 2.0                     # ancho de la gaussiana objetivo (en celdas del mapa)
UNFREEZE_FROM = "block_13_expand"
FOLDS = 5

# Pares simétricos: al voltear la imagen hay que intercambiar las etiquetas
SWAP = {0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11,
        6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5,
        14: 15, 15: 14, 18: 19, 19: 18, 20: 21, 21: 20}
PERM = np.array([SWAP.get(i, i) for i in range(NUM_KP)])


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def cargar(entry):
    """Imagen RGB 224×224, coordenadas normalizadas (sin recortar) y visibilidad."""
    img = (Image.open(os.path.join(SE_DIR, "sample_imgs", entry["img_path"]))
           .convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    w, h = entry["img_width"], entry["img_height"]
    kps = np.array([[p[0] / w, p[1] / h] for p in entry["joints"]], dtype="float32")
    vis = np.array([1.0 if p[2] > 0 else 0.0 for p in entry["joints"]], dtype="float32")
    return np.array(img, dtype="uint8"), kps, vis


def aumentar(img, kps, vis, g):
    """Rotación ±15°, escala 0.8-1.2, traslación ±6 %, volteo L/R y fotometría."""
    S = IMG_SIZE
    ang = np.deg2rad(g.uniform(-15, 15))
    s = g.uniform(0.8, 1.2)
    t = g.uniform(-0.06, 0.06, 2) * S
    c = np.array([S / 2, S / 2])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
    A = np.eye(3)
    A[:2, :2] = R
    A[:2, 2] = c + t - R @ c

    if g.random() < 0.5:                                # volteo + intercambio L/R
        F = np.array([[-1.0, 0, S - 1], [0, 1.0, 0], [0, 0, 1]])
        A = F @ A
        kps, vis = kps[PERM].copy(), vis[PERM].copy()

    Ai = np.linalg.inv(A)                               # PIL usa la transformada inversa
    im = Image.fromarray(img).transform(
        (S, S), Image.AFFINE, (Ai[0, 0], Ai[0, 1], Ai[0, 2], Ai[1, 0], Ai[1, 1], Ai[1, 2]),
        resample=Image.BILINEAR)

    ph = np.concatenate([kps * S, np.ones((NUM_KP, 1))], 1) @ A.T
    k2 = (ph[:, :2] / S).astype("float32")
    vis = vis * ((k2 >= 0) & (k2 < 1)).all(1)           # fuera de cuadro => no visible

    im = ImageEnhance.Brightness(im).enhance(g.uniform(0.7, 1.3))
    im = ImageEnhance.Contrast(im).enhance(g.uniform(0.75, 1.25))
    im = ImageEnhance.Color(im).enhance(g.uniform(0.7, 1.3))
    return np.array(im, dtype="uint8"), np.clip(k2, 0, 1), vis


def mapas_objetivo(kps, vis):
    """(n, 24, 2) + (n, 24) -> (n, 56, 56, 24): gaussianas en los puntos visibles."""
    ys, xs = np.mgrid[0:HM, 0:HM]
    T = np.zeros((len(kps), HM, HM, NUM_KP), "float32")
    for n in range(len(kps)):
        for j in range(NUM_KP):
            if vis[n, j] > 0:
                cx, cy = kps[n, j] * HM - 0.5
                T[n, :, :, j] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * SIGMA ** 2))
    return T


def generar(imgs, kps, vis, idx, g, variantes):
    X, K, V = [], [], []
    for i in idx:
        for _ in range(variantes):
            a, b, c = aumentar(imgs[i], kps[i], vis[i], g)
            X.append(a), K.append(b), V.append(c)
    X, K, V = np.stack(X), np.stack(K), np.stack(V).astype("float32")
    T = mapas_objetivo(K, V)
    peso = np.broadcast_to(V[:, None, None, :], T.shape)   # 1 donde el punto está anotado
    return X, np.concatenate([T, peso], -1).astype("float32")


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def construir_modelo():
    """MobileNetV2 (ImageNet) + 3 deconvoluciones -> 24 mapas de calor de 56×56."""
    base = keras.applications.MobileNetV2(weights="imagenet", include_top=False,
                                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    nombres = [l.name for l in base.layers]
    for l in base.layers[nombres.index(UNFREEZE_FROM):]:
        if not isinstance(l, layers.BatchNormalization):   # BN siempre en modo inferencia
            l.trainable = True

    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    x = keras.applications.mobilenet_v2.preprocess_input(layers.Rescaling(1.0)(inp))
    x = base(x)
    for _ in range(3):                                      # 7 -> 14 -> 28 -> 56
        x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    out = layers.Conv2D(NUM_KP, 1, name="mapas_de_calor")(x)
    return keras.Model(inp, out, name="detector_puntos_clave")


def perdida(y_true, y_pred):
    """MSE de mapas de calor; el pico pesa más que el fondo y los puntos sin anotar casi no cuentan."""
    t, w = y_true[..., :NUM_KP], y_true[..., NUM_KP:]
    return ops.mean(ops.square(y_pred - t) * (1 + 30 * t) * (0.1 + 0.9 * w)) * 100


def entrenar(X, Y, epocas):
    model = construir_modelo()
    model.compile(loss=perdida, optimizer=keras.optimizers.AdamW(LR, weight_decay=1e-4))
    cb = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, min_lr=5e-5)
    model.fit(X, Y, batch_size=BATCH, epochs=epocas, verbose=2, callbacks=[cb])
    return model


# --------------------------------------------------------------------------- #
# Decodificación y métricas
# --------------------------------------------------------------------------- #
def decodificar(mapas):
    """(n, 56, 56, 24) -> kps (n, 24, 2) en [0,1] y pico (n, 24).

    Posición = centroide ponderado alrededor del máximo (precisión sub-celda).
    """
    n = len(mapas)
    kps = np.zeros((n, NUM_KP, 2), "float32")
    pico = np.zeros((n, NUM_KP), "float32")
    for a in range(n):
        for j in range(NUM_KP):
            h = mapas[a, :, :, j]
            r, c = np.unravel_index(h.argmax(), h.shape)
            pico[a, j] = h[r, c]
            r0, r1, c0, c1 = max(r - 2, 0), min(r + 3, HM), max(c - 2, 0), min(c + 3, HM)
            w = np.clip(h[r0:r1, c0:c1], 0, None) + 1e-9
            yy, xx = np.mgrid[r0:r1, c0:c1]
            kps[a, j] = [((xx * w).sum() / w.sum() + 0.5) / HM,
                         ((yy * w).sum() / w.sum() + 0.5) / HM]
    return kps, pico


def error_pct(pred, gt, vis):
    d = np.linalg.norm(pred - gt, axis=-1)
    return float((d * vis).sum() / vis.sum() * 100)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="prueba de humo: 2 épocas, sin CV")
    args = ap.parse_args()

    if not os.path.exists(JSON_PATH):
        raise SystemExit("Falta 'StanfordExtra-master/'. Ejecuta el cuaderno una vez "
                         "para que descargue el dataset de muestra.")
    keras.utils.set_random_seed(SEED)
    g = np.random.default_rng(SEED)
    epocas, variantes = (2, 4) if args.rapido else (EPOCHS, VARIANTS)

    datos = json.load(open(JSON_PATH, encoding="utf-8"))
    imgs, kps, vis = (np.stack(z) for z in zip(*(cargar(e) for e in datos)))
    gt = np.clip(kps, 0, 1)
    n = len(imgs)
    print(f"imágenes: {imgs.shape} | puntos anotados por imagen: {vis.sum(1).mean():.1f}/24")

    if not args.rapido:
        # ---- validación cruzada: error SOLO sobre imágenes no vistas ----
        errs, base_errs, picos_vis, picos_no = [], [], [], []
        for k in range(FOLDS):
            val = np.arange(n)[np.arange(n) % FOLDS == k]
            tr = np.array([i for i in range(n) if i not in val])
            t0 = time.time()
            X, Y = generar(imgs, kps, vis, tr, g, variantes)
            model = entrenar(X, Y, epocas)
            pk, pico = decodificar(model.predict(imgs[val], verbose=0))
            media = (gt[tr] * vis[tr][..., None]).sum(0) / np.maximum(vis[tr].sum(0)[:, None], 1)
            e = error_pct(pk, gt[val], vis[val])
            b = error_pct(np.broadcast_to(media, pk.shape), gt[val], vis[val])
            errs.append(e), base_errs.append(b)
            sup = np.arange(20)
            picos_vis += list(pico[:, sup][vis[val][:, sup] > 0])
            picos_no += list(pico[:, sup][vis[val][:, sup] == 0])
            print(f"[pliegue {k + 1}/{FOLDS}] error en no vistas {e:.2f} % | "
                  f"pose media {b:.2f} % | {time.time() - t0:.0f}s", flush=True)
            keras.backend.clear_session()

        # umbral de visibilidad: el que mejor separa picos de puntos anotados / no anotados
        cand = np.linspace(0.05, 0.9, 86)
        pv, pn = np.array(picos_vis), np.array(picos_no)
        acc = [((pv >= u).sum() + (pn < u).sum()) / (len(pv) + len(pn)) for u in cand]
        umbral = float(cand[int(np.argmax(acc))])
        resumen = {
            "error_no_vistas_pct": float(np.mean(errs)),
            "error_por_pliegue_pct": [round(e, 3) for e in errs],
            "baseline_pose_media_pct": float(np.mean(base_errs)),
            "umbral_pico_visibilidad": round(umbral, 3),
            "acierto_visibilidad_pct": round(100 * max(acc), 2),
            "pliegues": FOLDS, "epocas": epocas, "variantes_por_imagen": variantes,
        }
        os.makedirs(os.path.dirname(CV_PATH), exist_ok=True)
        json.dump(resumen, open(CV_PATH, "w", encoding="utf-8"), indent=2)
        print("\n=== VALIDACIÓN CRUZADA ===")
        print(json.dumps(resumen, indent=2))

    # ---- modelo final con las 25 imágenes ----
    X, Y = generar(imgs, kps, vis, np.arange(n), g, variantes)
    model = entrenar(X, Y, epocas)
    pk, _ = decodificar(model.predict(imgs, verbose=0))
    print(f"error sobre las imágenes de entrenamiento (referencia): {error_pct(pk, gt, vis):.2f} %")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    limpio = construir_modelo()                        # sin estado del optimizador (22 MB, no 60)
    limpio.set_weights(model.get_weights())
    limpio.save(MODEL_PATH)
    print(f"modelo guardado en {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
