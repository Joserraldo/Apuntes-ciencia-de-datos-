# -*- coding: utf-8 -*-
"""
puntos_clave.py — Núcleo reutilizable del detector de puntos clave en perros.

Se usa desde:
  · app_streamlit.py      (despliegue)
  · evaluar_modelo.py     (evaluación + figuras)
  · el cuaderno           (misma visualización que la app)

Contenido:
  · NOMBRES / COLORES      metadatos oficiales de los 24 keypoints de StanfordExtra
  · cargar_modelo()        carga el .keras
  · predecir()             imagen PIL -> (24, 2) coordenadas normalizadas
  · contorno_estimado()    envolvente convexa + suavizado Chaikin (el "borde")
  · dibujar()              puntos + contorno con anti-aliasing (sin OpenCV)
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

IMG_SIZE = 224
NUM_PUNTOS = 24
UMBRAL_PICO = 0.19            # pico del mapa de calor a partir del cual un punto cuenta como visible
SUPER = 3                      # factor de supermuestreo para anti-aliasing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelos", "modelo_puntos_clave.keras")

# Nombres y colores oficiales del dataset (keypoint_definitions.csv)
NOMBRES = [
    "Pata delantera izq. · almohadilla",
    "Pata delantera izq. · articulación media",
    "Pata delantera izq. · parte superior",
    "Pata trasera izq. · almohadilla",
    "Pata trasera izq. · articulación media",
    "Pata trasera izq. · parte superior",
    "Pata delantera der. · almohadilla",
    "Pata delantera der. · articulación media",
    "Pata delantera der. · parte superior",
    "Pata trasera der. · almohadilla",
    "Pata trasera der. · articulación media",
    "Pata trasera der. · parte superior",
    "Inicio de la cola",
    "Fin de la cola",
    "Base de la oreja izquierda",
    "Base de la oreja derecha",
    "Nariz",
    "Barbilla",
    "Punta de la oreja izquierda",
    "Punta de la oreja derecha",
    "Ojo izquierdo",
    "Ojo derecho",
    "Cruz (withers)",
    "Garganta",
]

COLORES = [
    "#d82400", "#d82400", "#d82400",   # patas delanteras izquierdas
    "#fcfc00", "#fcfc00", "#fcfc00",   # patas traseras izquierdas
    "#48b455", "#48b455", "#48b455",   # patas delanteras derechas
    "#0090aa", "#0090aa", "#0090aa",   # patas traseras derechas
    "#d848ff", "#d848ff",              # cola
    "#fc90aa", "#006caa",              # base de las orejas
    "#d89000", "#d89000",              # nariz / barbilla
    "#fc90aa", "#006caa",              # punta de las orejas
    "#ededed", "#ededed",              # ojos
    "#a9d08e", "#a9d08e",              # cruz / garganta
]

# Compatibilidad imgaug / NumPy 2.x (el cuaderno aplica el mismo parche)

# --------------------------------------------------------------------------- #
# Puntos que la muestra de StanfordExtra SÍ anota.
#
# En `StanfordExtra_sample.json` los puntos 20-23 (ojo izq., ojo der., cruz y
# garganta) aparecen siempre con visibilidad 0: solo existen en el subconjunto
# Animal-Pose. Como nunca se anotan, el modelo no tiene señal para aprenderlos y
# devuelve valores arbitrarios (típicamente pegados a la esquina (0, 0)).
# Se dibujan con un anillo hueco y NO se usan para el contorno: si se usaran,
# arrastrarían la silueta hacia la esquina de la imagen.
SUPERVISADOS = list(range(20))

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, str, bytes],
    }
for _a, _v in [("bool", bool), ("int", int), ("float", float), ("complex", complex)]:
    if not hasattr(np, _a):
        setattr(np, _a, _v)


# --------------------------------------------------------------------------- #
# Modelo e inferencia
# --------------------------------------------------------------------------- #
def cargar_modelo(ruta=None):
    """Carga el modelo Keras entrenado (sin la pérdida de entrenamiento)."""
    import keras

    return keras.models.load_model(ruta or MODEL_PATH, compile=False)


def _decodificar(mapas):
    """(56, 56, 24) mapas de calor -> kps (24, 2) en [0, 1] y pico (24,).

    La posición es el centroide ponderado alrededor del máximo (precisión sub-celda).
    """
    h_, w_, n = mapas.shape
    kps = np.zeros((n, 2), dtype="float32")
    pico = np.zeros(n, dtype="float32")
    for j in range(n):
        h = mapas[:, :, j]
        r, c = np.unravel_index(h.argmax(), h.shape)
        pico[j] = h[r, c]
        r0, r1, c0, c1 = max(r - 2, 0), min(r + 3, h_), max(c - 2, 0), min(c + 3, w_)
        w = np.clip(h[r0:r1, c0:c1], 0, None) + 1e-9
        yy, xx = np.mgrid[r0:r1, c0:c1]
        kps[j] = [((xx * w).sum() / w.sum() + 0.5) / w_, ((yy * w).sum() / w.sum() + 0.5) / h_]
    return kps, pico


def predecir(model, imagen: Image.Image):
    """
    Imagen PIL (cualquier tamaño) -> (kps, vis)

      kps : (24, 2) coordenadas normalizadas [0, 1]
      vis : (24,)  confianza de que el punto se vea (0..1; >= 0.5 = visible)

    El modelo devuelve 24 mapas de calor de 56×56. La confianza sale de la altura del
    pico de cada mapa, reescalada para que el umbral calibrado por validación cruzada
    (UMBRAL_PICO) corresponda a 0.5.
    """
    img = ImageOps.exif_transpose(imagen).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)[None, ...]
    mapas = np.asarray(model.predict(arr, verbose=0))[0]
    kps, pico = _decodificar(mapas)
    vis = np.clip(pico / (2 * UMBRAL_PICO), 0.0, 1.0)
    return np.clip(kps, 0.0, 1.0), vis.astype("float32")


# --------------------------------------------------------------------------- #
# Geometría del contorno
# --------------------------------------------------------------------------- #
def _envolvente(puntos):
    """Envolvente convexa (monotone chain). puntos: iterable de (x, y)."""
    pts = sorted(set(map(tuple, puntos)))
    if len(pts) < 3:
        return pts

    def cruz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    inferior, superior = [], []
    for p in pts:
        while len(inferior) >= 2 and cruz(inferior[-2], inferior[-1], p) <= 0:
            inferior.pop()
        inferior.append(p)
    for p in reversed(pts):
        while len(superior) >= 2 and cruz(superior[-2], superior[-1], p) <= 0:
            superior.pop()
        superior.append(p)
    return inferior[:-1] + superior[:-1]


def _chaikin(poligono, iteraciones=3):
    """Suaviza un polígono cerrado recortando esquinas (algoritmo de Chaikin)."""
    pts = list(poligono)
    for _ in range(iteraciones):
        nuevo, n = [], len(pts)
        for i in range(n):
            p = np.asarray(pts[i], float)
            q = np.asarray(pts[(i + 1) % n], float)
            nuevo.append(tuple(0.75 * p + 0.25 * q))
            nuevo.append(tuple(0.25 * p + 0.75 * q))
        pts = nuevo
    return pts


def contorno_estimado(kps_norm, forma_imagen, escala=1, vis=None, umbral=0.5):
    """Borde del perro = envolvente suavizada de los puntos ANOTADOS y VISIBLES."""
    w, h = forma_imagen
    usados = []
    for i in SUPERVISADOS:
        if i >= len(kps_norm):
            continue
        if vis is not None and float(vis[i]) < umbral:
            continue
        usados.append(kps_norm[i])
    pts = [(float(x) * w * escala, float(y) * h * escala) for x, y in usados]
    return _chaikin(_envolvente(pts), iteraciones=3)


# --------------------------------------------------------------------------- #
# Dibujo
# --------------------------------------------------------------------------- #
def dibujar(imagen: Image.Image, kps_norm, vis=None, contorno=True, etiquetas=False,
            grosor=3, tam_punto=1.0, umbral=0.5) -> Image.Image:
    """
    Devuelve una copia de `imagen` con:
      · el contorno azul (envolvente suavizada de los puntos visibles), y
      · los puntos con el color oficial de su grupo anatómico.

    `vis` es la probabilidad de visibilidad por punto que predice el modelo:
      · vis >= umbral  -> punto relleno (se usa también para el contorno)
      · vis <  umbral  -> anillo tenue (el modelo no ve ese punto en la imagen)
    Los puntos que StanfordExtra no anota (SUPERVISADOS) nunca son rellenos.

    Se dibuja a resolución SUPER× y se reduce al final: bordes y círculos suaves
    sin necesidad de OpenCV.
    """
    base = imagen.convert("RGB")
    w, h = base.size
    grande = (w * SUPER, h * SUPER)
    lienzo = base.resize(grande, Image.LANCZOS).convert("RGBA")
    if vis is None:
        vis = np.ones(len(kps_norm), dtype="float32")

    # ---------- contorno ----------
    if contorno and len(kps_norm) >= 3:
        poli = contorno_estimado(kps_norm, (w, h), escala=SUPER, vis=vis, umbral=umbral)
        if len(poli) >= 3:
            relleno = Image.new("RGBA", grande, (0, 0, 0, 0))
            ImageDraw.Draw(relleno).polygon(poli, fill=(56, 189, 248, 42))

            halo = Image.new("RGBA", grande, (0, 0, 0, 0))
            ImageDraw.Draw(halo).line(poli + [poli[0]], fill=(56, 189, 248, 160),
                                      width=int(grosor * 3 * SUPER), joint="curve")
            halo = halo.filter(ImageFilter.GaussianBlur(radius=max(1, int(2.2 * grosor * SUPER))))

            linea = Image.new("RGBA", grande, (0, 0, 0, 0))
            ImageDraw.Draw(linea).line(poli + [poli[0]], fill=(125, 211, 252, 255),
                                       width=int(grosor * SUPER), joint="curve")

            lienzo = Image.alpha_composite(lienzo, relleno)
            lienzo = Image.alpha_composite(lienzo, halo)
            lienzo = Image.alpha_composite(lienzo, linea)

    # ---------- puntos ----------
    capa = Image.new("RGBA", grande, (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    r = max(2.5, min(w, h) * 0.009 * tam_punto) * SUPER

    for i, (x_n, y_n) in enumerate(kps_norm):
        cx, cy = float(x_n) * w * SUPER, float(y_n) * h * SUPER
        color = COLORES[i % len(COLORES)]
        if i in SUPERVISADOS and float(vis[i]) >= umbral:
            # halo blanco: el punto se ve sobre cualquier fondo
            d.ellipse([cx - r * 1.75, cy - r * 1.75, cx + r * 1.75, cy + r * 1.75],
                      fill=(255, 255, 255, 205))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(15, 23, 42, 230),
                      width=max(1, int(SUPER * 0.6)))
        else:
            # no anotado o no visible: solo un anillo tenue
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(203, 213, 225, 190),
                      width=max(2, int(SUPER * 0.8)))
        if etiquetas:
            tx, ty = cx + r * 1.9, cy - r * 2.6
            ancho = 7 if i + 1 < 10 else 13
            d.rectangle([tx - 2 * SUPER, ty - 1 * SUPER, tx + ancho * SUPER, ty + 11 * SUPER],
                        fill=(15, 23, 42, 175))
            d.text((tx, ty), str(i + 1), fill=(255, 255, 255, 255))

    lienzo = Image.alpha_composite(lienzo, capa)
    return lienzo.convert("RGB").resize((w, h), Image.LANCZOS)
