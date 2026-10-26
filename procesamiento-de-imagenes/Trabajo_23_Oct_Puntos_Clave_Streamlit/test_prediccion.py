# -*- coding: utf-8 -*-
"""
Validación rápida del modelo guardado (antes de desplegar en Streamlit).

Comprueba que el .keras carga, que entrega 24 mapas de calor de 56×56 y que los puntos
NO colapsan en un solo sitio. Sale con código 1 si algo falla (sirve en CI).

Uso:
    python test_prediccion.py
"""
import io
import os
import sys

import numpy as np
from PIL import Image

from puntos_clave import MODEL_PATH, SUPERVISADOS, cargar_modelo, dibujar, predecir

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
MUESTRAS = os.path.join(BASE, "muestras")


def main():
    modelo = cargar_modelo(MODEL_PATH)
    print(f"[1] Modelo cargado: {os.path.basename(MODEL_PATH)} "
          f"({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")
    print(f"    salida del modelo: {modelo.output_shape[1:]} (24 mapas de calor)")
    assert tuple(modelo.output_shape[1:]) == (56, 56, 24), "la salida debe ser 56×56×24"

    os.makedirs(os.path.join(BASE, "resultados"), exist_ok=True)
    ok = True
    for nombre in sorted(os.listdir(MUESTRAS)):
        if not nombre.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(MUESTRAS, nombre)).convert("RGB")
        kps, vis = predecir(modelo, img)

        assert kps.shape == (24, 2), f"forma inesperada {kps.shape}"
        assert kps.min() >= 0.0 and kps.max() <= 1.0, "coordenadas fuera de [0, 1]"

        usados = [i for i in SUPERVISADOS if vis[i] >= 0.5]
        disp = (np.linalg.norm(kps[usados] - kps[usados].mean(0), axis=1).mean() * 100
                if len(usados) >= 2 else 0.0)

        estado = "OK " if disp > 5 else "COLAPSO"
        ok = ok and disp > 5
        print(f"[2] {nombre:20s} visibles {len(usados):2d}/20 · dispersión "
              f"{disp:5.1f} % · confianza media {vis[SUPERVISADOS].mean():.2f} -> {estado}")

        # se guarda una copia anotada para inspección visual
        dibujar(img, kps, vis=vis).save(os.path.join(BASE, "resultados",
                                                     f"muestra_{nombre}"))

    print("\n[OK] El modelo predice puntos distribuidos sobre el perro" if ok
          else "\n[ERROR] Los puntos siguen colapsando: revisa el entrenamiento")


if __name__ == "__main__":
    main()
