# -*- coding: utf-8 -*-
"""
evaluar_modelo.py — Métricas y figuras del detector de puntos clave.

Uso:
    python evaluar_modelo.py

Qué hace:
  1. Carga `modelos/modelo_puntos_clave.keras`.
  2. Predice sobre las 25 imágenes de muestra de StanfordExtra y calcula:
       · dispersión del esqueleto (qué tanto se separan los 24 puntos),
       · error medio vs. la anotación real, en % de la imagen,
       · aciertos por debajo de 0,10 (10 % de la imagen).
  3. Genera en `resultados/`:
       · `figura_predicciones.png`  (contacto 2×3 con la predicción anotada),
       · `figura_comparacion.png`   (original | predicción | ground truth).

Requiere la carpeta `StanfordExtra-master/` (el cuaderno la descarga sola).
"""

import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from puntos_clave import (MODEL_PATH, SUPERVISADOS, cargar_modelo, dibujar,
                          predecir)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SE_DIR = os.path.join(BASE_DIR, "StanfordExtra-master")
JSON_PATH = os.path.join(SE_DIR, "StanfordExtra_sample.json")
IMG_DIR = os.path.join(SE_DIR, "sample_imgs")
SALIDA = os.path.join(BASE_DIR, "resultados")


def _fuente(tam=18):
    try:
        return ImageFont.truetype("arial.ttf", tam)
    except OSError:
        return ImageFont.load_default()


def anotar(imagen, kps, vis=None, contorno=True):
    return dibujar(imagen, kps, vis=vis, contorno=contorno, grosor=3, tam_punto=1.0)


def contacto(celdas, columnas, ancho_celda=420, titulo=""):
    """Compone una hoja de contacto con PIL (sin matplotlib)."""
    filas = int(np.ceil(len(celdas) / columnas))
    alto_celda = int(ancho_celda * 0.78)
    margen, pie = 14, 34
    W = columnas * ancho_celda + (columnas + 1) * margen
    H = filas * (alto_celda + pie) + (filas + 1) * margen + (46 if titulo else 0)
    hoja = Image.new("RGB", (W, H), (12, 18, 32))
    d = ImageDraw.Draw(hoja)
    f_tit, f_pie = _fuente(26), _fuente(17)

    y0 = margen
    if titulo:
        d.text((margen, margen), titulo, font=f_tit, fill=(226, 232, 240))
        y0 += 46

    for i, (img, pie_txt) in enumerate(celdas):
        r, c = divmod(i, columnas)
        x = margen + c * (ancho_celda + margen)
        y = y0 + r * (alto_celda + pie + margen)
        img.thumbnail((ancho_celda, alto_celda), Image.LANCZOS)
        hoja.paste(img, (x + (ancho_celda - img.width) // 2, y))
        d.text((x + 4, y + alto_celda + 6), pie_txt, font=f_pie, fill=(148, 163, 184))
    return hoja


def main():
    if not os.path.exists(JSON_PATH):
        raise SystemExit(
            "No se encontró 'StanfordExtra-master/'. Ejecuta el cuaderno una vez "
            "para que descargue el dataset de muestra."
        )
    os.makedirs(SALIDA, exist_ok=True)
    modelo = cargar_modelo(MODEL_PATH)
    datos = json.load(open(JSON_PATH, encoding="utf-8"))
    print(f"Modelo: {MODEL_PATH}\nImágenes de evaluación: {len(datos)}\n")

    errores, dispersiones, celdas_cmp, celdas_pred = [], [], [], []
    acc_vis = []
    filas = []
    for e in datos:
        ruta = os.path.join(IMG_DIR, e["img_path"])
        img = Image.open(ruta).convert("RGB")
        kps, vis_pred = predecir(modelo, img)

        w, h = e["img_width"], e["img_height"]
        gt = np.array([[p[0] / w, p[1] / h] for p in e["joints"]], dtype="float32")
        vis = np.array([p[2] for p in e["joints"]], dtype="float32")
        err = np.linalg.norm(kps - gt, axis=1) * 100          # % de la imagen
        err_vis = float(err[vis > 0].mean()) if (vis > 0).any() else float("nan")
        # dispersión sobre los puntos que el modelo considera visibles
        usados = [i for i in SUPERVISADOS if vis_pred[i] >= 0.5] or SUPERVISADOS
        disp = float(np.linalg.norm(kps[usados] - kps[usados].mean(0), axis=1).mean() * 100)
        aciertos_vis = float(np.mean((vis_pred[SUPERVISADOS] >= 0.5) == (vis[SUPERVISADOS] > 0)))
        acc_vis.append(aciertos_vis)

        errores.append(err_vis)
        dispersiones.append(disp)
        filas.append((e["img_path"], err_vis, disp))
        print(f"  {e['img_path'][:44]:44s} error={err_vis:5.2f}%  dispersión={disp:5.2f}%")

        pred_img = anotar(img, kps, vis_pred)
        gt_img = anotar(img, np.clip(gt, 0, 1), vis)
        celdas_pred.append((pred_img, f"{e['img_path'].split('/')[-1]} · error {err_vis:.1f}%"))
        if len(celdas_cmp) < 3:
            lado = max(img.size)
            fila = Image.new("RGB", (lado * 3 + 24, lado), (12, 18, 32))
            for j, (im, et) in enumerate([(img, "original"), (pred_img, "predicción"),
                                          (gt_img, "anotación real")]):
                im2 = im.copy()
                im2.thumbnail((lado, lado), Image.LANCZOS)
                fila.paste(im2, (j * (lado + 12), 0))
                ImageDraw.Draw(fila).text((j * (lado + 12) + 8, 8), et,
                                          font=_fuente(22), fill=(255, 255, 255))
            celdas_cmp.append((fila, e["img_path"].split("/")[-1]))

    errores = np.array(errores)
    dispersiones = np.array(dispersiones)
    print("\n=== RESUMEN ===")
    print(f"Error medio (puntos visibles): {errores.mean():.2f} % de la imagen")
    print(f"Acierto de visibilidad:        {100 * np.mean(acc_vis):.1f} % de los 20 puntos anotados")
    print(f"Error mediano:                 {np.median(errores):.2f} %")
    print(f"Puntos a menos de 10 %:        {100 * (errores < 10).mean():.0f} % de las imágenes")
    print(f"Dispersión media del esqueleto: {dispersiones.mean():.2f} %")
    print("   (si es < 1 % los 24 puntos se están dibujando en el mismo sitio)")

    hoja = contacto(celdas_pred, columnas=5, ancho_celda=360,
                    titulo="Predicciones sobre la muestra de StanfordExtra (25 imágenes)")
    p1 = os.path.join(SALIDA, "figura_predicciones.png")
    hoja.save(p1)

    hoja2 = contacto(celdas_cmp, columnas=1, ancho_celda=1200,
                     titulo="Original | Predicción | Anotación real")
    p2 = os.path.join(SALIDA, "figura_comparacion.png")
    hoja2.save(p2)

    with open(os.path.join(SALIDA, "metricas.txt"), "w", encoding="utf-8") as f:
        f.write("imagen,error_pct,dispersion_pct\n")
        for nombre, err, disp in filas:
            f.write(f"{nombre},{err:.3f},{disp:.3f}\n")
        f.write(f"\nERROR_MEDIO,{errores.mean():.3f}\n")
        f.write(f"ERROR_MEDIANO,{np.median(errores):.3f}\n")
        f.write(f"DISPERSION_MEDIA,{dispersiones.mean():.3f}\n")
        f.write(f"ACIERTO_VISIBILIDAD,{100 * np.mean(acc_vis):.3f}\n")

    print(f"\nFiguras guardadas en:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
