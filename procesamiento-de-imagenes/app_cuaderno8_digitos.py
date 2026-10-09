"""
Cuaderno 8 A2 - dibuja un numero con el mouse y el modelo te dice que digito es.

Uso:
    .venv-dibujo/Scripts/python.exe app_cuaderno8_digitos.py                (usa mnist_conv2d.keras de esta carpeta)
    .venv-dibujo/Scripts/python.exe app_cuaderno8_digitos.py otra_ruta.keras

Abre http://127.0.0.1:5000 en el navegador.
"""

import base64
import io
import os
import sys

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "mnist_conv2d.keras")

if not os.path.exists(RUTA_MODELO):
    sys.exit("No encuentro el modelo: " + RUTA_MODELO +
             "\nDescarga mnist_conv2d.keras desde tu Drive (MyDrive/modelos_cnn) y ponlo en esta carpeta.")

print("Cargando", RUTA_MODELO, "...")
import tensorflow as tf
modelo = tf.keras.models.load_model(RUTA_MODELO)
print("Modelo cargado. Entrada esperada:", modelo.input_shape)

app = Flask(__name__)


def a_mnist(tinta):
    """Convierte el dibujo (uint8, 255 = trazo) al formato MNIST: 28x28, digito de 20 px
    centrado por centro de masa. Sin esto el modelo falla con dibujos reales."""
    ys, xs = np.where(tinta > 40)
    if xs.size == 0:
        return None
    recorte = tinta[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = recorte.shape
    escala = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * escala))), max(1, int(round(w * escala)))
    chico = np.asarray(Image.fromarray(recorte).resize((nw, nh), Image.LANCZOS), np.float32) / 255.0

    lienzo = np.zeros((28, 28), np.float32)
    oy, ox = (28 - nh) // 2, (28 - nw) // 2
    lienzo[oy:oy + nh, ox:ox + nw] = chico

    fy, fx = np.mgrid[0:28, 0:28]
    masa = lienzo.sum()
    if masa > 0:
        dy = int(round(14 - (lienzo * fy).sum() / masa))
        dx = int(round(14 - (lienzo * fx).sum() / masa))
        lienzo = np.roll(np.roll(lienzo, dy, axis=0), dx, axis=1)
    return lienzo


def a_png_b64(imagen_gris, tamano):
    img = Image.fromarray(imagen_gris.astype(np.uint8)).resize((tamano, tamano), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/predecir")
def predecir():
    png = base64.b64decode(request.get_json()["imagen"].split(",")[-1])
    gris = np.asarray(Image.open(io.BytesIO(png)).convert("L"), np.uint8)
    x = a_mnist(255 - gris)          # el canvas dibuja negro sobre blanco
    if x is None:
        return jsonify({"error": "Dibuja algo primero"})

    probs = modelo.predict(x[None, ..., None], verbose=0)[0]
    return jsonify({
        "digito": int(np.argmax(probs)),
        "probabilidad": float(np.max(probs)),
        "probabilidades": [round(float(p), 4) for p in probs],
        "lo_que_ve": a_png_b64(x * 255, 140),
    })


PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Que numero es</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;padding:24px;background:#111;color:#eee;
      display:flex;gap:28px;flex-wrap:wrap;justify-content:center}
 .tarjeta{background:#1c1c1c;border:1px solid #333;border-radius:14px;padding:18px}
 canvas{background:#fff;border-radius:10px;touch-action:none;cursor:crosshair;display:block}
 button{margin-top:12px;margin-right:8px;padding:10px 18px;font-size:15px;border-radius:8px;
        border:1px solid #444;background:#2a2a2a;color:#eee;cursor:pointer}
 button:hover{background:#3a3a3a}
 #salida{font-size:90px;font-weight:700;line-height:1;color:#7ee787;text-align:center}
 #conf{text-align:center;color:#9aa;margin-top:4px}
 .fila{display:flex;align-items:center;gap:8px;font-size:13px;margin:5px 0;color:#bbb}
 .barra{background:#333;height:9px;flex:1;border-radius:5px;overflow:hidden}
 .barra i{display:block;height:100%;background:#7ee787}
 img.mini{width:140px;image-rendering:pixelated;border-radius:8px;background:#000}
 small{color:#888}
</style></head><body>

<div class="tarjeta">
  <canvas id="lienzo" width="280" height="280"></canvas>
  <button id="limpiar">Borrar</button>
  <small>Dibuja un solo digito, grande y grueso (como los de MNIST).</small>
</div>

<div class="tarjeta" style="min-width:320px">
  <div id="salida">?</div>
  <div id="conf">dibuja para predecir</div>
  <div id="barras"></div>
  <div style="margin-top:14px">
    <small>Lo que ve el modelo (28x28):</small><br>
    <img id="mini" class="mini" alt="">
  </div>
</div>

<script>
const cv = document.getElementById('lienzo'), ctx = cv.getContext('2d');
let pintando = false;
ctx.lineWidth = 18; ctx.lineCap = ctx.lineJoin = 'round'; ctx.strokeStyle = '#000';
function pos(e){ const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; }
function borrar(){ ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cv.width, cv.height); }

cv.addEventListener('pointerdown', e => { pintando = true; cv.setPointerCapture(e.pointerId);
  const [x, y] = pos(e); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + .1, y + .1); ctx.stroke(); });
cv.addEventListener('pointermove', e => { if (!pintando) return;
  const [x, y] = pos(e); ctx.lineTo(x, y); ctx.stroke(); });
cv.addEventListener('pointerup', () => { pintando = false; predecir(); });
cv.addEventListener('pointerleave', () => { if (pintando) { pintando = false; predecir(); } });

async function predecir(){
  const r = await fetch('/predecir', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({imagen: cv.toDataURL('image/png')})});
  const d = await r.json();
  if (d.error){ document.getElementById('salida').textContent = '?';
    document.getElementById('conf').textContent = d.error; return; }
  document.getElementById('salida').textContent = d.digito;
  document.getElementById('conf').textContent = 'confianza ' + (d.probabilidad * 100).toFixed(1) + '%';
  document.getElementById('mini').src = d.lo_que_ve;
  document.getElementById('barras').innerHTML = d.probabilidades.map((p, i) =>
    '<div class="fila"><b>' + i + '</b><span class="barra"><i style="width:' + (p * 100).toFixed(1) + '%"></i></span>' +
    (p * 100).toFixed(1) + '%</div>').join('');
}

document.getElementById('limpiar').onclick = () => { borrar(); document.getElementById('salida').textContent = '?';
  document.getElementById('conf').textContent = 'dibuja para predecir'; document.getElementById('barras').innerHTML = '';
  document.getElementById('mini').src = ''; };

borrar();
</script></body></html>"""


@app.get("/")
def inicio():
    return PAGINA


if __name__ == "__main__":
    print("Abre esto en el navegador:  http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
