# -*- coding: utf-8 -*-
"""Prueba E2E de app_streamlit.py con AppTest (simula al usuario)."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app_streamlit.py", default_timeout=300)
at.run()
print("[1] excepciones:", at.exception)
assert not at.exception, "la app lanzó una excepción al cargar"

print("[2] tabs:", [t.label for t in at.tabs])
print("[3] metricas:", [(m.label, m.value) for m in at.metric])
print("[4] dataframes:", len(at.dataframe),
      "->", at.dataframe[0].value.shape if at.dataframe else None)
print("[5] radio:", at.sidebar.radio[0].options, "-> seleccionado:", at.sidebar.radio[0].value)
print("[6] toggles:", [(t.label, t.value) for t in at.sidebar.toggle])
print("[7] sliders:", [(s.label, s.value) for s in at.sidebar.slider])
print("[8] errores:", [e.value for e in at.error])
assert not at.error, "la app muestra mensajes de error"
assert at.dataframe, "falta la tabla de coordenadas"
assert at.dataframe[0].value.shape[0] == 24, "la tabla debe tener 24 puntos"
print("[9] selectbox:", [s.value for s in at.sidebar.selectbox])
print("[10] botones de descarga:", [b.label for b in at.get("download_button")])

# --- otra rama: subir imagen ---
at.sidebar.radio[0].set_value("Subir imagen")
at.run()
print("[11] tras 'Subir imagen' -> excepciones:", at.exception)
assert not at.exception
print("[12] file_uploader presente:", len(at.get("file_uploader")) > 0)

# --- toggles de visualización al extremo ---
at.sidebar.radio[0].set_value("Usar imagen de muestra")
at.run()
at.sidebar.toggle[0].set_value(False)   # sin contorno
at.sidebar.toggle[1].set_value(True)    # con numeración
at.sidebar.slider[0].set_value(5)       # grosor máximo
at.sidebar.slider[1].set_value(1.7)     # puntos grandes
at.run()
print("[13] contorno OFF + números ON -> excepciones:", at.exception)
print("[14] metricas:", [(m.label, m.value) for m in at.metric])
assert not at.exception, "falló con contorno OFF + números ON"
assert at.metric, "no se muestran métricas"

print("[OK] Prueba E2E superada")
