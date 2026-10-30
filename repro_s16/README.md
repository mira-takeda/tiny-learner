# Repro S16: A01 vs A/B solas en ACQ18

Comandos, desde la raíz del repo:

    python3.14 -m venv .venv
    .venv/bin/pip install -r repro_s16/requirements.txt
    .venv/bin/python repro_s16/repro.py

Verifica 10 hashes (escenarios, pesos, código), ejecuta 3600
episodios en CPU (A01 + A-sola + B-sola, 400×3 thetas cada
brazo) e imprime llegadas, colisiones, timeouts y retorno
medio por theta y media. ~58 s en esta máquina (Intel Mac).
Solo necesita numpy + stdlib; sin API ni volúmenes privados.

Salida comprobada (2026-10-30, también desde copia limpia con
venv nuevo, 60.1 s): idéntica a s15_eval.json (A01) y
s16_eval.json (A/B) — cuentas exactas, retorno a 1e-3.
Global: A01 0.6467, A 0.1450, B 0.2458.

## Ejecución real vs muestra grabada

Todo número impreso se computa en este proceso ejecutando el
entorno Tiny congelado con semillas deterministas (ver
MANIFEST.json y el banner REAL LOCAL EXECUTION). No es una
muestra grabada de participación de Meta ni una respuesta
del asistente: por CONTEXT.md §2, el aprendizaje y la
evaluación ocurren en el sistema pequeño; Meta/Muse solo
actúan como diseñador/maestro/evaluador fuera del episodio.
Ninguna llamada por episodio sale de esta máquina.
