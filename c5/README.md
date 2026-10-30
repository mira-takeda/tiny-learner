# c5/ — C5: interacción C4+C2 en C aislada (diag/ y pilot/ untouched)

Siguiente experimento mínimo tras el NO-GO del v2-pilot (bloquea C).
Prueba ÚNICAMENTE la interacción de las dos intervenciones que
funcionaron: fitness one-hot (C4) + entrenamiento greedy (C2).

## Diseño final (congelado)

- Familia C aislada, arquitectura modular, seeds frescas 401–403.
- 400 generaciones, n_fit=8, ES v1 (32 pares, σ=0.1, α=0.05).
- Fitness one-hot (skill 2) + rollouts greedy (fitness y checks).
- Sin shaping, sin presupuesto extra: 41.28M steps/job (asertado).
- Entrenador: `diag_train.run_job` SIN MODIFICAR (mismo code path que
  produjo C2/C4) + job C5 = C0 + fit1hot + greedy.
- Medición (post-entreno, sin tocar theta): heldout 200 eps blended
  (heredado del diag, informativo) + heldout 200 eps SKILL PURA
  sampled (primaria) y greedy (secundaria), con código/baterías frozen.
- Artefactos por seed (9 total): `C5_{seed}.json` (incluye
  `heldout_pure`), `C5_{seed}_theta.npz`, `C5_{seed}_snaps.npz`.
- Ficheros: `c5_train.py`, `modal_c5.py`, `smoke_c5.py`.
  Volumen `tiny-learner-c5`, resultados en `/vol/c5/runs`.

## Criterio de decisión (CONGELADO antes del lanzamiento)

GO (volver a diseñar v2 composicional) si y sólo si la media over
3 seeds del heldout sampled de skill pura ≥ 0.80. NO-GO si queda por
debajo. Sin otra lectura posible: una métrica, un listón.

Aclaración pre-lanzamiento (fijada AQUÍ, antes de ver resultados): la
métrica primaria es `heldout_pure.sampled`, NO el `heldout` blended
del JSON. Precedente C4: blended sampled 0.165 con skill pura en
0.735 — puntuar sobre blended haría el listón inalcanzable por
construcción y rompería la comparabilidad con C4/C2/pilot, todos
puntuados en skill pura. El blended se conserva en el JSON sólo como
informativo comparable al diag crudo.

## Steps / coste / wall estimados

- Entreno: 3 × 41.28M = 123.84M steps lógicos (exactos, asertados).
- Medición: 4 baterías heldout × 200 eps × 2 (blended+pure) por seed,
  despreciable y fuera de presupuesto (convención v1/diag/pilot).
- Coste: ≈$0.15–0.25 escalando del diag (1.36B steps → $1.65–2.89).
  Misma salvedad: facturación CPU de Modal sin confirmar.
- Wall: ≈0.5–1h concurrentes (rango C4–C2 por job).

## Lanzamiento (Terminal del host, requiere GO explícito)

Desde `/Users/satoru/Documents/hackathon/tiny-learner`:

`python3 -m modal run c5/modal_c5.py --detach`

Salida que confirma el batch correcto (el id `fc-…` varía):

```
creating ONE C5 batch call with 3 input(s) via experimental_spawn_map: [401, 402, 403]
C5 batch call created: <fc-…> (3 inputs running detached; results land in /vol/c5/runs on volume tiny-learner-c5)
```

Verificar: "3 input(s)", "[401, 402, 403]", "/vol/c5/runs",
"tiny-learner-c5".

## Congelación

- c5_train.py sha256[12]: d83dac81002a (2026-10-28, SMOKE PASS)
- modal_c5.py sha256[12]: af52e2d531da (2026-10-28, SMOKE PASS)
- Smoke: PASS local 2026-10-28 (config exacta, manipulación greedy
  verificada viva, e2e 12 gens con steps exactos + heldouts
  blended/pure). Salidas en /tmp/c5_smoke (desechables).
