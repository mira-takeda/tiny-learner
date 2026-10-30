# pilot/ — v2-pilot secuencial (v1 CLOSED, diag/ untouched)

Pregunta única: ¿la corrección fitness one-hot (C4/B2) funciona también en
el contexto secuencial real modular-frozen, con co-skills entrenadas y
congeladas? Este pilot decide si se puede volver a la pregunta de
composición. No hay matriz v2 todavía.

## Diseño final (congelado)

- Seeds: 301, 302, 303 (streams frescos, sin solape con v1 ni diag).
- Arquitectura: modular frozen, idéntica a v1 (4×1833 + gate 612 = 7944).
- Secuencia: A → B → C → D, congelando cada skill tras su fase (como A1).
- Único cambio vs v1: el fitness de cada fase atómica usa rollouts
  one-hot puros de la skill de la fase (skill_only=k). Todo lo demás es
  mecánica v1-A1: entorno, obs, acciones, rewards, ES (32 pares,
  σ=0.1, α=0.05), 400 gens/fase, n_fit=8, checks one-hot sampled cada
  10 gens (record-only), slice Xavier fresco por fase, train_meta_v1.
- Evaluación: heldout 200 eps sampled (primaria, comparable v1) +
  greedy (descriptiva secundaria), skill pura, tras cada fase para cada
  skill entrenada hasta ese punto.
- Sin shaping C3, sin n_fit=32, sin greedy training, sin gate, sin I1/H.
- Artefactos por seed (9 total): `{seed}.json`, `{seed}_theta.npz`
  (post-D), `{seed}_ckpts.npz` (init, post_A/B/C/D).
- Presupuesto lógico entreno: 4×41,280,000 = 165,120,000 steps/seed
  (asertado en código). Medición heldout (~0.8M steps/seed) separada,
  no entra en el presupuesto (misma convención que v1).
- Ficheros: `pilot_train.py` (entrenador), `modal_pilot.py` (launcher),
  `smoke_pilot.py` (smoke local). Volumen `tiny-learner-pilot`,
  resultados en `/vol/pilot/runs`.

## Criterio de decisión (CONGELADO antes del lanzamiento)

GO (avanzar a composición) si y sólo si:
1. Para cada skill S ∈ {A,B,C,D}: media over 3 seeds del heldout
   sampled puro al final de su propia fase ≥ 0.80.
2. Para cada skill previamente adquirida: caída media de retención
   (fin de su fase → post_D) ≤ 0.10.

Notas de consistencia v1 (fijadas ANTES de lanzar, a petición):
- (a) La puerta es la MEDIA over seeds (práctica aplicada en v1); el
  95% CI se reporta como descriptivo. Con n=3 el CI-t es muy ancho y
  pedirlo sobre el límite inferior exigiría medias ≈1.00, lo cual no
  es lo que v1 aplicó. v1 usó 5 seeds; el pilot usa 3 (mismo
  estadístico, menos seeds).
- (b) Con pesos congelados y evaluación determinista, la caída de
  retención es exactamente 0.00 por construcción: la regla ≤0.10 es un
  chequeo de maquinaria, no un test empírico. Cualquier valor distinto
  de cero invalida el run (bug) y obliga a investigar antes de usar
  los resultados. El código además aserta igualdad bit-exacta de los
  slices congelados entre checkpoints.
- (c) "Previamente adquirida" = media de fin de fase ≥ 0.80. Si una
  skill no adquiere, la puerta 1 ya falla (NO-GO); la puerta 2 no la
  rescata.
- (d) "Fin de su fase" = heldout sampled puro de 200 eps, misma
  definición que la batería de adquisición de v1.

## Steps / coste / wall estimados

- Entreno: 3 × 165.12M = 495.36M steps lógicos (exactos, asertados).
- Medición heldout: ≈2.4M steps total (20 baterías×200 eps×2 modos
  por seed, despreciable, fuera de presupuesto como en v1).
- Coste: ≈$0.60–1.05 escalando del diag (1.36B steps → $1.65–2.89).
  Misma salvedad: facturación CPU de Modal sin confirmar.
- Wall: ≈2–4h concurrentes (3 seeds en paralelo; decide la más lenta).
  A/D por fase sin dato diag: estimación con incertidumbre.

## Lanzamiento (Terminal del host, requiere GO explícito)

Desde `/Users/satoru/Documents/hackathon/tiny-learner`:

`python3 -m modal run pilot/modal_pilot.py --detach`

Salida que confirma el batch correcto (el id `fc-…` varía):

```
creating ONE v2-pilot batch call with 3 input(s) via experimental_spawn_map: [301, 302, 303]
v2-pilot batch call created: <fc-…> (3 inputs running detached; results land in /vol/pilot/runs on volume tiny-learner-pilot)
```

Verificar: "3 input(s)", "[301, 302, 303]", "/vol/pilot/runs",
"tiny-learner-pilot".

## Congelación

- pilot_train.py sha256[12]: af0c99136139 (2026-10-28, SMOKE PASS)
- modal_pilot.py sha256[12]: 1d3d4cbfdb53 (2026-10-28, SMOKE PASS)
- Smoke: PASS local 2026-10-28 (equiv bit-exacta vs frozen, fitness
  one-hot verificado vs rollouts manuales, e2e 12 gens×A-D con steps
  exactos, freeze bit-exacto, retención 0.0). Salidas en /tmp/pilot_smoke
  (desechables). Nota: el smoke falló una vez por bug del propio test
  (4 slices idénticos por seed repetida), no del entrenador; corregido
  y re-pasado.
