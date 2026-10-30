# pilot2/ — PROTOCOLO CONGELADO 2026-10-28 (implementar bajo este texto)

Secuencial modular-frozen A→B→C→D pareado con el v2-pilot, con UNA
intervención: fase C a 800 generaciones. A/B/D en 400. Decide el
retorno a composición. Nada en este archivo cambia tras congelar.

## 1. Diseño

- Seeds 301, 302, 303, PAREADAS con el pilot original. Cada corrida
  arranca de su inicialización original (mismos streams RNG que el
  pilot → mismo theta inicial); NO continúa de pesos finales.
- Intervención única: C a 800 gens. Todo lo demás idéntico al pilot:
  arquitectura, fitness one-hot, ES v1 (32 pares, σ=0.1, α=0.05),
  n_fit=8, datos (train_meta_v1 / test_battery_v1), checks one-hot
  sampled cada 10 gens (record-only), slice Xavier fresco por fase,
  freeze por construcción.
- Fase C implementada como dos segmentos 400+400 del mismo runner del
  pilot con streams continuados (bit-exacto vs un run único de 800;
  probado en smoke). C@g400 es comparable bit a bit con post_C pilot.
- Evaluación: heldout pura 200 eps sampled (primaria) + greedy
  (secundaria), tras cada fin de fase para cada skill entrenada, más
  medición extra en C@g400 (skills A/B/C) para comparación.
- Checkpoints por seed: init, post_A, post_B, post_C400 (=C@g400),
  post_C (=C@g800, fin de fase C), post_D (=theta final).
- Sin gate, sin I1/H. Sin snapshots per-50.

## 2. Equivalencia con el pilot (verificada, no asumida)

- El entrenador pilot2 reutiliza el código del pilot (importa
  pilot_train: streams, fitness, runner de fase, scoring). Diferencia
  única: el mapa de generaciones por fase {A:400,B:400,C:800,D:400}.
- Smoke: pilot2 con todas las fases a 12 gens == pilot_train a 12 gens
  bit-exacto (thetas, checkpoints, histories, checks, heldouts).
- Post-hoc pre-registrado: slices A/B/D de pilot2.post_D ==
  pilot.post_D bit-exacto; pilot2.post_C400[rebanada C] ==
  pilot.post_C[rebanada C] bit-exacto; heldouts post_A/post_B
  idénticos. Cualquier diferencia inesperada se documenta (no se
  asume ni se silencia).

## 3. Puertas (congeladas)

1. Media over 3 seeds de heldout pura sampled por skill al final de
   su fase ≥ 0.80 (A,B,D a g400; C a g800). CI descriptivo.
2. Caída media de retención (fin de fase → post_D) ≤ 0.10 por skill
   previamente adquirida (media ≥ 0.80).
3. Greedy sólo secundaria/descriptiva.

## 4. Integridad vs rendimiento (doctrina congelada)

- Los parámetros congelados permanecen idénticos bit a bit (asertado
  en código entre checkpoints).
- Retención exactamente 0.0 es lo esperable al repetir idéntica
  evaluación, incluidas sus semillas.
- Una diferencia de rendimiento exige diagnosticar evaluación,
  aleatoriedad y freeze; NO se etiqueta automáticamente como bug ni
  se modifica la puerta ≤0.10. Procedimiento: (a) re-evaluar local
  con código frozen el checkpoint sospechoso; (b) comparar
  slice a slice contra el artefacto pareado; (c) dictaminar con
  evidencia antes de usar el número.

## 5. Presupuestos (derivados del código, convención pilot/diag)

Fórmula por fase (runner frozen): gens×64×8×200 + (gens//10)×40×200.
- A/B/D: 400×64×8×200 + 40×40×200 = 41,280,000 c/u.
- C: 800×64×8×200 + 80×40×200 = 82,560,000.
- Total entreno/seed: 206,400,000 (asertado). Batch 3 seeds: 619.2M.
- Medición heldout fuera de presupuesto (convención v1).

## 6. Archivos esperados (9)

Por seed S ∈ {301,302,303} en /vol/pilot2/runs (volumen
tiny-learner-pilot2, separado; no toca pilot/C6):
- `P2_S.json`: config, fases (history/checks: A/B/D 400/40, C 800/80),
  heldout_after (post_A/B/C400/C/D), retention_sampled, steps, wall.
- `P2_S_theta.npz`: theta final post_D (7944).
- `P2_S_ckpts.npz`: init, post_A, post_B, post_C400, post_C, post_D.

## 7. Lanzamiento (Terminal host, Modal 1.6.2, requiere GO)

`python3.11 -m modal run --detach pilot2/modal_pilot2.py`

Salida esperada (varía fc-…):
`creating ONE pilot2 batch call with 3 input(s) via experimental_spawn_map: [301, 302, 303]`
`pilot2 batch call created: <fc-…> (3 inputs running detached; results land in /vol/pilot2/runs on volume tiny-learner-pilot2)`

## 8. Análisis pre-registrado (pilot2/score_pilot2.py)

Integridad (9/9, config, steps, freeze bit-exacto) → gates §3 →
comparación pareada vs pilot_runs (deltas + checks bit-exactos §2) →
veredicto GO/NO-GO a composición. Coste ≈$0.75–1.30, wall ≈2.5–3.5h.

## Congelación

- Protocolo: FROZEN 2026-10-28 (este texto).
- pilot2_train.py sha256[12]: 45ffb14f90df (2026-10-28, SMOKE PASS)
- modal_pilot2.py sha256[12]: 319120b2b58c (2026-10-28, SMOKE PASS)
- Smoke: PASS local 2026-10-28 (T1 equivalencia bit-exacta con pilot;
  T2 segmentos 10+10 == run único; T3 captura mid-C; T4 aritmética
  206.4M). Incidencias del propio test, no del entrenador: segmentos
  desalineados con cadencia de checks (fijado con 10+10 y assert
  C_MID%10==0) y nombre de clave sin zero-pad (fijado). Salidas en
  /tmp/pilot2_smoke (desechables).
