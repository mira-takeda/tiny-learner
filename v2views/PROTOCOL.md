# v2views/ — PROTOCOLO VISTAS-ON vs v2 CONGELADO 2026-10-28

Contraste pareado del contrato de vistas por skill. Único factor:
qué slots observa cada skill. Nada en este archivo cambia tras congelar.

## Diseño (propuesto)

- Arranque: post_D pilot3, seeds 301–303, sin selección. Skills
  congeladas, sin reentreno. Gate desde ceros originales (NO el gate
  final de v2). Mismos streams [seed,4,purpose], ES v1, 400 gens,
  fitness 4×G1+4×G2, checks record-only, presupuesto 41.28M/seed.
- Sin cambios de arquitectura, recompensa, escenarios de entreno
  ni presupuesto. Volumen y runs separados; v2_runs/ intacto.

## 1. Máscaras exactas (índices obs 14-dim)

Base: vel[0:2] siempre; S1 goal[2:5]; S2 hazard[5:8]; S3 obj[8:11];
S4 zone[11:14]. Ausencia = ceros exactos (codificación de entreno).
- Skill A: conserva [0:5]; cero en [5:14].
- Skill B: conserva [0:2]+[5:8]; cero en [2:5]+[8:14].
- Skill C: conserva [0:2]+[8:14]; cero en [2:8].
- Skill D: conserva [0:2]+[2:5]; cero en [5:14].
- Gate: [0:14] completo, sin máscara.
Vistas ante CADA forward de skill (fitness, checks, eval, controles
de mezcla); mundo intacto. Hazard-mask global (S2→0 pre-vistas) sin
cambiar de definición. Single-skill refs con su vista puesta.

## 2. Smoke pre-run (verificaciones)

- Vistas intactas en tarea atómica: por skill, batería atómica
  frozen completa (200 eps), forward con/sin vista bit-idéntico
  en CADA paso (los slots enmascarados ya son cero allí).
- Identidad fitness/eval (patrón v2 T0) bajo vistas; freeze
  bit-exacto; RNGs de entreno intactos ante evaluaciones.
- Aritmética 41.28M, seeds, constantes, hashes (heldout G1/G2 +
  batería nueva §4).

## 3. Comparación pareada (Q1/Q2 y umbrales intactos)

- Mismas seeds/streams/init ⇒ diferencias sólo por vistas.
- Q1: I1≥0.70 (si no, H UNINTERPRETABLE), H≥0.60 c/u,
  H-media≥ref+0.40 (ref = mejor skill sola CON vista, regla v2).
- Q2: gate−control en H-media ≥+0.10 (uniforme, shuffle sobre
  skills con vista); per-batería descriptivo; SIN-VENTAJA = no
  alcanzar el margen (no equivalencia).
- GO pilot ⇔ Q1 total PASS en la batería NUEVA. Q2 modula
  afirmación. Monolito sigue pendiente.

## 4. Evaluación nueva + referencia v2 (fijada pre-run)

- Batería nueva I1/H1/H2/H3 (mismas distribuciones objetivo,
  generador frozen, streams frescos SeedSequence(9000–9003),
  200 eps c/u, hasheada ANTES del entreno; construcción exacta al
  implementar). Cero tuning/selección con ella.
- Referencia comparable: gates finales v2 ORIGINALES evaluados en
  esa misma batería nueva (sólo medición). Deltas pareados
  vistas-ON − v2 por seed/batería (descriptivos + mismas reglas).
- G1/G2-heldout existente reutilizado (informativo, mismo archivo).

## 5. Baterías antiguas como diagnóstico

Resultados en I1/H1/H2/H3 frozen originales: descriptivos de
desarrollo, SEPARADOS del veredicto (batería nueva). No deciden GO.

## Alcance causal (congelado en redacción)

Todas las máscaras juntas = UNA intervención de diseño. Su
resultado NO identifica qué máscara adicional aporta o perjudica;
sólo B–S1 tiene respaldo previo (D4+D5). Atribución por máscara
requeriría ablaciones separadas, no autorizadas aquí.

## Riesgos registrados (corrige "ninguno")

- El currículo G1/G2 podría seguir permitiendo estrategias que no
  requieran B.
- Sólo B–S1 tiene respaldo causal previo; el resto viaja sin probar.
- Preservar las atómicas no garantiza coordinar sus salidas en
  composiciones.
- Batería fresca y pareado controlan aspectos del experimento, pero
  no eliminan incertidumbre.

## Lanzamiento y análisis (congelado)

- Volumen tiny-learner-v2views, ruta /vol/v2views/runs. 9 archivos:
  `VV_{301,302,303}.{json,theta.npz,ckpts.npz}`.
- Comando (host, Modal 1.6.2, Python 3.11):
  `python3.11 -m modal run --detach v2views/modal_vviews.py`
- Salida: `creating ONE v2views batch call with 3 input(s) via
  experimental_spawn_map: [301, 302, 303]` + results en
  /vol/v2views/runs on volume tiny-learner-v2views.
- Coste ≈$0.15–0.30, wall ≈1–2h concurrentes.
- Análisis: v2views/score_vviews.py (integridad → batería nueva
  Q1/Q2 → deltas vs referencia v2 → diagnóstico viejo → veredicto).

## Congelación

- Protocolo: FROZEN 2026-10-28 (este texto).
- Batería nueva: v2views/eval_new_v1.json sha256
  2fd9a68c0d26f884a23eea2aad8e77a6cd27ec1d71cc00bfa41b063c8cce51e9
  (I1N/H1N/H2N/H3N 200 c/u, SeedSequence(9000-9003), 0 colisiones de
  semilla/cfg con test_battery_v1.json; generada y hasheada ANTES de
  entrenar, sin inspección de resultados)
- views_train.py sha256[12]: 12264471fd76 (2026-10-28, SMOKE PASS)
- modal_vviews.py sha256[12]: fcce4830ac82 (2026-10-28, SMOKE PASS)
- Smoke: PASS local 2026-10-28 (T0 vistas intactas 60168 pasos;
  T1 sin mutación; T2 identidad fitness/eval; T2b shuffle frozen;
  T3 run 12 gens; T4 RNG intactos; T5 constantes). Incidencia real
  del entrenador, cazada por T1: mats desempaquetadas por path
  equivocado (mlp_forward vs _forward_mats); fijado espejando
  ModularNet.probs frozen + T2b de equivalencia. Salidas en
  /tmp/vviews_smoke (desechables).
