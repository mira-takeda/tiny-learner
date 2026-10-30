# v2conf/ — PROTOCOLO ADQUISICIÓN-BAJO-CONFLICTO CONGELADO 2026-10-28

Contraste pareado: gate con vistas + conflicto en entreno. Nada en este
archivo cambia tras congelar. Veredictos v2/v2views intactos.

## Diseño (congelado)

- Arranque post_D pilot3, seeds 301–303, sin selección. Skills
  congeladas bit-exacto, vistas v2views idénticas (A [0:5], B
  [0:2]+[5:8], C [0:2]+[8:14], D [0:2]+[2:5]; gate completo).
- Gate desde ceros, streams [seed,4,purpose], ES v1, 400 gens,
  n_fit=8. Fitness/generación: 2×G1-orig + 2×G1-conflicto + 4×G2
  (orden fijo). Checks record-only idénticos a v2 (20 G1-orig +
  20 G2; adquisición-conflicto se mide post-hoc, no en checks).
- G1-conflicto: mismas reglas G1a/G1b/G1adv (200/140/60) cambiando
  SÓLO db a banda 12–25. Set separado, semillas SeedSequence(7000),
  hasheado. Cero episodios compartidos con evaluación.
- Sin cambios de recompensa, arquitectura ni presupuesto (41.28M).

## Pregunta primaria y veredictos

- Adquisición: media sampled ≥0.70 en batería ACQ nueva (200 eps
  distribución conflicto, fijada pre-run) ⇒ PASS; si no ⇒ FAIL y H
  UNINTERPRETABLE (regla análoga a v2: sin adquisición no hay
  lectura de transferencia).
- H (H1C/H2C/H3C nuevas, mismas distribuciones objetivo): cada una
  ≥0.60; H-media ≥ ref+0.40 (ref = mejor skill con vista; check
  1.00−ref ≥ 0.40 reportado); Q2 +0.10 vs uniforme/shuffle.
- SIN-VENTAJA = no alcanzar margen (no equivalencia). Monolito
  pendiente. Resultados viejos sólo diagnóstico de desarrollo.

## Aportación de B (medida separada)

- Pesos por distancia en ACQ: descriptivos; modular NO es requisito.
- Ablación pareada: SÓLO q_B→uniforme(1/9), pesos y demás intactos;
  primaria en ACQ (+ H descriptivo). Caída media (normal−ablación)
  ≥0.10 ⇒ aportación práctica (criterio NUEVO; no prueba necesidad
  absoluta ni suficiencia).

## Evaluación (fijada y hasheada pre-run)

- ACQ: normal, greedy, ablación-B, skill-0..3 (estas últimas
  diagnóstico: ¿resuelve alguna sola el conflicto?).
- H*C: set veredicto completo (normal, greedy, uniform, shuffle,
  mask, skill-0..3). G1H/G2H: normal informativo.
- Referencia: gates v2views ORIGINALES en las MISMAS baterías
  nuevas (medición, sin reentreno); deltas pareados descriptivos.

## Lanzamiento y análisis

- Volumen tiny-learner-v2conf, /vol/v2conf/runs. 9 archivos:
  `VC_{301,302,303}.{json,theta.npz,ckpts.npz}` (ckpts init/post_gate).
- Comando (host, Modal 1.6.2, Python 3.11):
  `python3.11 -m modal run --detach v2conf/modal_vconf.py`
- Salida: `creating ONE v2conf batch call with 3 input(s) via
  experimental_spawn_map: [301, 302, 303]` + results en
  /vol/v2conf/runs on volume tiny-learner-v2conf.
- Coste ≈$0.15–0.35, wall ≈1–2.5h concurrentes.
- Análisis: v2conf/score_vconf.py.

## Congelación

- Protocolo: FROZEN 2026-10-28 (este texto).
- train_conflict_v1.json sha256
  37c2524a2afbef40b67695032564fcbc8d1e5633f47d573a98e62770d27932e2
  (400 G1-conflicto G1a200/G1b140/G1adv60, SeedSequence(7000), 0 dups)
- eval_conf_v1.json sha256
  feba0472c4d217b865e8c42ae81f2fab48f9d747819dba5faca2eda98185c553
  (ACQ200 + H1C/H2C/H3C 200 c/u, SeedSequence(9500-9503), 0 dups;
  ACQ sin episodios compartidos con train_conflict)
- conf_train.py sha256[12]: 0e04e65f9d1f (2026-10-28, SMOKE PASS)
- modal_vconf.py sha256[12]: 704c60019b6c (2026-10-28, SMOKE PASS)
- Smoke: PASS local 2026-10-28 (T0 sets válidos 600 cfgs + hashes;
  T1 pools 2+2+4 en orden, 128 evals instrumentados; T2 ablación
  = normal con sólo B→uniforme; T3 run 12 gens + claves +
  weights==run_battery; T4 sin mutación, RNG intactos; T5 41.28M y
  constantes). Salidas en /tmp/vconf_smoke (desechables).
