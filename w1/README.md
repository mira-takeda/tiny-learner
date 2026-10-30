# w1/ — follow-up rediseñado post-I1 (GO 2026-10-30, Windows host)

W1 ensancha G1 de forma procedural y general desde la definición
de la tarea (corrección metodológica: ningún número derivado de
I1). I1 retirado permanentemente. Recetas verbatim N1b + N2; el
ÚNICO cambio es la distribución G1. Spec congelado: PREREG_W1.md.

## Generador wide-G1 (congelado, sha en MANIFEST)

Espejo de gen.sample_config 'guarded'; cambian sólo dos cotas,
ambas de constantes registradas (PREREG_W1 §2):
agent-haz [35,70] (unión bandas spawn familia B pursuit),
beacon-haz >= MIN_SEP=10 sin cota superior, speed [3.0,3.5]
(soporte G1), sin restricción angular, sin regiones.
Mismas físicas/obs/rewards; Tiny original 7944 params.

## Congelación y pools (2026-10-30)

- VC==P3 bit-exacto 4/4 x3 re-verificado; pins externos
  coinciden con registros previos (train_meta, n1_demos, VC).
- Pools: demo 18 / espool 12 / gate-G1 200 / gate-G2 200 /
  sing-gate 80 (bloque semillas 21100-21107, calendario regla
  N2). Disjunción: 0 colisiones vs 10648 entradas previas
  (19 ficheros) + interna 0.
  w1_pools.json 6050b153...1e0856c (completa en RUNLOG).
- Cobertura descriptiva pre-entreno: gate-G1 d_gh
  [10.84,101.19] med 48.15; ang [0.0,3.13] med 1.06,
  sextiles [56,42,37,29,17,19] (todos no vacíos).
  Cercana→lejana + rango angular completo confirmados.

## Entrenamiento (receta respaldada, 3 thetas 301-303)

- Colecta: 54/54 planner P-AB en wide-G1 (105 s de 1800 s),
  52 éxitos + 2 fallos incluidos por protocolo.
  w1_demos_g1.json 31200964...
- Imitación N1b-verbatim (54 wide-G1 + 54 G2 reutilizadas +
  singles reutilizados), 119 s: eval acc 0.769/0.778/0.849
  vs const ~0.34; D-mass 220/233/200 > 0 (assert pasado).
  Clases A/B/C presentes en demos (B 642-683 train).
  w1_imit_models.json 6f763e21... / results 011437f1...
- ES-duro N2-verbatim (8 pares, s=.1, a=.01, 25 gens,
  rank-tied, ventanas 2G1+2G2+ABCD), 266.5 s de 2400 s,
  3x25 gens completos, 0 incompletos, skills P3 intactas
  (shas pre+post). w1_es_models.json d7710053...

## Puerta fresca (apertura única, 2640 eps, 41 s) — FAIL

Brazos {W1-final, W1-imit} x {gate-G1, gate-G2} x 3 th.
Barra congelada: media W1-final >= 0.70 en CADA familia.

- G1: final 0.543 (0.56/0.51/0.56) vs imit 0.508 -> FAIL.
- G2: final 0.678 (0.745/0.755/0.535) vs imit 0.608 -> FAIL.
- Singles spot (final): A 59/60, B 45/60, C 38/60, D 60/60.
- w1_gate.json 169d2930... Veredicto: GATE FAIL.

ES mejora sobre imitación en ambas familias frescas
(+3.5pp G1, +7.0pp G2) pero el coordinador conjunto no
alcanza nivel de funcionamiento en la distribución wide.
G2-th303 bajo ya en imit (0.455) -> arrastre de init/datos,
no causado por ES. La puerta mide funcionamiento, NO
transferencia: sin claim de transferencia posible.

## STOP — H sellado

Puerta FAIL -> STOP por protocolo: ningún retune, H1/H2/H3
NO se abren. w1_h.py jamás ejecutado; cero episodios H
corridos; H intacto como test sellado para futuras líneas
(sólo cargas mecánicas de dup-check, patrón N-series).
Línea W1 cerrada salvo nuevo diseño autorizado por el usuario.

## Archivos

PREREG_W1.md (spec), gen_wide.py, w1_collect.py,
w1_imitate.py, w1_es.py, w1_gate.py, w1_h.py (congelados,
MANIFEST), w1_freeze.json, w1_pools.json, w1_coverage.json,
w1_demos_g1.json, w1_imit_{models,results}.json,
w1_es_{models,state}.json, w1_gate.json, budgets.
Este README es registro (no cubierto por MANIFEST).
