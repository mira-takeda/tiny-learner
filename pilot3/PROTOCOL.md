# pilot3/ — PROTOCOLO CONGELADO 2026-10-28 (implementar bajo este texto)

Extensión pareada a C=1200 gens tras pilot2 NO-GO (C=0.7933). Esta
propuesta autoriza diseñar UNA extensión hasta 1200; no autoriza
aumentos posteriores. Si falla: conservar NO-GO y diagnosticar antes
de proponer otra corrida. Nada en este archivo cambia tras congelar.

## Corrección registrada

Retirado "misma seed débil" (pilot2): C6 usa seeds 501–503 y pilot2
usa 301–303. Coincidencias de puntuación entre campañas con seeds
distintas no establecen correspondencia.

## 1. Diseño (congelado)

- Seeds 301, 302, 303, pareadas con pilot/pilot2 desde la
  inicialización original (mismos streams; no continuar de pesos).
- Único cambio vs pilot2: fase C a 1200 generaciones
  (segmentos 400+400+400 con streams continuados, cada frontera
  múltiplo de 10); A/B/D en 400. Tope de esta extensión: 1200.
- Todo lo demás idéntico: arquitectura, fitness one-hot, ES v1,
  n_fit=8, datos, checks, freeze, heldout pura sampled (primaria)
  + greedy (secundaria).
- Checkpoints C: g400, g800 (=fin de pilot2), g1200 (=fin de fase).
  Claves: post_C400, post_C800, post_C; más init, post_A, post_B,
  post_D. Heldout pura en cada captura para A/B/C.
- Las capturas mid-C (copia de theta + scoring con semillas de
  batería) NO consumen ni alteran los streams RNG de entrenamiento;
  verificado en smoke con el recorrido completo del entrenador.

## 2. Equivalencia (dos niveles, no confundir)

- Nivel smoke (reducido, pre-lanzamiento): pilot3 == pilot/pilot2 a
  gens cortos bit-exacto, CON capturas activas; segmentación
  10+10+10 == run único; streams RNG intactos tras capturas.
- Nivel producción (post-hoc vs pilot2_runs): slices A/B/D de
  post_D bit-exactos; rebanada C en g400 y g800 bit-exacta;
  heldouts post_A/post_B/post_C400/post_C(=g800 pilot2) idénticos.
  Desvíos → triage pilot2 §4 (diagnosticar; sin auto-bug).

## 3. Puertas originales (intactas)

1. Media over 3 seeds de heldout pura sampled por skill al final de
   su fase ≥ 0.80 (C a g1200). CI descriptivo.
2. Caída media de retención ≤ 0.10 (esperado exactamente 0.0).
3. Greedy secundaria. GO sólo si TODAS pasan.

## 4. Lectura de tendencia (primaria: Δ=g1200−g800, pura sampled)

- Plateau si |Δ| ≤ 0.07 en ≥2/3 seeds.
- Todavía aprendiendo si Δ > +0.07 en ≥2/3 seeds.
- Resto: INCONCLUSIVE.
- Base: pilot2 Δ(g800−g400) +0.070/+0.225/+0.100 (>+0.07 en 2/3).

## 5. Presupuesto y artefactos (verificados en código/smoke)

- C1200: 1200×64×8×200 + 120×40×200 = 123,840,000 steps.
- Total/seed: 3×41.28M + 123.84M = 247,680,000 (asertado).
- 9 archivos: `P3_{301,302,303}.{json,theta.npz,ckpts.npz}` en
  volumen separado tiny-learner-pilot3, ruta /vol/pilot3/runs.
  JSON: fases (A/B/D 400/40; C 1200/120), heldout_after
  (post_A/B/C400/C800/C/D), retention, steps, wall.

## 6. Decisiones de §6 (cerradas)

Seeds 301–303 pareadas; C=1200/A/B/D=400; puertas §3 intactas;
regla Δ §4; tope 1200; comando Modal 1.6.2
`python3.11 -m modal run --detach pilot3/modal_pilot3.py` con salida
`creating ONE pilot3 batch call with 3 input(s) via
experimental_spawn_map: [301, 302, 303]` y results en
/vol/pilot3/runs on volume tiny-learner-pilot3. Coste ≈$0.90–1.55,
wall ≈3–4.5h concurrentes. Análisis: pilot3/score_pilot3.py.

## 7. Riesgo (interpretación corregida)

Las puertas congeladas determinan GO/NO-GO, pero NO eliminan el
ruido ni sustituyen reportar resultados por seed e incertidumbre.
Estando C en el borde (~0.80), el veredicto coexiste con el reporte
completo; ninguna relectura post-hoc altera puertas ni veredicto.

## Congelación

- Protocolo: FROZEN 2026-10-28 (este texto).
- pilot3_train.py sha256[12]: 396ef4969d7b (2026-10-28, SMOKE PASS)
- modal_pilot3.py sha256[12]: 4120dc1074ce (2026-10-28, SMOKE PASS)
- Smoke: PASS local 2026-10-28 (T0 espejo bit-exacto vs pilot2;
  T1 capturas sin perturbar el run completo; T2 10+10+10 == único;
  T3 streams RNG intactos; T4 aritmética 247.68M). Incidencia del
  propio test: T3 comparaba mal la forma del dict de estado RNG
  (KeyError, no movimiento de streams); fijado y T3/T4 re-pasados
  sobre código de producción intacto. Salidas en /tmp/pilot3_smoke
  (desechables). Esto es equivalencia de smoke reducido; la
  equivalencia de producción hasta g800 se comprueba post-hoc vs
  pilot2_runs.
