# v2/ — PROTOCOLO COMPOSICIONAL CONGELADO 2026-10-28

Pilot mínimo de composición. Nada en este archivo cambia tras congelar.

## Correcciones registradas

- pilot3: |Δ|≤0.07 = plateau OPERATIVO; no prueba ruido, ausencia de
  aprendizaje ni pendiente agotada. C g800→g1200: +0.05 media, causa
  no demostrada. Presupuesto cerrado por tope.
- pilot2: retirado "misma seed débil" (seeds distintas por campaña).

## Demostrado vs pendiente

- Demostrado: adquisición (A 1.00, B 0.83, C 0.84, D 1.00) y
  conservación (retención 0.0, freeze bit-exacto).
- Pendiente: TODO lo composicional (Q1 y Q2).

## Dos preguntas separadas

- Q1: ¿resuelve el sistema combinaciones no entrenadas?
- Q2: ¿aporta el gate aprendido frente a controles? Superar una skill
  sola NO demuestra coordinación (una mezcla uniforme podría superarla).

## 1. Arranque, gate, mezcla (frozen)

- Arranque: post_D de pilot3_runs (P3_301/302/303_theta.npz), seeds
  301–303, SIN selección. Gate a ceros exactos verificado (= mezcla
  uniforme inicial). Proveniencia verificada post-hoc bit a bit.
- Gate: obs 14-dim (sin task ID) → MLP 14→32→4 + softmax; acción por
  mezcla Σ w_k·softmax(skill_k). Muestreo primario, argmax secundario.

## 2. Entreno: escenarios, fitness, identidad, freeze

- Datos frozen train_meta_v1 (G1/G2 400+400). Fitness ES: media 4×G1
  + 4×G2, recompensas tasks.py intactas. Sólo slice gate (612),
  skills congeladas. ES v1, 400 gens, n_fit 8, checks record-only
  media(20 G1 + 20 G2) cada 10 gens. Streams frescos [seed,4,purpose]
  (convención gate de v1; sin colisión con pilots).
- IDENTIDAD fitness/evaluación (requisito): misma mezcla y política
  sampled en ambos (blend completo + rng.choice sobre modular_probs),
  un solo code path o igualdad por episodio probada en smoke.
- Skills bit-congeladas pre/post (assert; cambio = run inválido).
- Evaluaciones usan sólo semillas de batería; no tocan streams de
  entreno (probado en smoke por estados RNG + equivalencia).

## 3. Éxitos, H3, imposibles, factibilidad

- I1: beacon <6 sin contacto. H1/H2: delivery/fetch-delivery sin
  contacto (haz 4.5). H3: delivery por gap 10–12. Baterías frozen
  íntegras (200/200), sin exclusiones post-hoc.
- H3 PASS = evidencia de transferencia hacia una combinación que
  incluye D (habilidad no practicada en G1/G2). Cuenta en el
  veredicto global como las demás (§7).
- Factibilidad por diseño (rangos/separaciones), no probada por
  episodio. Oráculos diag/ sólo diagnósticos: su fallo NO demuestra
  imposibilidad. Cero exclusiones por rendimiento.

## 4. Controles (fijados pre-run)

- Uniforme (w=1/4), shuffle (gate aleatorio, semilla frozen 0),
  mejor skill sola, hazard-mask. Definidos en código frozen (hash en
  congelación); sin ajustes post-hoc.
- "Mejor skill sola": 4 skills frozen 1-hot por H (sampled, 200 eps);
  por batería, la mayor MEDIA entre seeds; referencia H-media =
  promedio de las tres máximas. Sin selección por episodio ni seed.

## 5. Monolito y presupuestos

- Pilot SIN brazo monolito (A3-v1 no adquirió C). Este pilot NO
  prueba superioridad vs monolito aunque todo pase; espera a v2
  completa (igual presupuesto/fase, 41.28M).
- Pilot: 3 gates × 41.280.000 steps (asertado) + evaluación.
  Coste ≈$0.15–0.30, wall ≈1–2h concurrentes.

## 6. Métricas y umbrales (procedencia + precisiones congeladas)

- Primarias: success rate sampled, 200 eps, media over 3 seeds + 95%
  CI (método t). Theta final sólo; record-only en tren. Prohibido
  tunear o elegir checkpoints con I1/H/G1G2-heldout.
- G1/G2-heldout (adquisición del gate): v2/g1g2_heldout_v1.json
  (200+200 eps, SeedSequence(8000/8001), sha256
  f684fc34561835830eaf0ba000a75f074e3a4d8289b39c651d48c73f6726c997).
  Barra informativa ≥0.70 (NUEVA): si falla, H-FAIL se lee "gate no
  adquirió". No decide GO.
- Umbrales I1≥0.70, H≥0.60, +40pp: TODOS de v1 PREREG §5 (no nuevos).
- Márgenes (+40pp skill-sola; +0.10 Q2) = criterios prácticos
  preregistrados, NO garantías estadísticas: la incertidumbre de una
  diferencia depende de su variabilidad y del diseño.
- Límite 40pp (sin datos H nuevos): exige H-media ≥ max(0.60,
  ref+0.40); si ref > 0.60 es inalcanzable. Bien planteado aquí
  (chance frozen ≈0.0–0.1). Si la referencia lo hiciera imposible,
  se reporta fallo DE ESE CRITERIO, no incapacidad de resolver.
- Margen Q2: diferencia pareada por seed (gate − control) en H-media
  ≥ +0.10, por control (uniforme, shuffle); per-batería descriptivo.
- Reporte en TRES bloques separados: (a) absoluto I1/cada H;
  (b) ventaja vs mejor skill sola; (c) ventaja vs uniforme/shuffle.
- SIN-VENTAJA = no alcanzó +0.10 preregistrado. NO demuestra
  equivalencia ni ausencia de cualquier beneficio.

## 7. Veredictos

- Q1 (GO intacto): I1 ≥ 0.70 + cada H ≥ 0.60 + H-media ≥ ref+0.40
  ⇒ PASS; I1 < 0.70 ⇒ H UNINTERPRETABLE.
- Q2 por control: VENTAJA (≥+0.10) / SIN-VENTAJA (<+0.10).
- H resuelto + SIN-VENTAJA ⇒ éxito de la combinación SIN atribuir
  ventaja al routing. H no resuelto ⇒ Q2 descriptiva.
- GO (v2 completa) ⇔ Q1 total PASS. Q2 modula la AFIRMACIÓN, no GO.

## 8. Lanzamiento y análisis (congelado)

- Volumen tiny-learner-v2, ruta /vol/v2/runs. 9 archivos:
  `V2_{301,302,303}.{json,theta.npz,ckpts.npz}` (ckpts: init,
  post_gate; JSON: gate 400/40, heldouts, retención skills, steps).
- Comando (host, Modal 1.6.2, Python 3.11):
  `python3.11 -m modal run --detach v2/modal_v2.py`
- Salida: `creating ONE v2 batch call with 3 input(s) via
  experimental_spawn_map: [301, 302, 303]` + results en
  /vol/v2/runs on volume tiny-learner-v2.
- Análisis: v2/score_v2.py (integridad → equivalencia arranque →
  bloques (a)(b)(c) → veredictos).

## Congelación

- Protocolo: FROZEN 2026-10-28 (este texto).
- gate_train.py sha256[12]: 4c7854ab5357 (2026-10-28, SMOKE PASS)
- modal_v2.py sha256[12]: 280ffa47b3d9 (2026-10-28, SMOKE PASS)
- Smoke: PASS local 2026-10-28, primer intento (T0 identidad
  fitness/eval por episodio + determinismo 64 eps; T1 run 12 gens
  con steps exactos, freeze bit-exacto, gate cambiado, claves eval
  completas, hashes; T2 streams RNG intactos; T3 41.28M, seeds,
  shuffle 0, heldout, gates a cero). Salidas en /tmp/v2_smoke
  (desechables).
