# S13 propuesta: afinar T-STU-S10 con ES + retorno (SIN EJECUTAR)

Pregunta: ¿afinar el gate T-STU-S10 mediante retorno del
entorno mejora su control frente a conservarlo congelado?

## Diseño congelable antes del GO

- Arranque: T-STU-S10 final por theta (identidad de pesos
  verificada contra s10_models.json). Una corrida ES por theta.
- Gate 14→32→4 original, obs-14 originales, skills congeladas,
  vistas intactas. Sin docente, etiquetas ni planificador.
- Política dura en TODO: argmax-4 → skill greedy con vista,
  igual en fitness que en evaluación. Sin mezcla suave ni
  sampled en ningún punto (diferencia declarada vs ES previo).
- Fitness: media del retorno G1 exacto (mismo `ret` de los
  rollouts) sobre F=6 episodios por candidato.
- Entrenamiento: ACQ15-train, 12 escenarios nuevos (4/región),
  sellados pre-entreno con semilla nueva + hash + chequeo de
  duplicados contra ACQ/ACQ7/ACQ9/ACQ10/ACQ12/ACQ14.
- Ventana fitness determinista: generación g evalúa los 6
  escenarios `[(2g+k) mod 12, k=0..5]` (cubre los 12 cada
  2 gens; fórmula congelada, sin selección por desenlace).
- Evaluación: ACQ15, 200 escenarios nuevos sellados pre-entreno;
  se abren solo con los 3 modelos finales listos. Brazos:
  afinado (gen 25) vs T-STU congelado, mismos escenarios.
- ES idéntico al previo salvo escala: antitético, centered-rank
  (`src/es.py`), sigma=0.1, alpha=0.05; pop_pairs=8 (16 cand),
  generations=25. RNG: SeedSequence nueva, spawn(3), congelada.
- Checkpoint final gen 25, sin selección ni descarte de thetas.
  Historial de fitness-train gratis (del propio ES); sin
  chequeos que toquen ACQ15.
- Primaria: llegadas afinado−congelado por theta y media, con
  pareados. Secundarias: retorno, colisiones, timeouts.

## Relación con el ES anterior (v2conf)

Mismo optimizador (antitético, rank shaping, σ=0.1, α=0.05).
Cambia: punto de partida (pilot3 → T-STU-S10), política
(suave+sampled → dura+greedy en train y eval), datos
(pools g1/g2 → ACQ15-train sellado) y escala (400×64×8 →
25×16×6). Una mejora NO sería atribuible al optimizador
(constante salvo pop) ni solo al punto de partida
(confundido con política y datos).

## Generaciones, episodios y coste (evidencia)

- Ritmo medido: S7, 2400 rollouts greedy / 143.1 s = 0.060
  s/ep (misma política dura, escenarios ACQ frescos).
  Cruces: S12 ~0.05/rollout; S4 ≤0.10 (incluye baselines).
  Uso 0.06 s/ep, conservador. Álgebra ES despreciable
  (612 params, numpy).
- Train: 25 gens × 16 cand × 6 eps = 2400 eps/theta;
  ×3 thetas = 7200 eps → ~432 s.
- Eval: 2 brazos × 600 = 1200 eps → ~72 s.
- Generación+sellado datos, prep, hashes: ~30 s (estimación
  por precedentes mkdata; se mide al ejecutar).
- Total estimado ≈ 535 s. Tope viable: 900 s acumulativos
  CPU local, incremental, INCOMPLETE sin ampliación.
  Margen ~68% sobre la estimación. Sin Modal.
