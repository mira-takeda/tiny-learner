# W1 — PREREGISTRO CONGELADO (follow-up rediseñado post-I1)

Fecha: 2026-10-30. GO del usuario con corrección metodológica.
W1 es un follow-up REDISEÑADO después del diagnóstico I1-GAP
(decisión A + corrección W1). NO es continuación ciega del
preregistro original: nueva distribución G1, nueva puerta,
H re-sellado para esta línea. I1 queda retirado permanentemente
como puerta/test ciego (abierto en N5); ningún número de I1
parametriza nada en W1.

## 1. Antecedente y corrección metodológica

- N5: coordinador N3b (entrenado en G1 train-manifest) 0.148 en I1.
- I1-GAP: G1-train visto d_gh en [35.6,104.1] (restricción
  registrada beacon-haz>=35) vs I1 en [12.0,24.9]: solape cero.
  Decisión A aceptada: I1 estaba fuera del soporte geométrico G1.
- Corrección W1 (vinculante): el nuevo generador NO usa bandas
  derivadas de I1 ("d_gh 10-35", "ángulos pequeños" u otras).
  Wide-G1 cubre el espacio físicamente válido de G1 de forma
  procedural y general, con límites derivados SOLO de la
  definición registrada de la tarea (gen.py + tasks.py).

## 2. Generador wide-G1 (w1/gen_wide.py, congelado y hasheado)

Espejo exacto de la rama 'guarded' de gen.sample_config (mismo
orden de sorteos, mismos TRIES=600 externos / 50 internos,
mismas primitivas _agent/_pt): lo ÚNICO que cambia son dos
cotas, cada una derivada de constantes registradas:

- agent ~ U[20,80]^2, beacon ~ U[5,95]^2 (idéntico).
- agent-beacon >= MIN_SEP = 10 (idéntico; tarea no trivial).
- hazard ~ U[5,95]^2 con:
  - agent-haz en [35,70]: UNIÓN de las bandas spawn de la
    familia B (pursuit) registrada — B1 [50,70], B2 [40,60],
    B3 [40,55], Badv [35,45]. G1 contiene una persecución;
    [35,70] es el soporte pursuit válido registrado (Badv,
    el piso, fue adquirido: pilot3 B=0.83). Fuera de ese
    soporte la supervivencia no está verificada (degenerado)
    o no pertenece a la tarea.
  - beacon-haz >= MIN_SEP = 10, SIN cota superior (limitado
    por la arena, máx ~127). 10 es la separación mínima
    tarea-espacio para entidades distintas (agent-beacon,
    agent-obj, H2 agent-haz). Cubre cercana→lejana.
  - SIN restricción angular: alineado/lateral/detrás emergen
    del muestreo uniforme de posiciones (rango completo válido).
- speed ~ U[3.0,3.5]: soporte de velocidad G1 registrado
  (unión G1a/G1b/G1adv). El eje ampliado es geometría;
  velocidades y físicas intactas.
- Sin regiones: UNA distribución general (las regiones eran
  estructura curricular, no definición de tarea).

Tabla de derivación (número -> fuente registrada, ninguna I1):

| número      | fuente                                     |
|-------------|--------------------------------------------|
| 10 (ag-be)  | MIN_SEP, gen.py (definición tarea)         |
| [35,70] agz | unión REGIONS B spawn_lo/hi (prereg)       |
| 10 (be-haz) | MIN_SEP entidades distintas (tarea-espacio)|
| sin cota db | arena [5,95]^2 vía _pt (física)            |
| [3.0,3.5]   | unión REGIONS G1 speed (prereg)            |
| sin ángulo  | construcción (sin restricción)             |

Mismas físicas, observaciones (14-dim) y rewards (tasks.py
intacto, hasheado). Mismo éxito G1 (llegada <6 sin contacto <5).

## 3. Pools y semillas (bloque W1: SeedSequence 21100-21107)

Ninguna semilla reutilizada de etapas previas. Patrón N-series:
semilla by_id = generate_state(1,uint32) de spawn(i).

- W1G1-DEMO: 18 wide-G1 (ss 21100), IDs W1G1-D00..D17.
  Partición imitación: primeros 12 train / últimos 6 eval
  (regla por orden de generación, pre-datos).
- W1G1-ESPOOL: 12 wide-G1 (ss 21101), IDs W1G1-E00..E11.
- W1G1-GATE: 200 wide-G1 (ss 21102), IDs W1G1-G000..G199.
  SELLADO pre-entreno; apertura única en puerta.
- W1G2-GATE: 200 G2 vía gen.sample_config congelado, ciclo
  G2a/G2b/G2c (ss 21103), IDs W1G2-G000... SELLADO pre-entreno.
- W1SING-GATE: 80 singles vía sample_config (20/familia,
  ciclo de sus regiones; ss 21104). Spot-check control.
- Planner per-run kids: 54 de ss 21105 (orden plan fijo).
- Init imitación: 3 de ss 21106 (documentadas en freeze).
- ES perturbaciones: semilla 21107 (entero, receta N2).

Disjunción (semillas + cfg completo): entre los 5 pools W1 y
contra train_meta, train_control, test_battery, eval_conf,
eval_new, train_conflict y baterías S/ACQ/N (vía freezes).
Cero colisiones exigido.

## 4. Reutilización congelada (distribuciones NO cambiadas)

G2/singles/planner/política/skills/físicas son idénticos a la
línea respaldada; se reutilizan artefactos congelados (sha):

- Demos G2: 54 runs registrados n1_demos.json (18 esc × 3 th)
  + partición G2 N1 intacta (12 train / 6 eval). G2 no cambió;
  re-correr sería cómputo idéntico en semántica.
- Singles imitación: n1b_pairs.json (600/200 por clase/th/split).
- ES pool G2 (12) + pools singles (8×4) + regla calendario N2
  (gen g: G1 idx (2g,2g+1)%12, G2 igual, singles g%8;
  ventana 2G1+2G2+ABCD). Slots G1 <- W1G1-ESPOOL.
- Planner G1: ref_plan.shoot/snap (H25/K40/5x5/replan/retorno,
  asserts + sha fichero). Fallos incluidos (protocolo N1).
- Política: argmax-4 gate -> greedy viewed skill
  (s4_closed.greedy_skill, sha fichero); fitness = retorno
  exacto de tarea. Igual en entreno y evaluación.
- Skills: pilot3_runs/P3_{301,302,303}_theta.npz (sha) +
  re-verificación VC==P3 bit-exacta en congelación.
- src congelado: tasks/env/models/policies/gen,
  v2views/views_train (shas en MANIFEST).

## 5. Recetas (verbatim N1b + N2; único cambio: distribución G1)

Imitación (w1_imitate.py): gate 14->32->4 (612 params),
Xavier oculta lim=sqrt(6/46) + ceros salida (S3), Adam
full-batch lr=1e-3 (0.9/0.999/1e-8), CE 4-clases, 5000
épocas, ckpt final, sin selección. Datos: 54 demos wide-G1
nuevas + 54 demos G2 reutilizadas + singles reutilizados.
Assert regresión: D-mass>0 en singles-eval. Offline only.

ES-duro (w1_es.py): 8 pares antitéticos / 16 cands,
sigma=0.1, alpha=0.01, 25 gens, rank-shaping con empates
(receta S14), inits W1-imit, calendario §4, skills
congeladas (shas pre+post). 25 ckpts + normas (registro,
sin selección). 3 thetas 301-303, sin selección.

Tiny original intacto: gate 612 + 4 skills pilot3 = 7944
params totales (asertado). Sin cambio de arquitectura.

## 6. Puerta fresca (w1_gate.py, apertura única)

Brazos {W1-final, W1-imit} × {W1G1-GATE, W1G2-GATE} × 3 th
= 2400 eps. Primaria: media W1-final >= 0.70 en CADA
familia (medias sobre 3 th). Secundaria: dif pareada vs
imit; spot-check W1-final en W1SING-GATE (descriptivo).
PASS ambas familias -> abrir H una vez. Si no -> STOP,
reporte, ningún retune. La puerta mide funcionamiento
del coordinador, NO transferencia novedosa.

## 7. H (w1_h.py, sólo si puerta PASS; apertura única)

Lee SÓLO entradas H1/H2/H3 de test_battery (asertado;
I1 y resto de baterías jamás). Brazos {W1-final,
A/B/C/D-only} × 3H × 200 × 3 th = 9000 eps.
Barras de REFERENCIA descriptivas (follow-up, NO
preregistradas): cada H >= 0.60; H-media >= mejor
single + 0.40. Sin monolito (pendiente; declarado).
CIs: convención t(2) n=3 (S1). H es el test real de
composición no vista de esta línea post-I1.

## 8. Presupuestos y disciplina

collect 1800 s / imitate 900 s / es 2400 s / gate 900 s /
H 1200 s, acumulativos con resume; INCOMPLETE honesto
(nunca silencio). Prohibido: leer/ejecutar I1 (siempre);
abrir H sin puerta PASS; abrir puerta/H más de una vez;
cambiar arquitectura, semillas, pools o receta tras ver
resultados; selección de checkpoint; Modal (CPU local).

## 9. Registro de congelación

Registro máquina: w1/MANIFEST.json (código + este doc + freeze)
y w1/w1_freeze.json (sección "pins": reutilización y datos
externos). Registro humano: bloque de hashes en RUNLOG.md al
congelar (patrón BEGIN-HASHES) + shas de pools al generarlos.
Este documento NO se edita tras congelar (rompería MANIFEST);
§9 es plantilla intencional: los valores viven en MANIFEST,
w1_freeze.json y RUNLOG.
