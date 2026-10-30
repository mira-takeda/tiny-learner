# v2conf_diag/ — sonda conmutación fija en ACQ (medición, sin entreno)

Artefactos frozen (v2conf_runs/, pilot3_runs/, datos) intactos: todo
aquí es sólo-lectura sobre ellos. Veredictos v2conf aceptados sin
modificar: adquisición FAIL (ACQ 0.102 < 0.70), aportación-B por
debajo del margen, transferencia UNINTERPRETABLE.

## S1. Sonda fija "B si haz<15, si no A" (switch_probe.py)

Regla fijada pre-run, sin barrido, histéresis ni ajustes posteriores:
h_obs = obs[7]·DMAX (slot S2 de la obs 14-dim visible al gate);
si h_obs < 15 → skill B con su vista, si no → skill A con su vista.
Gate no usado. Sin información privilegiada en la política.

Codificación verificada pre-run: env._slot = [ux, uy, dist/DMAX],
hazard en obs[5:8], DMAX = 100·√2 = 141.4214; obs[7]·DMAX == dist
real con peor error 7.11e-15 en 200 pasos (env sólo para verificar).

Batería ACQ 200 (familia G1), thetas VC 301–303 (skills = pilot3
post_D bit-exacto, verificado), sampled, mismas semillas/horizonte.
Mundo, recompensa y éxito intactos (TaskRunner frozen).
Ramas A/B validadas: reproducen skill-sola logadas al exacto
(6/6: 0.130/0.325, 0.140/0.230, 0.110/0.165).

## Resultado (switch_summary.json, switch_records.json)

Por seed (200 eps): éxito/colisión/timeout, duración media,
fracción de pasos en B, cambios A↔B por episodio.

| theta | éxito | colisión | timeout | T_med | frac_B | camb_med |
|-------|-------|----------|---------|-------|--------|----------|
| 301   | 20    | 180      | 0       | 16.0  | 0.276  | 2.4      |
| 302   | 30    | 170      | 0       | 22.6  | 0.323  | 3.1      |
| 303   | 13    | 187      | 0       | 17.0  | 0.299  | 2.4      |

Media sr = 0.105, IC95% = [0.000, 0.211] (t(2) = 4.303, convención
score_vconf). Referencia ≥0.70: NO alcanzada. Comparación (sampled,
refs frozen): gate 0.100/0.115/0.090; A-sola 0.130/0.140/0.110;
B-sola 0.325/0.230/0.165. La sonda queda a nivel del gate y por
debajo de B-sola en las 3 seeds.

## Lectura acotada

Sólo esta conmutación fija resultó insuficiente en ACQ (S1
aceptada). No se concluye módulo/tarea ni imposibilidad. Alcance
v2conf: este currículo y presupuesto no produjeron coordinación;
no queda demostrado que la solución exija memoria, ni que toda
forma de entrenamiento bajo conflicto falle.

## S2. Planificador de referencia privilegiado (ref_plan.py)

NO capacidad de Tiny: usa estado exacto + simulador. Éxito =
solución encontrada; fracaso de búsqueda limitada ≠ imposibilidad.
Fijo pre-run: H=25, K=40, 5 bloques×5 pasos (P-prim: primitiva
sostenida por bloque; P-AB: skill por bloque con greedy
recalculado por estado), replanificación cada paso real, puntuación
= retorno G1 exacto, desempate primer-máximo, RNG secuencias 8100.
Sin barridos ni ajustes.

Validaciones: V0 batería/thetas; V1 clon-con-mismo-ruido ==
entorno bit-exacto (obs/done/retorno, 41 pasos×3 entradas, prueba
RNG-gemelo); V2 modelo-sin-ruido == entorno salvo la modificación
declarada ruido=0 (nota impl: el clip `*=` in-place del env difiere
1 ulp del `*`; espejado exacto); V3 greedy == composición frozen
skill/vista/argmax en 360 sondas por observación.

Micropiloto (IDs pre-fijados, 1 por región): A-ACQ-0000 (G1a),
A-ACQ-0001 (G1b), A-ACQ-0002 (G1adv). Conteos crudos (12/12
completos, 0 incompletos; total 67.1 s de 300 s):

| run | causa | T | pasos-imag | wall |
|-----|-------|---|------------|------|
| prim 0000/0001/0002 | llegada ×3 | 21/44/29 | 12k/24k/14k | 0.7/1.3/0.8 s |
| AB ×3 thetas ×3 eps | llegada 9/9 | 21–50 | 11k–34k | 3.0–9.5 s c/u |

P-AB: frac_B 0.19–0.73, cambios 4–18 por episodio (detalle en
refplan_pilot.json). Calibración medida: 14 378 pasos-imag/s
(modelo), 4 261 (modelo+MLP).

Lectura (corregida): soluciones encontradas con primitivas (3/3)
y con selección privilegiada + skills greedy (9/9) EN ESTOS 3
ESCENARIOS. P-AB alto NO demuestra que el gate actual pueda
aprenderlas. Riesgos vigentes: búsqueda insuficiente, modelo sin
ruido, restricción greedy. Sin inferencia estadística fuerte
desde 3 escenarios.

Estimación siguiente etapa (costes medidos: prim ~0.9 s/ep,
P-AB ~5.9 s/run; varía con longitud/dificultad): 30 eps ≈ prim
~1 min + P-AB ~9 min; ACQ completo ≈ prim ~3–4 min + P-AB ~60 min.

## S2-30. Expansión a 30 ACQ (s2_expand.py, s2_30.json, s2_traces.json)

Misma búsqueda importada de ref_plan (H/K/bloques/puntuación/
modelo/replanificación idénticos, asertados). 30 IDs = índices
0..29 (10/región); 12 runs del micropiloto REUTILIZADOS, 108
nuevos (prim×27, AB×27×3, orden episodio-major). Semillas por
ejecución: SeedSequence(8101) spawn en orden fijo, registradas;
trazas de llegadas conservadas (99). 500.8 s de 900 s, 0
incompletos. Sin ajustes ni ampliación a ACQ completo.

Conteos (llegada/colisión/timeout/incompleto): prim 30/0/0/0
(10/10 por región); AB 81/9/0/0 — por theta 301: 26/4, 302: 27/3,
303: 28/2; por región×theta: todos 8–10/10. Muertes AB: 0007/302,
0008/301+302, 0009/303, 0011/301, 0013/301, 0024/301, 0028/302,
0029/303.

Mismo escenario: prim resuelve 30/30; AB-cualquiera 30/30
(ambos 30, sólo-prim 0, sólo-AB 0). Por theta: sólo-prim 4/3/2,
sólo-AB 0, ninguno 0.

Uso/costes: P-AB frac_B medio 0.34–0.49 por theta, cambios
7–10/ep; llegadas frac_B 0.45 vs muertes 0.30 (descriptivo).
prim ≈39 s, P-AB ≈510 s (sumas redondeadas), total 500.8 s.

Ref ≥0.70 (diagnóstica): prim 1.00, AB 0.90 global
(0.867/0.900/0.933 por theta) — todos por encima.

Info privilegiada usada (fuera de las 14 obs del gate): posiciones
absolutas (agente/haz/goal); velocidad del hazard + ley de
persecución (la obs S2 da dirección+distancia, NADA de movimiento);
geometría de muros [0,100] + rebotes (cero slots de muro en obs:
el gate no distingue junto-a-muro de centro con igual geometría
relativa); modelo de transición + simulación multi-paso sin ruido
y puntuación de futuros (gate = una pasada feedforward sin
memoria ni simulación). Vel. propia agente NO es privilegiada
(obs vel/5 invertible). El planificador NO lee el RNG del
episodio (modelo ruido=0).

Conclusión: "soluciones encontradas mediante selección
privilegiada de skills". No es capacidad demostrada del gate; el
fallo del gate no se atribuye sólo al entreno (objetivo,
optimización y representación siguen sin distinguir).

## S3. Imitación del planificador (s3_repro.py, s3_imitate.py)

Pregunta: ¿el gate 14→32→4 predice A/B del docente desde sus 14
obs? Offline sobre estados del docente; NO éxito de control.
Estudiante NO ejecutado como controlador.

(b) Reproducción: 9/9 fallos idénticos (causa, T, fracB,
switches, imag_steps), 38.1 s. Retorno no estaba logado en
origen; imag_steps idénticos es el check más fuerte.

Datos: 81 trazas AB (72 llegadas + 9 fallos), pool 27
escenarios; micropiloto fuera. Replay en env frozen →
2717 obs exactas pre-acción + etiqueta skill. Réplica obs
bit-exacta paso a paso; alineación temporal verificada;
estados redondeados ±2e-3; desenlace replay = registrado.

Partición compartida 18/6 train / 9/3 eval por región, orden
fichero. Init: Xavier frozen (semilla 9200+theta) en oculta,
ceros en resto (probs iniciales uniformes). Diferencia vs ES:
ES rompe simetría con ruido gaussiano sobre ceros; Adam desde
ceros la mantendría en la oculta (gradientes idénticos).
Smoke: grad rel-err 7.5e-10; CE 1.386→1.108 (50ep), |dW1|>0;
500ep full 0.574 < solo-biases 1.039.

Entreno: CE A/B 4-clases, Adam full-batch lr=1e-3, 500 épocas,
modelo final, sin tuning. Por theta (train n/acc | eval n/acc,
bal, cm[A/B], P/R, CDpred | const, d15 en eval):
- 301: 530/0.715 | 368/0.660, bal 0.652, [[83,50],[75,160]],
  P 0.53/0.76 R 0.62/0.68, CD 0 | const 0.361, d15 0.402.
- 302: 668/0.711 | 241/0.598, bal 0.592, [[103,67],[30,41]],
  P 0.77/0.38 R 0.61/0.58, CD 0 | const 0.705, d15 0.415.
- 303: 658/0.661 | 252/0.591, bal 0.580, [[96,55],[48,53]],
  P 0.67/0.49 R 0.64/0.53, CD 0 | const 0.401, d15 0.500.

Lectura: imitación parcial (~0.6 eval), supera d15 siempre y a
constante en 301/303 (en 302 pierde acc bruta 0.598<0.705 por
eval cargado a A, pero gana balanceada 0.592>0.500). Predice
ambas clases (no solo-A); nunca C/D. Cambio ejecucional
pendiente de declarar si actúa: docente = duro+greedy, ES =
suave+sampled; no atribuir diferencias solo al método.

Cómputo: 38.1 + 5.7 = 43.8 s de 600 s. Archivos: s3_repro*.json,
s3_data/models/results.json. Veredictos anteriores intactos.

## S4. Estudiante en lazo cerrado (s4_closed.py, s4_results.json)

27 episodios: 9 eval-S3 × 3 thetas. Estudiante época-500 intacto:
argmax-4 → greedy de la skill con vista. Sin C/D-mask ni fallback.
Baselines mismos escenarios: docente registrado (sin re-consulta),
constante-mayoría-train greedy (301/302→A, 303→B), d15 greedy,
gate-v2conf greedy. Diagnóstico hard+greedy; NO protocolo
suave+sampled; veredictos intactos. 2.8 s de 300 s, 0 incompletos.

Llegada/colisión/timeout por theta (student, const, d15, vgate,
teacher): 301: 1/7/1, 1/8/0, 0/9/0, 1/8/0, 8/1/0; 302: 3/6/0,
2/7/0, 1/8/0, 2/7/0, 8/1/0; 303: 5/3/1, 2/0/7, 1/8/0, 1/8/0,
8/1/0. Global: student 9/16/2 (sr 0.33), const 5/15/7, d15 2/25/0,
vgate 4/23/0, teacher 24/3/0. Ref 0.70: ningún brazo la alcanza;
teacher 0.89 (privilegiado, no comparable como capacidad).

Uso/cambios student: 301 uses [141,272,0,0] sw 5.6; 302 [92,87,0,0]
sw 0.8; 303 [88,375,0,0] sw 4.7. C/D: student 0 pasos (nunca sale
de A/B); vgate-argmax-w 33 pasos en D (th 301). Las 9 llegadas
student caen en escenarios resueltos por el docente; ninguno de
los 3 fallos-docente lo resuelve el estudiante.

Estados: hobs visitado-student (n=1055) min 5.3 media 32.5 max
75.1; estados-docente-eval (n=861) min 5.2 media 28.1 max 71.9.
Rangos solapados; el estudiante muere antes en la mayoría (T
medias 46/20/51 con 2 timeouts de 200) — distribuciones
distintas aunque el soporte-h se solape.

Precisiones aceptadas: (a) docente y estudiante usan routing duro
+ greedy; esa diferencia ejecucional NO explica la brecha entre
ambos (sí limita comparaciones con ES suave+sampled). (b) Solapar
rangos de distancia-hazard NO demuestra solape del espacio
completo de observaciones.

## S5. Anchura × presupuesto (s5_width.py, s5_results.json)

Protocolo (congelado pre-run + 5 precisiones): 6 entrenos
(32/128 × 3 thetas) a 5000 épocas, mismos datos S3 (obs-14
completas verificadas S1+S2 activas; vistas solo en skills del
docente, nunca en inputs), partición, CE, Adam. Celdas-500 del
mismo run. Init única por anchura×theta (9200/9300+theta).
Repro-S3 @500 ancho-32: bit-exacta 0.00e+00 ×3 (check antes de
interpretar). Offline; sin control, docente ni planificador.

Por theta, celdas 32/500, 32/5000, 128/500, 128/5000
(trainCE→acc | evalCE→acc/bal):
- 301: 0.57→0.72|0.63→0.66/0.65; 0.09→0.98|2.36→0.57/0.60;
  0.53→0.74|0.65→0.63/0.59; 0.02→1.00|5.22→0.53/0.57.
- 302: 0.52→0.71|0.65→0.60/0.59; 0.17→0.93|1.86→0.59/0.60;
  0.49→0.73|0.61→0.68/0.67; 0.03→1.00|3.15→0.66/0.66.
- 303: 0.61→0.66|0.69→0.59/0.58; 0.19→0.94|2.08→0.60/0.61;
  0.57→0.71|0.68→0.64/0.62; 0.03→1.00|3.59→0.65/0.66.
C/D predichas: 0 en las 12 celdas. Curvas en s5_results.json.

Efectos: AJUSTE mejora siempre con anchura y presupuesto
(train 0.66–0.74 → 0.93–1.00). GENERALIZACIÓN inconsistente:
301 empeora con ambos; 302/303 mejoran con anchura (@500:
+0.09/+0.05 acc; @5000: +0.07/+0.05) y planas con presupuesto;
CE-eval empeora con presupuesto en las 6 (sobreajuste).
Una sola init por celda: no identifica efecto general de
capacidad. Sin lectura de control, observabilidad o memoria.

Coste: 78.1 s de 300 s, 0 incompletos. Updates 5000/run;
muestra-épocas 2.65M/3.34M/3.29M por theta (igual entre
anchuras); FLOPs ≈ épocas×(108·H·N+12P): 32 ≈ 9–12 GFLOP,
128 ≈ 37–46 GFLOP (~4×). Veredictos intactos.

## S6. Celdas S5 en lazo cerrado (s6_closed.py, s6_results.json)

4 celdas fijas × 9 eval-S3 × 3 thetas. 32/500 reutilizada de S4
(pesos bit-idénticos 0.0e+00, política importada de s4_closed);
81 episodios nuevos (32/5000, 128/500, 128/5000). Política S4,
sin C/D-mask ni planificador. Docente/baselines reutilizados
(teacher 24/27, const 5, d15 2, vgate 4 llegadas). Diagnóstico
reutilizado, no confirmación independiente. 1.4 s de 300 s.

Llegadas por theta (32/500, 32/5000, 128/500, 128/5000):
301: 1, 4, 2, 2 (col 7/5/7/7, tmo 1/0/0/0; T 46/25/22/18;
uses [141,272]/[111,111]/[56,144]/[87,77], sw 5.6/6.4/1.6/4.3).
302: 3, 4, 2, 7 (col 6/5/7/2, tmo 0; T 20/31/25/32;
uses [92,87]/[143,139]/[104,117]/[155,136], sw 0.8/6.4/1.0/6.1).
303: 5, 8, 3, 7 (col 3/1/6/2, tmo 1/0/0/0; T 51/38/34/41;
uses [88,375]/[178,160]/[95,208]/[186,179], sw 4.7/12.6/3.0/14.1).
Global: 9/16/7/16 de 27. C/D: 0 pasos en las 4 celdas.

Pareados por escenario (mejora=nonarr→arr, empeora=arr→nonarr):
presupuesto@32: 301 +4−1, 302 +3−2, 303 +4−1 (neto +3/+1/+3);
presupuesto@128: 301 +1−1, 302 +6−1, 303 +5−1 (neto 0/+5/+4);
anchura@500: 301 +2−1, 302 +1−2, 303 +1−3 (neto +1/−1/−2);
anchura@5000: 301 +2−4, 302 +3−0, 303 +1−2 (neto −2/+3/−1).
IDs en s6_results.json (trazas completas por episodio).

Lectura: MIXTO. Presupuesto (mejor ajuste, CE-eval peor en las
6) mejora control neto en 5/6 pareados pese a eval-acc plana o
peor. Anchura (mejor ajuste, eval mixta) da control mixto: @500
peor/plano, @5000 mixto por theta. Mejor ajuste offline NO
predice uniformemente mejor control; sin selección de modelo.

## S7. Escenarios nuevos ACQ7 (s7_mkdata/freeze/eval.py, s7_*.json)

200 ACQ frescos, mismo generador conflict_cfg (sha d7c2d472b166),
ciclo regional i%3, SeedSequence(9600). sha
bbe1fb9435d8; 0 colisiones/dups con baterías previas; banda ok.
Congelado pre-eval (candidato 32/5000 ×3 thetas, sin selección
por theta; elección POST-S6 documentada; ref 0.70 diagnóstica).

4 brazos × 200 × 3, greedy, mismas semillas/horizonte. 2400/2400
en 143.1 s de 300 s. Llegadas (tasa por theta + media):
- st325000: 73/97/108 → 0.365/0.485/0.540, media 0.463.
- st32500: 35/47/117 → 0.175/0.235/0.585, media 0.332.
- A-sola: 36/28/18 → 0.180/0.140/0.090, media 0.137.
- B-sola: 54/46/31 → 0.270/0.230/0.155, media 0.218.
Colisiones: st325000 127/103/92; st32500 155/153/78; A
164/172/182; B 32/12/20. Timeouts: ~0 salvo B (114/142/149) y
st32500 (10/0/5). T medias: st325000 26/38/38; st32500 36/23/50;
A ~11; B 140/162/163. C/D: 1 paso en 2400 episodios.

Pareados st325000 vs control (mejora/empeora llegadas): vs
st32500: +59−21, +63−13, +41−50; vs A: +50−13, +78−9, +97−7;
vs B: +55−36, +76−25, +90−13.

Lectura: candidato primero en media (0.463 < 0.70, ref NO
alcanzada) y en 8/9 comparaciones por theta (pierde 303 vs
32/500: 0.540<0.585). Selección en S6 + evaluación en S7 nuevo
= separación válida selección/evaluación (elección post-S6
documentada por transparencia, sin limitación adicional por
ello). No GO composicional; veredictos intactos.

## S8. Demostraciones 18 vs 36 (s8_mkdata/run.py, s8_*.json)

ACQ8-train18 (SeedSequence 9700, sha e987b1df…) + ACQ9-eval200
(9800, sha 57909132…), 0 dups, fuera S3-eval/S7. Congelado
pre-docente (54 consultas orden fijo, semillas SeedSequence
8300). Docente S2 54/54 (fallos incluidos); replay bit-exacto,
1652 pasos nuevos. A18 verificado (datos/partición/config/pesos
+ shadow bit-idéntico ×3). A36: 5000 updates, init 9200+theta.
Pasos/balance por theta (orig+nuevos): 301: 530 B.49+473 B.39;
302: 668 B.43+615 B.39; 303: 658 B.52+564 B.45. Promedio por
paso conservado. Eval ACQ9 común al final. 381.1 s de 600 s.

Primaria A36−A18 llegadas: 301: 81−64=+0.085; 302: 87−96=
−0.045; 303: 105−99=+0.030; media +0.023. Pareados
(mejora/empeora): 301 +44−27, 302 +30−39, 303 +42−36. Tasas:
A18 0.320/0.480/0.495, A36 0.405/0.435/0.525 (ref 0.70 lejos).
C/D 0 pasos. T/usos/switches en s8_eval.json.

Lectura: media positiva pequeña con signos mixtos (+,−,+) =
mejora observada, no ventaja robusta. Apoyo débil a ampliar
demostraciones; un negativo no probaría memoria/observabilidad.
Veredictos intactos.

## S9. Gate informado vs original (s9_mkdata/run.py, s9_*.json)

ACQ10-200 sellado (SeedSequence 9900, sha d6d411fa…, 0 dups).
Congelado pre-entreno. Dataset train-A36 exacto (1003/1283/1222,
IDs/etiquetas/orden verificados); x/y/t pre-acción por replay
validado (réplica bit-exacta, pristine intacto); hspeed de cfg
(procedencia registrada). Entrada informada en copia:
[8..11]=x/100,y/100,hs/5,t/200, [12:14]=0. Skills siempre con
obs pristine + vistas (asertado). Original = s8_models36
verificado. 5000 updates, init 9200+theta. 98.6 s de 300 s.

Primaria informado−original llegadas/200: 301: 80−83=−0.015;
302: 98−80=+0.090; 303: 109−96=+0.065; media +0.047.
Pareados (mejora/empeora): +35−38, +46−28, +41−28. Tasas:
orig 0.415/0.400/0.480, inf 0.400/0.490/0.545 (ref 0.70 lejos).
C/D 0 pasos. T/usos/switches en s9_eval.json.

Lectura: media positiva pequeña, signos mixtos (−,+,+) con
churn ~35% = apoyo débil a las variables como PAQUETE (sin
identificar cuál). Un negativo no descartaría observabilidad,
memoria ni otras limitaciones; brecha planificación-reactiva
intacta. Referencia privilegiada, no capacidad Tiny.

## S10. Etiquetas en estados estudiante vs docente (s10_*.py/json)

ACQ11-train9 (9930, sha 6132ae4c…) + ACQ12-eval200 (9940, sha
a0632017…), 0 dups. Congelado pre-recogida (fórmula índices
m=min + floor(j(L−1)/(m−1)), 27 runs control semillas 8400,
query-seeds por hash). Colector A36 congelado verificado
(identidad ejecución S4); 27 T-TEA trazas completas; queries
puntuales mismo pab_decide() (RNGs separados, sin tocar
trayectorias). 9 escenarios íntegros, fallos incluidos.

Etiquetas nuevas/theta (igual por pareja): 186/225/248.
Queries puntuales 659 (381k imag); trazas T-TEA 27 (551k imag).
Entreno 6×5000upd init 9200+theta. Eval ACQ12 común al final.
355.9 s de 900 s, 0 incompletos.

Primaria T-STU−T-TEA llegadas/200: 301: 103−96=+0.035;
302: 120−117=+0.015; 303: 121−113=+0.040; media +0.030.
Pareados +39−32, +36−33, +39−31. Tasas: T-STU
0.515/0.600/0.605, T-TEA 0.480/0.585/0.565, A36-ref
0.465/0.410/0.535 (ref 0.70 lejos). C/D ~0 (A36/301: 5 pasos).

Lectura: positivo pequeño, MISMO signo (+,+,+) = apoya esta
forma concreta de recoger estados del estudiante frente a
añadir más trazas docente. Ambos brazos superan A36 en las 3
thetas. Veredictos intactos.

## S11. Segunda ronda STU vs TEA (s11_*.py/json)

ACQ13-train9 (9950, sha f5c3bebb…, 3/3/3) + ACQ14-eval200 (9960,
sha caa5f959…), 0 dups. Congelado pre-recogida. Base = T-STU-S10
completo (1189/1508/1470, verificado); recolector T-STU-S10
congelado (identidad pesos + ejecución S4). 27 control
(semillas 8450) + 765 queries puntuales (mismo pab_decide,
seeds S11q, RNGs separados). Fallos incluidos.

Etiquetas nuevas/theta (igual por pareja): 234/281/250.
Queries 765 (479179 imag); control 27 trazas (500633 imag).
Entreno 6×5000upd init 9200+theta. Eval ACQ14 común al final.
422.8 s acumulados de 900 s, 0 incompletos.

Primaria STU−TEA llegadas/200: 301: 111−103=+0.040; 302:
123−113=+0.050; 303: 104−99=+0.025; media +0.038.
Pareados +36−28, +34−24, +45−40. Tasas: STU
0.555/0.615/0.520, TEA 0.515/0.565/0.495, TSTU10-ref
0.555/0.655/0.550 (ref 0.70 lejos). C/D: 0 pasos exactos de
56255. T/usos/switches en s11_eval.json.

Lectura: segunda ronda CONSERVA la ventaja STU (mismo signo,
media +0.038 ≈ S10 +0.030), pero frente al modelo S10 empata
en 301 y pierde en 302/303. La ventaja relativa no constituye
mejora acumulativa. Veredictos intactos. Sin tercera ronda.

## S11-diag. Offline vs control (s11_diag.json, solo lectura)

3 modelos × 3 bloques: ambos brazos S11 ajustan sus etiquetas
nuevas (0.80–0.90) con erosión leve en base (−1–2pp, las 3
thetas); TEA ajusta bien su train pero controla peor. Desacuerdo
STU/TEA: base ~10–12%, bloques nuevos 14–31%. ACQ14 por región:
G1a más fácil para todos; timeouts 0. Pareados vs TSTU10:
STU +32−32/+24−32/+30−36, TEA +31−39/+14−32/+25−36 (churn
~1/3). Pesos: fracción nueva ~15–17% en ambas rondas. Config
idéntica verificada en código (5000/lr/init/Adam/media-por-paso
/gate). Falta: curvas por época S8/S10/S11 (no se guardaron);
estados visitados ACQ14 (no se guardaron) — sin reconstruir.

Lectura: el ajuste offline NO ordena el control (TEA imita bien
y controla peor; TSTU10 no vio etiquetas S11 y empata/supera).
Indicios: erosión leve de base + mejora de imitación que no se
traduce. La causa no se identifica solo con esto.

## S12. Intervención única primera-diferencia (s12_eval.json, completo)

Regla congelada (s12_freeze.json): reproducir T-STU-S10 en
ACQ14×3, buscar primer paso donde STU-S11 propone otra PRIMITIVA
ejecutable sobre la misma obs, re-ejecutar con S10 hasta ahí, una
sola acción S11 y continuar con S10. Ruido por paso preservado
(RNG separado verificado; prefijo obs bit-igual). Nulos kept.

Historial de topes: INCOMPLETE bajo tope original 300 s
(exec-104: repro 600/600, 569/600 intervenciones, wall 322.5 s).
Continuación operativa +180 s autorizada 2026-10-30 completó
solo las 31 de 303 (5.3 s; 569 intactas bit a bit). Causa del
corte: guardado cuadrático (~estimado, ver TIMING180.json).
Parche O(1)+reapertura (OPFIX180, e503ce9d) validado en sandbox.
Respaldo pre-180 con código ejecutado original (3907e9ab),
hashes y reconstrucción de tiempos: s12_backup_pre180/.

Primaria intervención−S10 en llegadas (nulos incluidos):
301 +0.0100 (+27−25, 17 nulos); 302 +0.0150 (+23−20, 10 nulos);
303 +0.0450 (+25−16, 8 nulos). Media global +0.0233 (+75−61,
429 empates no nulos, 35 nulos). S10: 352/600; int: 366/600.
Timeouts 0 en ambos. Paso mediano t=3–4; skills en t solo A↔B.

Lectura: la primera sustitución bajo continuación S10 produce
efecto neto pequeño y positivo, con churn 22.67% (mejoras
12.50%, empeoramientos 10.17%; 429 empates excluyen 35 nulos).
No explica automáticamente toda la diferencia entre políticas;
decisiones posteriores no descartadas. Veredictos intactos.

## S12-cruce. INT vs STU-S11 completo (s12_cross.json/md)

Pareo por escenario/theta de S10, intervención y STU-S11
(s11_eval, correspondencia documental verificada: mismo ACQ14,
modelos y política; repro S10 0 discrepancias). INT−STU:
301 +0.0100 (+26−24); 302 +0.0550 (+29−18); 303 +0.0750
(+39−24). Global 366 vs 338, d=+0.0467 (+94−66, 440 empates).
Orden: INT > S10 (352) > STU. Solo compara continuación S10
vs S11 tras la primera sustitución compartida, en ACQ14.

## S13. Afinado ES desde T-STU (s13_eval.json, completo)

ES antitético rank-shaping con empates promediados (verificado:
igual→igual peso, todo-empatado→update cero), σ=0.1 α=0.05,
16 cand × 6 eps × 25 gens = 2400 eps/theta. Fitness: retorno G1
exacto en ACQ15-train12 sellado (ventanas 2/región, calendario
fijo, mismas semillas todos los candidatos). Init T-STU-S10
verificado; skills hasheadas inicio+fin. Fallo prep (spawn bug,
0.8 s) descartado con presupuesto preservado. Eval ACQ16-200
abierta solo con 3×25 gens. Wall 133.4 s de 900 s.

Primaria afinado−congelado en llegadas: 301 +0.1150 (+50−27);
302 +0.1000 (+50−30); 303 −0.1050 (+30−51). Global 358 vs 336,
d=+0.0367 (+130−108). Secundarias: retorno medio acompaña
(301 5.34/2.05, 302 6.95/4.43, 303 2.07/4.07); timeouts solo
en afinado (5/2/1 vs 0); ||Δθ||≈10–11 las tres thetas.

Lectura: signos mixtos por theta, media positiva pequeña.
Optimizador constante vs ES previo salvo población; mejora no
atribuible al optimizador ni solo al arranque (confundido con
política dura y datos). Veredictos intactos.

## S13-diag. Ajuste vs generalización (s13_diag.json, 0.9 s)

Congelado vs afinado-final en ACQ15-train12 (72 eps, política
dura, retorno exacto). Llegadas TUNED−FROZEN en train: 301 +2
(8/6, +3−1), ret 5.92/3.09; 302 +2 (10/8, +2−0), ret 9.87/6.77;
303 −2 (8/10, +1−3), ret 6.92/10.27. Sin timeouts en train.
Vs ACQ16 (+23/+20/−21): 301/302 mejoran en ambos; 303 empeora
en AMBOS → no es brecha de generalización; revisar eficacia
o estabilidad del ajuste en esa theta. Ningún resultado
identifica el mecanismo por sí solo.
Normas: ||init|| 35.77/32.65/33.13; ||Δ|| 11.10/10.85/10.03
(ratio ~0.30–0.33, cos ~0.95). Updates por generación NO
guardados (solo theta final + history de best-fitness);
no reconstruir. Sin selección de checkpoint; S13 intacto.

## S14. ES alpha=.01 vs .05 (s14_eval.json, completo)

Idéntico a S13 salvo alpha .05→.01: mismo init T-STU,
ACQ15-train12, calendario, es_seed 4242 (perturbaciones
iguales por fórmula+semilla, verificado textual), σ=.1,
8 pares, rank-shaping con empates, 25 gens. ACQ17-eval200
nuevo sellado, cero dups. Eval 3 brazos sin reentrenar.
25 checkpoints + normas de update guardados (registro).
Wall 209.4 s de 300 s.

Primaria A01−A05 en llegadas: 301 −0.0400 (+34−42);
302 −0.0200 (+23−27); 303 +0.1500 (+47−17). Global 385 vs
367, d=+0.0300 (+104−86). Secundaria vs congelado (354):
301 A01 +0.025/A05 +0.065; 302 +0.125/+0.145; 303 +0.005/
−0.145. Retorno acompaña; timeouts 1/0/0 A01, 5/5/1 A05,
0 frozen.

Lectura: la reducción limita la regresión 303 (A01 empata
al congelado donde A05 pierde 29) y cede levemente en
301/302. Apoya esta reducción concreta; no demuestra por
sí sola la causa de S13. Veredictos intactos.

## S15. Confirmación A01 en ACQ18-400 (s15_eval.json, completo)

Solo evaluación: A01-final S14 vs T-STU congelado, modelos y
skills verificados, misma política dura y semillas. ACQ18
(134/133/133) sellado, cero dups. 2400/2400 eps, wall 26.9 s
de 300 s.

Primaria A01−congelado: 301 +0.0600 (+69−45); 302 +0.1575
(+77−14); 303 +0.0225 (+47−38). Global 776 vs 680/1200,
d=+0.0800 (+193−97). Secundarias: retorno acompaña (301
4.43/3.00, 302 8.54/4.65, 303 4.52/3.77); timeouts 0/0/0
A01, 1/0/0 frozen; uso C/D cero ambos; switches similares.
Ref .70 descriptiva: A01 0.590/0.758/0.593, media 0.647 —
no es GO composicional. Veredictos intactos.

## S16. A01 vs skills solas en ACQ18 (s16_eval.json, completo)

Diagnóstico con batería reutilizada: A01-1200 de S15 sin
repetir; A/B solas greedy 2400 eps nuevos, mismos escenarios,
semillas y hashes. Wall 44.8 s de 90 s.

A01−A por theta: +0.3925/+0.6200/+0.4925 (+182−25/+260−12/
+209−12). A01−B: +0.2975/+0.5125/+0.3925 (+166−47/+228−23/
+194−37). Global: A01 0.6467, A 0.1450, B 0.2458 (=ref-max);
d vs A +0.5017 (+651−49), vs B +0.4008 (+588−107). A sola
muere por colisión (321/345/360); B sola sobrevive por
timeout (215/269/272) sin llegar. Retorno: A01 4.43/8.54/
4.52; A −3.14/−4.90/−5.69; B 1.15/1.57/0.24.

Lectura: la selección aprendida supera a cada skill constante
en las tres thetas y media. Diagnóstico en batería abierta;
candidato y criterios intactos.

## Consolidado A01 + repro mínima (repro_s16/)

Candidato fijo: A01-final S14. S15 confirma mejora vs S10 en
ACQ18 nuevo (+.080 medio, tres thetas positivas). S16: ventaja
vs A/B constantes en esa batería (0.647 vs 0.145/0.246).
Comando comprobado (58 s, CPU): `python3.14 repro_s16/repro.py`
— verifica 10 hashes, ejecuta 3600 episodios, reproduce los
archivados (cuentas exactas). Ejecución real local, no muestra
grabada. Procedencia y faltantes: repro_s16/PROVENANCE.md.

## N0. Planner A/C privilegiado en G2 (n0_g2.json, completo)

Pregunta: ¿contienen A+C congeladas una solución G2 bajo
selección privilegiada? Maquinaria S2 (H/K/bloques/replan/
scoring idénticos, asserts) + mirror G2 NUEVO (objeto,
touched/dwell/prev; G2 sin hazard → EXACTO, sin ruido).
V0 hash+VC==P3; V1 gemelo bit-exacto 5996 pasos×30; V3 k=A,C
720 sondas. Batería: 30 G2 train-manifest (10/región,
permitidos, no I1/H). Brazos ×3 thetas, mismas semillas.
Wall acumulativo 1311.3 s (900+300+150, INCOMPLETE honestos
intermedios, reanudación incremental).

A-only 0/90; C-only 71/90 (21/24/26); planner AC 88/90
(29/29/30). Planner-ok con ambas constantes fallando: 19/90
(+19−2 vs C-only). Usa ambas: A=22.7/C=48.0 pasos, 19.7
switches repartidos (cuartiles .01/.29/.54/.80/1.0).
Fallos AC (2): M-G2-2009/302 timeout sin touch; M-G2-2150/301
touch sin dwell (obj perdido, oz 29.9).

Lectura (rama pre-registrada POSITIVA): A+C contienen las
piezas; el problema es enseñar esa coordinación al gate
14→32→4 sin cambiar arquitectura. Siguiente: N1 imitación.

## N1. Imitación 4-vías G1+G2 (n1_*.json, completo)

108 demos planner (P-AB en 18 G1 + P-AC en 18 G2
train-manifest, ×3 thetas; 891 s de 1800 s) → gate
14→32→4 por theta, receta S3, Adam 5000ep, ckpt final.
Partición 24/12 compartida. D ausente (documentado; vía
N2-retorno). Bug contable + métrica-alias corregidos;
hash modelos idéntico pre/post-fix (d00bc8e2).

Train acc 0.881/0.849/0.915; eval acc 0.592/0.513/0.598
vs const 0.440/0.526/0.467 (302 bajo const). CE-eval
2.61/2.32/2.64: sobreajuste patrón-S5. Debilidad declarada:
B raro en train (21/36/45) porque G1-train-manifest es
fácil (planner T~5, casi todo A). Offline, no control.
Siguiente N2: ES-duro α=.01 sobre G1/G2+singles.

## N1b. Corrección singles (n1b_*.json, completo)

N1 no era 4-vías: D sin labels recibía presión CE negativa.
Corrección acotada (omisión de singles prevista en protocolo,
no tuning): 108 demos G1/G2 intactas (hash verificado) +
singles A/B/C/D con label trivial (600 train + 200 eval
pares/clase/theta exactos, episodios disjuntos, pools sin
heldout/N2-pools). Receta/init/seeds N1 idénticos, desde cero
(6 s recolecta + 93.5 s entreno).

Eval acc 0.768/0.785/0.836 (const 0.323/0.364/0.331).
Singles-eval sel A/B/C/D: 301 .835/1/.795/.97, 302
.785/1/.965/.97, 303 1/1/.945/1. D-mass 220/237/200 > 0
(assert regresión pasado). Pre-N2 control heldout (8 s):
G1 50/49/37, G2 35/33/26. N2 hypers congelados ANTES de
mirar control, sin cambios.

## N2. ES-duro desde N1b (n2_*.json, completo)

25 gens × 16 cand × 8 eps (2G1+2G2+ABCD, pools manifest),
α=.01, inits N1b verificados. Eval heldout sellado 60+60
×3 thetas, N2 vs N1b pareados. Wall 302.8 s. Falsa alarma
documentada: pre (ret redondeado a 3) vs post-N1b (ret
exacto) difieren ≤0.0005; causas/T/usos bit-idénticos
(0 discrepancias): batería pareada válida.

Control real: G1 N2=123 vs N1b=136, d=−0.0722 (por theta
−0.133/−0.050/−0.033); G2 N2=122 vs N1b=94, d=+0.1556
(+0.217/+0.100/+0.150). Neto +15/360. ES tradeó G1 por G2
(N2 usa más C). N2: G1 0.683, G2 0.678 — ambas bajo 0.70
y G1 regresó vs N1b (0.756). NO pasa limpio a I1.
Heldout one-shot consumido para esta config.
Siguiente N3: picos por familia (ES G1-solo vs G2-solo).

## N3. Picos por familia (n3_*.json, completo)

ES 15 gens G1-solo (N3a) y G2-solo (N3b) desde N1b; eval
one-shot heldout-2 + control N1b. Wall 306 s. Heldout-2
retirado tras la decisión.

Picos: N3a en G1 0.722 global (0.817/0.700/0.650); N3b en
G2 0.778 (0.800/0.767/0.767). Cruzados: N3a destruye G2
(0.433, olvido en 15 gens); N3b conserva G1 (0.689).
N1b en H2: 0.678/0.556 (consistente con H1).

Decisión pre-congelada: ambos picos ≥0.70 global → problema
de optimización conjunta, NO muro de capacidad (pico G1
marginal, 303 en 0.65). Hallazgo: N3b (G2-solo) es el mejor
modelo conjunto (0.689/0.778 > N2 en ambas).
Siguiente N4: receta conjunta N2 desde inits N3b, heldout-3.

## N4. Receta conjunta desde N3b (n4_*.json, completo)

25 gens receta-N2 exacta (ventanas/pools) desde inits N3b,
seed 4545. Eval one-shot heldout-3 (train_control, disjunto)
+ control N3b + check singles. Wall 342.9 s. Heldout-3
quemado tras la decisión.

N4 vs N3b: G1 0.678 vs 0.711 (−0.033); G2 0.772 vs 0.744
(+0.028). N4 NO alcanza listón conjunto (G1<0.70). PERO el
control N3b sí: 0.711/0.744 en manifiesto fresco. Decisión
registrada: avanzar N3b-final congelado como candidato I1
(selección sobre heldout-3 quema heldout-3; I1 sigue
sellado hasta apertura única). Singles: D 60/60 uso puro
en ambos (no olvidado); B 36/31 (habilidad, no selección).

## N5. Puerta I1, apertura única (n5_eval.json, completo)

N3b-final congelado + A/B-only en 200 I1 ×3 (1800 eps,
28 s). Solo entradas I1 leídas por código (assert); H
intacto. Listón ≥0.70.

N3b: 0.105/0.250/0.090 por theta, 89/600=0.1483 global.
A-only 0.1467, B-only 0.2200. El coordinador empata a
A-only y pierde vs B-only: la coordinación train-manifest
NO interpola a la geometría I1.

Veredicto pre-congelado: H UNINTERPRETABLE. Stop: H1/H2/H3
NO se abren; ningún retune mirando I1. Pendiente rethink
(decisión, no microtuning).

## Gap I1-vs-G1 (i1gap.json, offline/read-only, sin H/entreno)

Configs analíticas + gate N3b solo-forward + errores n5
existentes (sin nuevos runs I1). Hallazgo: d_gh train_seen
[35.6,104.1] vs I1 [12.0,24.9] — solape CERO; 100% I1 fuera
de soporte en d_gh (restricción registrada train beacon–haz
≥35; meta400 entero ≥35.1). Ángulo agente-goal-haz: I1
[0,0.49], 80% OOS (hazard alineado en el camino). Joint OOS:
I1 100% vs heldouts ~40% (estos dentro en d_gh, 3% OOS, y
rinden 0.68–0.78). NN-dist I1 2.28 vs ~1.9–2.0 heldouts.
Gate N3b en t0-I1 degenerado vs G1 (ej 301: 159A/1B/37D).
Dentro de I1, fallos 0.85 sin gradiente geométrico: toda la
batería está fuera de soporte.

Decisión A: I1 fuera del soporte geométrico G1 → ampliar
procedural/general (d_gh ~10–35, ángulos pequeños), ciego a
IDs I1, manteniendo Tiny original. I1 no puede volver a ser
puerta ciega para esta línea (N5 lo abrió); hará falta
batería fresca + H (sellado) para el veredicto.
