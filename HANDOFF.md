# HANDOFF — Tiny Learner (2026-10-30, historial/detalle)
Fuente de verdad operativa: CROSSCHECK.md. Este archivo conserva
historial y detalle técnico de continuidad (veredicto por sonda).

## Estado (S1–S16 cerrados; A01 congelado)
- pilot3: A=1.00, B=0.83, C=0.843, D=1.00; retención frozen intacta.
- v2: I1=0.047 → composición no interpretable. v2views: I1N=0.093.
- v2conf: ACQ=0.102, peso medio de B ≈0.001, ablación-B casi sin efecto.
- S1–S16: diagnósticos del cuello de botella de coordinación (detalle abajo).
- A01 fijo: 0.647 en ACQ18 vs A-only 0.145 y B-only 0.246. Demuestra
  selección aprendida útil; NO composición general ni transferencia I1/H.
- Monolito corregido sigue pendiente. Quedan cuatro días de hackathon.

## REGLA 4 DÍAS (2026-10-30, vinculante)
No se autoriza ninguna cadena nueva de sondas. Como máximo UNA última
bifurcación científica, sólo después de que el usuario elija entre:
A) una última prueba científica precongelada que pueda cambiar el
resultado de composición; B) cerrar experimentación y pasar a
demo/submission. No preparar S17 ni otra variante de tuning.

## v2conf aceptado (2026-10-30)
Adquisición FAIL, aportación-B bajo margen, transferencia
UNINTERPRETABLE. Alcance: este currículo y presupuesto no produjeron
coordinación. No demostrado: que la solución exija memoria, ni que
toda forma de entreno bajo conflicto falle.

## Sonda fija S1 (aceptada, v2conf_diag/)
B-vista si obs[7]·DMAX<15, si no A-vista. ACQ 200×3: media sr 0.105
IC95% [0.000,0.211]; ref 0.70 NO alcanzada. Lectura: sólo esta
conmutación resultó insuficiente. v2conf FAIL intacto.

## S2 micropiloto + S2-30 (hechos, v2conf_diag/)
Piloto 3 ACQ: prim 3/3, AB 9/9. Expansión 30 ACQ (s2_30.json):
prim 30/30; AB 81/90 (por theta 26/27/28; AB-cualquiera 30/30).
Ref 0.70 superada en ambos brazos (diagnóstico). Conclusión:
soluciones encontradas mediante selección privilegiada de
skills; no capacidad del gate; fallo del gate sin atribuir.
500.8 s de 900 s. ACQ completo sigue SIN autorizar.

## S3 imitación (hecha, v2conf_diag/s3_*.json)
(b) 9/9 fallos reproducidos idénticos. 81 trazas → 2717 pasos,
partición 18/9. Estudiantes 14→32→4 Adam 500ep: eval acc
0.660/0.598/0.591 (bal 0.652/0.592/0.580), superan d15 y mostly
constante; no solo-A; nunca C/D. Offline, no control. 43.8 s de 600 s.

## S4 lazo cerrado (hecho, v2conf_diag/s4_results.json)
27 eps (9 eval×3): student 9/16/2 (por theta 1/3/5 llegadas),
const 5/15/7, d15 2/25/0, vgate 4/23/0, teacher 24/3/0. Student
nunca C/D; sus 9 caen en escenarios-docente. Hard+greedy
diagnóstico; veredictos intactos. 2.8 s de 300 s.

## S5 anchura×presupuesto (hecho, v2conf_diag/s5_results.json)
6 entrenos a 5000ep, repro-S3 bit-exacta. Ajuste mejora siempre
(→0.93–1.00); generalización inconsistente (301 empeora,
302/303 ganan con anchura, CE-eval empeora con presupuesto).
Una init por celda; sin lectura capacidad/control/obs/memoria.
78.1 s de 300 s.

## S6 lazo cerrado 4 celdas (hecho, v2conf_diag/s6_results.json)
32/500 reutilizada (bit-idéntica); 81 nuevos. Llegadas/27:
9/16/7/16 (32/500, 32/5000, 128/500, 128/5000). Presupuesto
mejora control neto 5/6; anchura mixta. C/D nunca. Offline≠
control; sin selección de modelo. 1.4 s de 300 s.

## S7 ACQ7 frescos (hecho, v2conf_diag/s7_*.json)
200 eps nuevos sellados (0 dups). 4 brazos×200×3: st325000 media
0.463 (0.365/0.485/0.540); st32500 0.332; A 0.137; B 0.218. Ref
0.70 NO alcanzada. Gana 8/9 por theta (pierde 303 vs 32/500).
Separación válida selección/evaluación (S6→S7). 143.1 s de 300 s.

## S8 demos 18 vs 36 (hecho, v2conf_diag/s8_*.json)
54 consultas docente + A36 entrenado + eval ACQ9 común. Primaria
A36−A18: +0.085/−0.045/+0.030, media +0.023 (signos mixtos,
mejora observada no robusta). Ref 0.70 lejos. 381.1 s de 600 s.

## S9 gate informado (hecho, v2conf_diag/s9_*.json)
ACQ10 sellado; dataset A36 exacto + replay; informado 5000upd.
Primaria inf−orig: −0.015/+0.090/+0.065, media +0.047, churn
~35%. Apoyo débil al paquete (x,y,hspeed,t); ref 0.70 lejos.
Privilegiado, no capacidad Tiny. 98.6 s de 300 s.

## S10 estados estudiante vs docente (hecho, v2conf_diag/s10_*.json)
ACQ11-9 + ACQ12-200 sellados; 27+659 consultas; 2 brazos×3.
Primaria T-STU−T-TEA: +0.035/+0.015/+0.040, media +0.030,
mismo signo. Ambos superan A36. Apoya recogida-estudiante.
355.9 s de 900 s.

## Gap I1-vs-G1 (DIAG offline, sin H/entreno) → Decisión A
d_gh: train ≥35.6 vs I1 12–24.9 (solape cero, 100% OOS);
ángulo 80% OOS (hazard en el camino). Heldouts dentro
(3% OOS, rinden 0.68–0.78). Gate degenerado en I1.
I1 fuera de soporte → ampliar G1 procedural (d_gh~10–35),
Tiny original. I1 ya no puede ser puerta ciega; H sellado.

## N5 puerta I1 (COMPLETO, 28 s) → H UNINTERPRETABLE
N3b 0.105/0.250/0.090 (0.148 global) vs A 0.147/B 0.220.
Muy bajo 0.70. H1/H2/H3 NO abiertos. Stop + rethink.

## N4 conjunta desde N3b (COMPLETO, 342.9 s)
N4 0.678/0.772 vs N3b 0.711/0.744. N4 falla listón G1;
control N3b lo cumple en ambas. Decisión: N3b-final
congelado a I1 (heldout-3 quemado). D 60/60 intacto.

## N3 picos por familia (COMPLETO, 306 s)
N3a/G1 0.722 (0.817/0.700/0.650), N3b/G2 0.778 (3/3 ≥0.70).
N3a olvida G2 (0.433); N3b conserva G1 (0.689). Decisión:
opt conjunta, no muro capacidad (G1 marginal). N3b mejor
conjunto (0.689/0.778). Heldout-2 retirado.

## N2 ES-duro desde N1b (COMPLETO, 302.8 s)
G1: N2 123 vs N1b 136 (−0.072). G2: 122 vs 94 (+0.156).
ES tradeó G1 por G2. N2 0.683/0.678, ambas <0.70, G1
regresó. NO pasa a I1. Siguiente N3: picos por familia.

## N1b corrección singles (COMPLETO, 6+93.5 s)
108 demos intactas + singles 600/200/clase exactos. Eval acc
0.768/0.785/0.836; D-mass>0 (assert pasado). Pre-N2 control:
G1 50/49/37, G2 35/33/26. Misma receta/seeds, desde cero.

## N1 imitación G1+G2 (COMPLETO, 891+46 s)
108 demos planner → gate intacto 5000ep final. Eval acc
0.592/0.513/0.598 (302 bajo const); sobreajuste S5-like.
B raro en train (G1-manifest fácil). D ausente → N2.
Modelos d00bc8e2 (fix métrica no cambió pesos).

## N0 planner A/C en G2 (COMPLETO, 1311.3 s acum.)
30 G2 permitidos ×3 thetas. A-only 0/90, C-only 71/90,
planner 88/90 (+19−2 vs C, 19 exigen ambas). Usa A+C
(22.7/48.0 pasos, 19.7 sw repartidos). Rama POSITIVA:
Tiny original contiene las piezas. Siguiente N1: imitar
al gate 14→32→4. Mirror G2 exacto (V1 5996 pasos).

## Consolidado A01 (fijo, sin más ajustes)
S15: +0.080 vs S10 en ACQ18 nuevo, 3/3 positivas. S16: 0.647
vs A 0.145/B 0.246. Repro venv (probada en copia limpia,
60.1 s): ver README.md raíz. Detalle: repro_s16/.

## S16 A01 vs A/B solas (COMPLETO, 44.8/90 s)
ACQ18 reutilizada. A01 0.647 vs A 0.145, B 0.246 (ref-max).
Supera a cada constante en las 3 thetas. Diagnóstico;
candidato intacto.

## S15 confirmación A01 (COMPLETO, 26.9/300 s)
ACQ18-400 nuevo. A01−congelado: 301 +0.060, 302 +0.158,
303 +0.023; media +0.080 (776/680). Las tres positivas.
Media A01 0.647, ref .70 descriptiva, no GO composicional.

## S14 alpha .01 vs .05 (COMPLETO, 209.4/300 s)
Mismo diseño S13, solo alpha. ACQ17 nuevo. Primaria A01−A05:
301 −0.040, 302 −0.020, 303 +0.150; media +0.030 (385/367).
303: A01 supera al congelado en 1/200 (+.005), prácticamente
empate, no igualdad exacta; A05 −0.145. Apoya esta reducción;
no demuestra causa de S13.

## S13 afinado ES (COMPLETO, 133.4/900 s)
Desde T-STU, retorno G1, política dura, ACQ15-train/ACQ16-eval
sellados. Primaria: 301 +0.115, 302 +0.100, 303 −0.105; media
+0.0367 (358/336). Signos mixtos; timeouts solo afinado.
Optimizador constante vs previo; no atribuir a arranque solo.
Diag train12: 301/302 mejoran (+2); 303 empeora en train y
eval; una explicación basada únicamente en generalización
no basta. Sin descartar problemas de generalización.

## S12 intervención única (COMPLETO vía extensión +180 s)
INCOMPLETE bajo tope 300 s (569/600, wall 322.5 s); extensión
operativa autorizada completó las 31 de 303 (5.3 s, 569 intactas).
Primaria int−S10: 301 +0.010, 302 +0.015, 303 +0.045; media
+0.0233 (+75−61, 35 nulos). Churn 22.67% (12.50/10.17).
Cruce INT−STU: +0.010/+0.055/+0.075, media +0.0467 (366/338).
Trazas guardadas; respaldo y tiempos en s12_backup_pre180/.

## S11 segunda ronda (hecho, v2conf_diag/s11_*.json)
Base T-STU-S10 + 234/281/250 nuevas/pareja; ACQ14 común.
Primaria STU−TEA: +0.040/+0.050/+0.025, media +0.038, mismo
signo. Ningún brazo supera a T-STU-ronda-1 (0.555/0.655/0.550).
Ronda conserva utilidad de la forma, no acumula ganancia. C/D 0
de 56255. 422.8 s de 900 s.

## Puertas intactas en todas las campañas; monolito pendiente.
## Rutas: tiny-learner/{v2conf,v2views,v2,pilot3}_runs + v2diag/ + v2views_diag/ + v2conf_diag/.
