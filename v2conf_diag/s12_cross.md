# S12 cruce: S10 vs intervención única vs STU-S11 completo

Solo registros existentes (s12_eval.json + s11_eval.json). Sin
nuevos episodios, entrenamiento ni docente. Pareo por
(theta, id) sobre los 600 episodios ACQ14.

## Correspondencia documental S11-eval / S12

- Escenarios: mismo fichero s11_acq14.json; sha256 coincide con
  el congelado en S11 (caa5f959…) y S12 (verificado por assert
  en s12_run.py, re-comprobado hoy: True).
- Semillas de episodio: las de ACQ14, idénticas por fichero.
- Modelos: mismos ficheros s10_models.json (T-STU) y
  s11_models.json (STU); sha256 verificados por assert en
  s12_run.py y re-comprobados hoy (True, True).
- Política: argmax-4 del gate → skill greedy con vista
  (S4.greedy_skill), misma función en s11_run.py (greedy_a36)
  y s12_run.py (decide); skills/vistas intactas.
- Identidad de ejecución S10: repro S12 verificó causa/T/uses/
  switches contra s11_eval TSTU10 en 600/600; re-chequeo hoy
  causa+T: 0 discrepancias.
- STU-S11: registros s11_eval brazo STU 600/600 complete, sin
  re-ejecución (prohibida); correspondencia documental
  (fichero+política+escenarios idénticos).

## Primaria del cruce: intervención − STU-S11 en llegadas

| theta | INT | STU | d | + | − |
|---:|---:|---:|---:|---:|---:|
| 301 | 113 | 111 | +0.0100 | 26 | 24 |
| 302 | 134 | 123 | +0.0550 | 29 | 18 |
| 303 | 119 | 104 | +0.0750 | 39 | 24 |

Global: INT 366/600, STU 338/600, d = +0.0467 (+94−66,
440 empates). Verificado: 366−338 = 28 = 0.0467×600.

Referencia: S10 solo = 352/600. Orden: INT (366) > S10 (352)
> STU-S11 (338).

## Ocho combinaciones (A = llega, · = no)

| S10 | INT | STU | n |
|:---:|:---:|:---:|---:|
| · | · | · | 131 |
| · | · | A | 42 |
| · | A | · | 31 |
| · | A | A | 44 |
| A | · | · | 37 |
| A | · | A | 24 |
| A | A | · | 63 |
| A | A | A | 228 |

Lecturas directas: solo-INT 31+63=94 (los + pareados);
solo-STU 42+24=66 (los − pareados). En 131 episodios no
llega nadie; en 228 llegan los tres.

## Interpretación acotada

Compara continuación S10 frente a continuación S11 después
de la primera sustitución compartida (en nulos, INT=S10 por
construcción): la continuación S10 llega más (+28 neto).
No identifica qué decisión posterior concreta produce la
diferencia ni generaliza fuera de ACQ14.

## Fe de churn S12 (corrección)

75/600 mejoras = 12.50%; 61/600 empeoramientos = 10.17%;
total con cambio = 136/600 = 22.67%. Los 429 empates
excluyen los 35 casos nulos (429+35+136 = 600).
