# v2views_diag/ — trazas gate con vistas (medición, sin entreno)

Thetas VV 301–303 (vistas), política exacta, mismas semillas/horizonte.
27 episodios fijos: G1H/I1N/H1N × primeras 3 entradas × 3 thetas.

## Resultado (trace_views.py, traces.json)

- G1H: 7/9 éxitos; w_B = 0.00 en los 9 (éxitos y muertes).
- I1N: 0/9, todas contact-death (T 9–25); w_B = 0.00 en los 9.
- H1N: 0/9, todas contact-death (T 11–20); w_B ≤ 0.02.
- w_B por banda de hazard ≈ 0.000 en far/mid/near, las 3 baterías:
  cero reclutamiento y cero modulación. Ni tardío ni insuficiente
  en mezcla: ausente.
- Lectura acotada: el fallo coincide con ausencia de reclutamiento
  de B. No declara causalidad.

## Contraste mínimo propuesto (SIN implementar)

Adquisición-bajo-conflicto: entreno de gate con vistas + G1-train
ampliado a banda beacon–haz 12–25 (geometría I1), mismo presupuesto,
streams y seeds pareadas. I1N pasa a métrica de adquisición (NO
transferencia inédita); H1N/H2N siguen siendo combinaciones no
entrenadas (veredicto); H3N igual. Verificar aportación de B (la
geometría difícil presiona evasión pero no garantiza necesidad):
(i) w_B por distancia en heldout; (ii) ablación-B contrafactual
(salida B→uniforme, mismos episodios) ⇒ caída pareada = contribución
medida de B, sin reentreno.
