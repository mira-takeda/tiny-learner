# Vistas por skill (PROPUESTA, no aplicada, sin probar salvo lo indicado)

Principio: cada skill observa sólo los slots activos alguna vez en su
entrenamiento + velocidad compartida; el resto se fija a la codificación
de ausencia del entreno (ceros exactos). El gate observa el estado
completo (necesita información conjunta para enrutar). Mundo intacto;
vistas sólo ante el forward de cada skill.

## Contrato auditado (entreno, 400 cfgs/familia, train_meta_v1.json)

| skill | goal S1 | hazard S2 | obj S3 | zone S4 | vel |
|---|---|---|---|---|---|
| A | 400 | 0 | 0 | 0 | siempre |
| B | 0 | 400 | 0 | 0 | siempre |
| C | 0 | 0 | 400 | 400 | siempre |
| D | 400 | 0 | 0 | 0 | siempre |

Ausencia = zeros(3) exactos (env._slot(None); también si dist<1e-9).

## Vistas propuestas

- A: vel + S1; máscara S2/S3/S4 → 0. (SIN PROBAR)
- B: vel + S2; máscara S1/S3/S4 → 0. (S1 RESPALDADA por D4+D5 en I1;
  S3/S4 sin probar; en H1/H2 S3/S4 están activos y podrían interferir.)
- C: vel + S3 + S4; máscara S1/S2 → 0. (SIN PROBAR)
- D: vel + S1; máscara S2/S3/S4 → 0. (SIN PROBAR)
- Gate: 14-dim completo, sin máscara. (Por diseño; cambiarlo alteraría
  el routing aprendido y exigiría reentreno.)

## Estado

Propuesta documentada. NO aplicada al sistema completo, artefactos
congelados intactos, gate no reentrenado. Cada máscara no respaldada
requiere su propia intervención pareada antes de afirmar efecto.
