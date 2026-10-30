# c6/ — C6: test de presupuesto (C4 × 800 gens, única variable)

Tras el NO-GO de C5 (one-hot+greedy no supera a C4), este test aísla
presupuesto SIN asumir que el residual es capacidad. Una variable.

## Diseño final (congelado)

- C aislada modular, seeds frescas 501–503.
- Config idéntica a C4 (fitness one-hot, training SAMPLED, n_fit=8,
  sin shaping, misma arquitectura/entorno/reward/obs).
- Único cambio: 800 generaciones (16 snaps g050–g800, 80 checks).
- 82,560,000 steps/job (asertado). Métrica GO: `heldout_pure.sampled`.
- Entrenador: `diag_train.run_job` SIN MODIFICAR.
- Artefactos por seed (9 total): `C6_{seed}.json` (+`heldout_pure`),
  `C6_{seed}_theta.npz`, `C6_{seed}_snaps.npz` (16 snaps).
- Volumen `tiny-learner-c6`, resultados en `/vol/c6/runs`.

## Lecturas pre-registradas (CONGELADAS antes del lanzamiento)

1. Si media `heldout_pure.sampled` ≥ 0.80: evidencia de que
   presupuesto era un cuello de botella relevante.
2. Si sigue ≈ meseta C4/C5 con curva g400→g800 plana: no aumentar
   más generaciones; presupuesto deja de ser explicación principal y
   el siguiente contraste es capacidad/representación.

Operacionalización de la pendiente g400→g800 (fijada aquí): por seed,
Δ = heldout_pura sampled(g800) − heldout_pura sampled(g400), ambas en
200 eps (banda de ruido ≈ ±0.07). Plateau si |Δ| ≤ 0.07 en ≥2/3 seeds;
"todavía aprendiendo" si Δ > +0.07 en ≥2/3. Los checks (40 eps, más
ruido) sólo apoyan, no deciden.

## Steps / coste / wall estimados

- Entreno: 3 × 82.56M = 247.68M steps lógicos (exactos, asertados).
- Coste: ≈$0.30–0.50 escalando del diag. Misma salvedad de facturación.
- Wall: ≈1–2h concurrentes (velocidad C4 × 2).

## Lanzamiento (Terminal del host, requiere GO explícito)

Desde `/Users/satoru/Documents/hackathon/tiny-learner`:

`python3 -m modal run c6/modal_c6.py --detach`

Salida que confirma el batch correcto (el id `fc-…` varía):

```
creating ONE C6 batch call with 3 input(s) via experimental_spawn_map: [501, 502, 503]
C6 batch call created: <fc-…> (3 inputs running detached; results land in /vol/c6/runs on volume tiny-learner-c6)
```

Verificar: "3 input(s)", "[501, 502, 503]", "/vol/c6/runs",
"tiny-learner-c6".

## Congelación

- c6_train.py sha256[12]: 153cdeb4cda9 (2026-10-28, SMOKE PASS)
- modal_c6.py sha256[12]: f202defaf34a (2026-10-28, SMOKE PASS)
- Smoke: PASS local 2026-10-28 (C6==C4 exacto incl. sampled,
  aritmética 800 gens verificada, e2e 12 gens con steps exactos +
  heldouts blended/pure). Salidas en /tmp/c6_smoke (desechables).
