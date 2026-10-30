# diag/ — post-hoc diagnostic campaign (v1 CLOSED, untouched)

Isolated C/B skill training to explain v1 failures. Separate from
`runs_modal/`, PREREG, and v1 records. Seeds 201–203 (fresh streams).

Jobs (x 3 seeds = 24): C0 (C isolated, v1 mechanics), C1 (C0 + 32-ep
fitness), C2 (C0 + greedy), C3 (C0 + delivery-staged shaping: pickup +0.5,
+1/dwell-step 1–4, frozen +10 at 5; success def unchanged), C4 (C0 +
one-hot fitness, mismatch test), B0 (B modular isolated), B1 (B mono
isolated, fresh init), B2 (B0 + one-hot fitness). 400 gens each;
C0/C2/C3/C4/B0/B1/B2 41.28M steps, C1 164.16M steps. Budget ≈ $1.65–2.89,
wall ≈ 7h concurrent (C1 longest).

- `diag_train.py` — isolated job trainer (mirrors frozen mechanics;
  equivalence-tested in `smoke_diag.py`)
- `modal_diag.py` — Modal launcher (app/volume `tiny-learner-diag`,
  one `experimental_spawn_map` batch call, clean-subprocess workers)
- `smoke_diag.py` — local smoke (equivalence + tiny end-to-end per path)
- `oracles.py`, `beam.py`, `instrument.py` — audit controllers + tracing
- `audit_c.py`, `audit_b.py`, `audit_c2.py` — local audits (+ JSON outputs)
- Remote results land in `/vol/diag/runs/` on volume `tiny-learner-diag`
  (never in `runs_modal/`).

Launch (host terminal, needs explicit GO):
`python3 -m modal run diag/modal_diag.py --detach` from project root.
