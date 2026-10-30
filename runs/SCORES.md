# Registered scoring report — tiny-learner (attempt-3 artifacts)

Method (frozen, no changes): post-hoc scoring with frozen `src/evaluate.py`
(`make_policy` / `run_battery`), final thetas from `runs_modal/`, sampled
actions (matches frozen `retention_eval`), `seed_xor=0`. Driver
`runs/score_registered.py`, verdicts `runs/verdict.py` (rules in header);
raw numbers in `runs/scores_raw.json`, verdicts in `runs/verdict.json`.
Thresholds judged on means over 5 seeds; 95% CI = mean ± 1.96·SD/√5
(codebase convention). N/A per PREREG §7 keyed on driver FAIL flags
(compliance correction); N/A cells excluded, never zero.

## 1. Per-seed table: final full-net success rate (`*` = N/A)

| arm-seed | A | B | C | D | I1 | H1 | H2 | H3 |
|---|---|---|---|---|---|---|---|---|
| A1_101 | 1.000 | 0.000 | 0.280 | 1.000 | 0.055 | 0.000* | 0.000* | 0.165* |
| A1_102 | 1.000 | 0.000 | 0.345 | 1.000 | 0.095 | 0.000* | 0.000* | 0.285* |
| A1_103 | 1.000 | 0.000 | 0.100 | 1.000 | 0.155 | 0.000* | 0.000* | 0.175* |
| A1_104 | 1.000 | 0.000 | 0.230 | 1.000 | 0.065 | 0.000* | 0.000* | 0.200* |
| A1_105 | 1.000 | 0.000 | 0.140 | 1.000 | 0.095 | 0.000* | 0.000* | 0.140* |
| A2_101 | 1.000 | 0.000 | 0.185 | 1.000 | 0.145 | 0.000* | 0.000* | 0.175* |
| A2_102 | 1.000 | 0.000 | 0.185 | 0.675 | 0.120 | 0.000* | 0.000* | 0.235* |
| A2_103 | 1.000 | 0.000 | 0.400 | 1.000 | 0.060 | 0.000* | 0.000* | 0.320* |
| A2_104 | 1.000 | 0.000 | 0.200 | 1.000 | 0.150 | 0.005* | 0.000* | 0.175* |
| A2_105 | 1.000 | 0.000 | 0.290 | 1.000 | 0.100 | 0.000* | 0.000* | 0.235* |
| A3_101 | 1.000 | 0.125 | 0.330 | 1.000 | 0.185 | 0.000* | 0.000* | 0.265* |
| A3_102 | 1.000 | 0.000 | 0.335 | 1.000 | 0.065 | 0.000* | 0.000* | 0.280* |
| A3_103 | 1.000 | 0.000 | 0.220 | 1.000 | 0.115 | 0.005* | 0.000* | 0.145* |
| A3_104 | 1.000 | 0.000 | 0.355 | 0.970 | 0.105 | 0.000* | 0.000* | 0.250* |
| A3_105 | 1.000 | 0.000 | 0.375 | 1.000 | 0.125 | 0.000* | 0.000* | 0.250* |
| A4_101 | 1.000 | 0.000 | 0.345 | 1.000 | 0.130 | 0.000* | 0.000* | 0.190* |
| A4_102 | 1.000 | 0.000 | 0.325 | 0.990 | 0.105 | 0.000* | 0.000* | 0.320* |
| A4_103 | 1.000 | 0.000 | 0.240 | 1.000 | 0.100 | 0.000* | 0.000* | 0.150* |
| A4_104 | 1.000 | 0.000 | 0.200 | 1.000 | 0.100 | 0.000* | 0.000* | 0.165* |
| A4_105 | 1.000 | 0.000 | 0.050 | 1.000 | 0.175 | 0.000* | 0.000* | 0.085* |

All H cells N/A: skill C unacquired in 20/20 runs → H1/H2/H3 prerequisites
unmet and gate blocked per §7 (G2 needs C). Observed H numbers shown only
for transparency (H1≈0, H2=0, H3≈0.09–0.32).

## 2. Acquisition: driver flag + phase-end heldout sr (bar ≥ 0.80 on mean)

| arm | A flag/heldout | B flag/heldout | C flag/heldout | D flag/heldout |
|---|---|---|---|---|
| A1 | 5/5, 1.000±0.000 ✓ | 0/5, n/e ✗ | 0/5, n/e ✗ | 5/5, 0.920±0.157 ✓ |
| A2 | 5/5, 0.479±0.258 ✗ | 5/5, 0.005±0.010 ✗ | 0/5, 0.031±0.031 ✗ | 5/5, 0.198±0.181 ✗ |
| A3 | 5/5, 1.000±0.000 ✓ | 5/5, 0.816±0.055 ✓ | 0/5, 0.569±0.056 ✗ | 5/5, 1.000±0.000 ✓ |
| A4 | 5/5, 1.000±0.000 ✓ | 0/5, n/e ✗ | 0/5, n/e ✗ | 5/5, 0.890±0.213 ✓ |

Key divergence (descriptive): A2's train-check flags (A,B,D true) do NOT
predict heldout (≈0–0.48). Under plastic fixed-budget training, the
end-of-budget theta kept optimizing return after passing and diverged from
the mid-phase success behavior. N/A gating stays flag-based per the
correction (no re-gating); disagreements reported, not hidden.

## 3. I1 gate (bar ≥ 0.70 on A1 mean)

A1 0.093±0.034, A2 0.115±0.032, A3 0.119±0.038, A4 0.122±0.028.
Bar missed in all arms → H-main independently UNINTERPRETABLE.

## 4. Retention drops, phase_end − after_gate (log srs)

| arm | A | B | C | D |
|---|---|---|---|---|
| A1 | +0.000±0.000 | n/e | n/e | +0.000±0.000 |
| A2 | +0.033±0.453 | +0.005±0.010 | +0.021±0.025 | −0.013±0.065 |
| A3 | +0.000±0.000 | +0.816±0.055 | +0.198±0.112 | +0.090±0.162 |
| A4 | +0.000±0.000 | n/e | n/e | +0.000±0.000 |

- Retention criterion (A1 ≤ 0.10): MET, +0.000 exact — structural
  (code-asserted bit-identical frozen retention), not learned.
- Sensitivity (≥ 0.20): MET via A3/B (+0.816, per-seed all ≥ 0.745:
  catastrophic monolith forgetting of B).
- A2/A per-seed drops [+0.180, +0.725, −0.620, +0.190, −0.310]: wash —
  forgot big in 3 seeds, "improved" in 2 (gate-phase G1/G2 rebuilt
  approach behavior; final full-net A = 1.000 in A2).

## 5. Reacquisition events (bar: post ≥ 0.80 AND savings ≥ 2×)

13 events, 0 pass. Restore met 1/13 (A3_103/D post=1.000, savings 1.0);
savings met 5/13 (all A3/B, 2.25–7.25×, but post ≤ 0.435: fast relearning
to a worse asymptote); A3/C savings undefined (never acquired).
Criterion NOT MET.

## 6. A1 ablations, mean over seeds (descriptive; H is N/A)

Single-skill H1/H2 ≈ 0.000 (near-chance as expected); H3 0.056–0.112.
Shuffle: H1/H2 0.000, H3 0.110 vs full-net H3 0.193 (partial).
Mask: H1/H2 0.000 (floor — baseline already 0, uninformative);
mask@C 0.219 = full-net C 0.219 (no-op by construction: C has no hazard).

## 7. Comparisons (H-based VOID — all N/A)

I1 ≈ 0.09–0.12 all arms; acquisition identical A1 = A4 (flags A,D; no
curriculum signal); A3 acquired most at threshold (A,B,D). Chance context
(random-policy smoke): A .40, B .00, C .045, D .405, I1/H1/H2 .00, H3 .095.

## 8. VERDICTS

- **H-main: INCONCLUSIVE (UNTESTABLE)** — all H cells N/A per §7 (C missing
  everywhere, gate blocked); independently UNINTERPRETABLE (A1 I1 0.093 <
  0.70). The 60% / +15pp / +40pp bars never engage.
- **H-iso: FAIL** — sensitivity met literally via A3/B (+0.816), but the
  preregistered ablation comparison (A1 vs A2) shows no measurable
  reduction: pooled diff +0.010±0.216, CI includes 0 (per-skill A
  +0.033±0.453 null, D −0.013 reversed). No margin invented:
  measurability judged by the mandated 95% CIs.
- **H-meta: INCONCLUSIVE (UNTESTABLE)** — H N/A in A1 and A4 (n=0 both);
  the adversarial-subset disjunct is undefined in PREREG and not filled in.

Bottom line: prerequisites failed (C unacquired 20/20; B heldout only in
A3) → the compositional-transfer question is UNTESTABLE as preregistered.
Registered positive evidence: frozen arms retained perfectly (structural);
the monolith catastrophically forgot B (+0.816) while modular-plastic did
not systematically; fixed-budget end-of-budget thetas diverged from
mid-phase train checks under plastic training (A2). No thresholds changed,
no seeds dropped, no reruns.
