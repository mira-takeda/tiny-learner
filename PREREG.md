# Preregistration — tiny-learner compositional transfer experiment

**Status: FROZEN 2026-10-26, before any implementation run.**
Rules in this file must not change after freezing. Results are appended only to `RUNLOG.md`
(and summarized in `CONTEXT.md`). Any deviation from this file invalidates the affected claim.

## 1. Hypotheses and claims

- **H-main (compositional transfer):** the modular system acquires atomic skills A–D from
  interaction alone (no demonstrations) and solves held-out skill conjunctions H1–H3 above
  the §5 thresholds, beating the matched monolith.
- **H-iso (interference protection, validity-conditioned):** parameter isolation (freezing)
  measurably reduces sequential interference. Testable ONLY if §7's sensitivity condition holds.
- **H-meta (curriculum value):** Meta-selected curriculum beats the matched control schedule
  at equal step budgets on H-mean or the adversarial subset.

**Allowed if won:** modularity+isolation beat a matched monolith on unseen conjunctions under
this protocol; isolation reduced interference (only with sensitivity condition met); Meta
curriculum selection added value at equal budgets. All strictly within this task family.
**Forbidden always:** "composition emerged" (we supply factorization via the shared schema);
"solved catastrophic forgetting"; generalization beyond this family; gate-retention claims
(single gate phase); novelty-of-compositeness (only the specific conjunctions are novel).

## 2. Design summary

- Env: 2D arena 100×100, NumPy, seeded; obs 14-dim fixed schema; actions discrete (9).
- Skills: 4 MLPs 14→32→32→9 (~1833 params each); gate MLP 14→32→4 (~612); total ≈ 7944.
  Gate is observation-only; action = gate-weighted blend of skill outputs; gate cannot act.
- Optimizer: ES (primary) or discrete REINFORCE (fallback) per §6. Choice frozen before
  registered training; recorded in RUNLOG.
- Arms (all sequential A→B→C→D, same per-phase step budgets):
  A1 modular-frozen + Meta curriculum (main); A2 modular-plastic + Meta (isolation ablation);
  A3 monolithic + Meta (equal params ±5%, asserted in code); A4 modular-frozen + control
  schedule (curriculum ablation).

## 3. Generator schema gen-v1 (the shared scenario space)

Constants: arena 100, accel impulse 2.0, damping 0.90, vmax 5.0, episode 200 steps,
min entity separation 10, wall bounce ×−0.5. Obs slots: S1 approach-target, S2 hazard,
S3 manipulandum, S4 destination (absent = zeros). Rewards: potential-based shaping only
+ terminal bonuses; step penalty −0.01; hazard contact −10 + terminate (where hazards exist).
- A: beacon uniform; arrival radius 6, bonus +10.
- B: hazard pursues at configured speed + noise(0,0.5); survival +0.05/step gated by
  net-displacement ≥6 over trailing 40 steps; margin bonus capped; contact −10/end.
  Train speeds 3.0–4.0.
- C: object friction 0.92, contact radius 6, push coupling 0.35; 12×12 zone at random
  edge-midpoint; delivery = 5-step dwell, bonus +10.
- D: full-width horizontal wall at y∈[40,60], gap width train 14–20 at random x;
  start below, exit above; crossing progress shaping + exit bonus +10.
- G1 (gate-train): A-config + pursuing hazard speed 3.0–3.5, spawn ≥50 from agent.
- G2 (gate-train): object + zone, no hazard; reward = approach-object shaping until first
  touch, then C shaping (sequential A→C).
- H1: C-config + pursuing hazard speed 4.5, spawn ≥40. (B+C; hazards faster than any in G1.)
- H2: G2-config + pursuing hazard 4.5 spawning 15–25 from object (proximity unseen anywhere).
- H3: C-config + full-width wall+gap between object region and zone, gap width 10–12
  (narrower than any training gap), no hazard. (D used in composite for the first time.)
- I1: G1 with beacon–hazard spawn geometry outside G1 train ranges (interpolation probe).
- Atomic-heldout: 20% configs per family, disjoint seeds, same ranges as train.

## 4. Splits and seeds

- Training seeds (5): 101, 102, 103, 104, 105 (ES sampling + train-episode sampling).
- Eval batteries: 200 episodes each, per-episode seeds from SeedSequence(2000+battery_index),
  battery_index: A=0, B=1, C=2, D=3, I1=4, H1=5, H2=6, H3=7. Batteries frozen before training.
- Sanity seeds: SeedSequence(777); never reused in registered evals.
- Control schedule order seed: 31337. Train configs per phase: 400 (Meta-ordered or control).
- Curriculum snapshot MUST be written and hashed before test configs are sampled
  (hashes + timestamps in RUNLOG). Meta generator: Meta Muse Spark
  (`muse-spark-1.3-contributor`), given schema + skill defs only (no test configs).
  Honest limitation: generator and implementer share weights; mitigation is procedural
  (snapshot-before-tests) plus the automatic leakage audit.

## 5. Thresholds (all: success rate, mean ± 95% CI over 5 seeds unless noted)

- Acquisition: ≥80% on atomic-heldout battery per skill. Check every 10 generations on
  40 train-eval episodes; early-stop after 2 consecutive passing checks; cap 400 gens,
  hard max 800 (2×) then FAIL.
- I1 (gate sanity): ≥70% on I1 battery (200 eps). If missed → gate-training failure:
  H results reported but H-main marked UNINTERPRETABLE (not failed).
- H-main: each of H1/H2/H3 ≥60% AND H-mean ≥15pp above A3 monolith AND ≥40pp above the
  best single frozen skill on that H. All three must hold.
- Retention: A1 frozen-skill drop ≤10pp from phase-end (phase-end = acquisition battery).
- Sensitivity (validity condition for H-iso ONLY, §7): A2 or A3 drops ≥20pp on ≥1 old skill.
- Reacquisition: on any dropped behavior, 10% of original phase budget must restore to
  threshold (savings ≥2× vs from-scratch gens).
- H-meta: A1 H-mean or adversarial-subset mean ≥10pp above A4.

## 6. Optimizer and fallback rule

- Primary ES: antithetic 32 pairs (pop 64), sigma 0.1, alpha 0.05, centered-rank shaping,
  Xavier-uniform init, fitness = mean return over 8 train episodes. Same for skills/gate
  (gate fitness = mean over 4×G1 + 4×G2 episodes).
- Fallback REINFORCE: Adam lr 0.01, entropy 0.01, running-mean baseline (rate 0.05),
  same per-phase env-step cap as ES (mechanical switch, no new tuning).
- Sanity env (LineWorld, separate from everything registered): 1D track [0,100], start 50,
  target 100 fixed, 3 actions {−2, 0, +2}, 100 steps/episode, obs [pos/100, (target−pos)/100],
  progress shaping + arrival bonus 10. Policy MLP 2→16→16→3, Xavier init.
- Sanity ES config: 24 pairs (pop 48), sigma 0.1, alpha 0.1, fitness = mean over 2 eps.
- PASS bar (same for both optimizers): ≥90% over 100 eval eps (sanity seeds) within
  ≤150 generations / ≤14.4k episodes AND wall-clock ≤20 min on this machine.
- Rule: run formal sanity with ES after implementation debugging. If PASS → ES frozen.
  If FAIL → switch flag to REINFORCE, re-run sanity (same bar); if PASS → REINFORCE frozen.
  If both FAIL → STOP, report, await user. Debugging runs before the formal run are logged
  as unregistered. The formal run is the first clean run of final code, labeled FORMAL in RUNLOG.
- Sanity also records throughput (steps/sec, 20k random steps, LineWorld + Arena) for
  wall-clock projection. Throughput is recorded, not pass/fail.
- Once registered A/B/C/D training begins, the optimizer choice is frozen.

## 7. N/A / inconclusive / uninterpretable rules

- Unacquired skill (FAIL after 800 gens): dependent H-tests marked N/A (H1 needs B,C;
  H2 needs A,B,C; H3 needs C,D), never zero. If a gate-train prerequisite is missing
  (G1 needs A,B; G2 needs A,C), gate phase is BLOCKED and all H-tests are N/A:
  H-main is then UNTESTABLE (reported as such).
- Forgetting-sensitivity (≥20pp drop in A2 or A3 on any old skill) is a VALIDITY condition
  for H-iso, NOT a success condition for H-main. If unmet: H-iso is INCONCLUSIVE, no
  isolation claim is made, and H-main is judged independently on its own thresholds.
- I1 < 70% → gate-training failure → H-main UNINTERPRETABLE (reported, not failed).

## 8. Baselines and conditions

- A3 monolith: single MLP 14→H→H→9 with H chosen so total params match A1 within ±5%
  (asserted in code, exact counts in RUNLOG). Same schema, data, budgets, seeds as A1.
- A2 plastic: identical code to A1 except no freezing (all parameters optimize every phase).
- A4 control: identical to A1 except curriculum = fixed stratified-uniform schedule over
  the same gen-v1 regions (seed 31337), equal step budgets.
- Single-skill checks: each frozen A1 skill alone on each H battery (expect near-chance).
- Ablations: gate-shuffle (random gate → collapse expected); hazard-mask probe (zero S2 →
  H1/H2 collapse, pure-C unaffected).
- Test-time integrity: registered evaluation runs with network disabled; harness imports
  no API client (asserted); leakage audit (train vs test manifest hashes + config-space
  distance report) must pass before results count.

## 9. Reproducibility

- Snapshots: `data/curriculum_meta_v1.json`, `data/curriculum_control_v1.json`,
  `data/test_battery_v1.json` (+ manifests with hashes). Code version = git-less snapshot:
  sha256 of `src/*.py` recorded in RUNLOG at first registered run.
- No hyperparameter tuning on any registered battery. All runs logged with seeds.

## 10. Freeze receipt

- sha256 of this file at freeze is recorded in RUNLOG.md ("PREREG-FREEZE" entry).
