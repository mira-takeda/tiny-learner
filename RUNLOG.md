# tiny-learner — Run Log

Dated record of freezes, runs, and results. Rules live in `PREREG.md` (frozen);
this file appends facts only. Never edit past entries — append new ones.

## 2026-10-26 — PREREG-FREEZE (before any implementation run)

- PREREG.md sha256: `fd7de8cd9a5abd6d718ef330957abee3b5fb320364ebfd8419adb684c1ccaa2c`
- CONTEXT.md sha256: `79d84a24afc1d7485c3868ce25063c4c47d58f54a7a8896429dee58355d18811`
- Code: none yet (docs-only commit).
- Run interpreter: `/usr/local/bin/python3.14` (numpy 2.5.4).

## 2026-10-26 — OPTIMIZER SANITY (formal)

- ES formal run: PASS. sr=1.00 (100/100 sampled eps, bar 0.90), 10/150 gens, wall 8.0s (cap 1200s).
- Policy 371 params; init seed 1234; train seed 4242; eval SeedSequence(777), sampled actions.
- Throughput (random actions): lineworld=52306 steps/s, arena=10934 steps/s.
- Pre-run smoke (unregistered): PASS sr=1.00, 2 gens, 5.1s. No code changes between smoke and formal.
- DECISION: optimizer = ES, FROZEN per PREREG section 6. REINFORCE implemented but unused
  unless a deviation is explicitly authorized.

## 2026-10-26 — CURRICULA SNAPSHOT (before any test sampling)

- scenario_space_v1.json `2bd1ff62…60f07b`
- curriculum_meta_v1.json `c0debbd2…0b254` (authored from schema only; no test configs existed)
- curriculum_control_v1.json `12bb1aaa…ed75dd` (stratified-uniform, order seed 31337)
- train_meta_v1.json `7ebd5b9b…aae85` (2400 configs, 400/phase)
- train_control_v1.json `d4d773d9…fb8576` (2400 configs, 400/phase)
- Generator bug found + fixed BEFORE any snapshot (unregistered debugging): rejection
  sampler held the agent fixed while searching far-distance regions (A/A3 failed).
  Fix: resample agent each try; TRIES 300→600; H2 hazard-agent min-sep 10 added.
  No datasets existed yet; no deviation.

## 2026-10-26 — TEST BATTERY SNAPSHOT + AUDITS (all PASS)

- test_battery_v1.json `543ec818…51f23` (1600 configs: 8 batteries x 200).
- MANIFEST.json written with sha256 of all 6 dataset files.
- Leakage: exact-match hits = 0 (hard gate PASS). Config-space distance
  (report-only): min 0.0246, p5 0.0606, median 0.1779.
- Structural: counts exact (400/phase/arm, 200/battery); seeds match
  SeedSequence(2000+idx) scheme; subranges verified (G1 train speed max 3.5 <
  H 4.5; train gaps min 14.0 > H3 10-12; G1 train beacon-haz min 35.10 > I1
  max 25); curricula reference region IDs only; scenario matches code;
  no network imports in src. Full AUDIT PASS.
- Harness smoke with random policy (unregistered debugging, no test tuning):
  A .40, B .00, C .045, D .405, I1/H1/H2 .00, H3 .095.
- Protocol gap-fills (PREREG silent; fixed before training, not deviations):
  CONTACT_R=5.0; G1-train beacon-haz>=35; H2 hazard-agent min-sep 10;
  shared train manifests per arm with streams SeedSequence(6000/6001)
  (seeds 101-105 govern ES + episode sampling).

## 2026-10-26 — TRAINING CODE FREEZE (no registered runs yet)

- Implemented: modular skills + gate, mono arm, 4-arm ES driver, retention,
  reacquisition, ablations (single-skill / gate-shuffle / hazard-mask),
  curriculum handling, seed handling, result logging.
- Params: modular 7944 (4x1833 + 612), monolith 8043 (+1.25%, within 5%).
- Toy-only selftest: 22/22 PASS. Freeze-check: PASS (datasets unaltered,
  PREREG unaltered, seeds/hypers/curricula verified, no-net imports).
- Gap-fills (PREREG silent): gate phase runs fixed 400 gens (no registered
  gate threshold); A2 gate phase keeps all params plastic; A1/A4 assert
  bit-identical frozen retention after gate phase (evals are deterministic);
  budgets = equal caps + equal data (actuals logged per run).
- Executable hashes (sha256, src/):
  audit.py a88efe2f750d436f9e247826fb7378e1a117b07d88f4178da9e4c5167123aa20
  env.py 390b82f74778e00e0d437c91ed1e24e75f5dbeae85a4cb4478e6eb8e5bb4d968
  es.py 3778e1070beef5140454941844431660c87ce986d098654e870ced0f1a5c91a0
  evaluate.py ebc2ee94bf688f67bcfd64aacbfc8d93b38d56f47336f9fbfde0aa3d60cb99a0
  freeze_check.py f3ce3421e27a88e803ab4f2f34f907fd94b3cc38ea84bcb0eefa145bfdba64ee
  gen.py b78df222de6950563c5dbedf3ccf0992c3ef733e2fcf838564be66423c1223b5
  mkdata.py a1b58eb32a802fcb93a4f534ae0f14e53c078d66894043b0ec7ebbb82af5be58
  models.py 5801ff8e47fb66ff7fb2ccf710b52bae94400d8add66b7d5911349a51f31bb55
  policies.py 13ac7f7af80af9d9ddcd0303ebd68b23dc566e757f38f4f2dea49a819ac2c7e4
  reinforce.py 2fb977f4923c378b9ce6b3b14967eafb24348ffc53042ae725492826712750c8
  sanity.py 033786bde869cb3a7a62cebf076c2f2bf4a876b7878959e578736a9608575848
  selftest.py 4dd9a2d0da52b151a917ab0baddb438abf34ee3c1d09f0f1c9b31e9d72152377
  tasks.py 4fe0c74e7a69feb42eaa6da4def906b6cb500918bed039a011225289e1b9f495
  train.py 7425dbbfe08fb8fc63eaaebad38dc9cc05b207a2367bfcc36ba1b90fbd8237fd

## 2026-10-26 — COMPLIANCE CORRECTION 2: exact-step padding (pre-run)

- Equal generations/episodes still allowed unequal env steps via early
  termination. Fix: training rollouts pad to exactly HORIZON=200 accounted
  steps per episode (absorbing: zero reward, no state change after the first
  terminal event). Return/success/task semantics unchanged (first-terminal
  outcome preserved). Eval batteries keep natural first-terminal outcomes.
- Exact steps/phase/arm/seed: 400x64x8x200 (fitness) + 40x40x200 (checks) =
  41,280,000. Asserted per phase in code (skill + gate). Per arm-seed:
  206,400,000. Matrix total (20 runs): 4,128,000,000 + conditional
  reacquisition (measurement, early-stopped, logged separately).
- Proof: rollout adds exactly HORIZON to the counter (assert steps<=HORIZON);
  phase asserts delta==PHASE_STEPS; selftest proves 1-step episodes still
  account 200 with identical return/success.
- Concurrency benchmark (toy-only, 256 eps): 1 worker 29.3s, 2 workers 19.2s
  (1.53x), 4 workers 14.4s (2.03x). Results bit-identical across worker counts
  (explicit per-episode seeds; spawn context). CHOICE: 4 workers for training
  (fastest wall-clock; ~50% efficiency; machine will be saturated).
  Projected serial ~2.5h/phase at 4040 steps/s (~180-step mean episodes) ->
  ~5 days at measured 4-way scaling; first real phase recalibrates.
- Re-verified: selftest 26/26, freeze-check PASS (18 checks).
- New executable hashes (sha256, src/): only train.py, selftest.py,
  freeze_check.py changed:
  train.py a1977d12dd1896e5e8be5e451665328f7088fd805ab4913a81c1bf2ee50dbdb6
  selftest.py 82f819cc83962a30d09fbda2414fbe5102b26cd9da168435dab9f029058b6b0b
  freeze_check.py f1e334a93c8a3edbd921588727b189aa91dc50071eba4075faed58f31e15573c

## 2026-10-26 15:06 — REGISTERED TRAINING LAUNCHED (20 jobs, 4 workers)

- Code state: e8f3137 + docs 8bf2751. No experimental code/data changed for launch.
- Command: 20 "ARM SEED" jobs from runs/jobs.txt via xargs -P 4, each
  `python3.14 -u src/train.py --arm A --seed S --data data --out runs`,
  per-job logs runs/log_A_S.txt, skip-if-complete (valid JSON present).
- sandbox-exec network-disable UNAVAILABLE here (nested sandbox blocked);
  integrity rests on frozen mechanisms: assert_no_net() every battery run +
  static no-network-imports audit (both passing). Operational note, not a deviation.
- Startup evidence (sandbox blocks ps): 4 workers STARTed (A1 101-104); logs
  present and empty at T+3min (any startup crash would write tracebacks
  immediately), master session alive. No output expected until first job
  completes (~12h/job: 5 phases x ~2.5h).
- Session caveat: runtime-owned processes end with this session. If interrupted,
  relaunch the identical command (deterministic seeds => identical results);
  completed jobs skip automatically. Prevent machine sleep for wall-clock.
- .gitignore += runs/ (operational hygiene; run outputs excluded from history).

## 2026-10-26 — OPTIMIZATION INVESTIGATION (toy-only) => 24h INFEASIBLE via execution

- Profile (6783 toy steps): policy forward 58% (matmuls 28%, softmax 22%),
  env+obs+rewards ~40%. Per-step numpy-call overhead dominates.
- Built (additive, NOT wired into driver): VecArena + VecTaskRunner (bit-exact
  vectorized env/rewards; norms verified bit-identical batched) + batched policy
  forward + vec.run_batch with scalar (exact) and batched (fast) backends.
- KEY FINDING: batched matmul is NOT bit-identical to per-row matmul on this
  BLAS (different kernels). Exact methods can therefore batch env math but must
  loop policy calls per env.
- Measured (toy, long episodes): N=8 (fitness batch size): vec-scalar 1.02x,
  vec-batched 1.82x. N=40: vec-scalar 1.22x, vec-batched 4.10x. Fitness is
  99.2% of the budget at N=8, so governing speedups are ~1.0x exact / ~1.8x
  inexact. Needed for 24h: ~4.4x code speedup (5-day base / 1 day).
- VERDICT: <=24h NOT achievable via execution optimization within the
  equivalence requirement. STOPPED per instructions; no budget changed.
- Open decision (pre-result, needs explicit authorization): (a) runtime
  amendment shrinking the registered budget, and/or (b) tolerance for
  batched-policy numerical divergence (unbuilt cross-theta batching is the only
  remaining lever, magnitude uncharacterized at ES scale).
- Equivalence proof: vec scalar-backend == scalar rollout EXACTLY (all 10
  families x 2 archs x 6 eps, identical seeds) + committed selftest
  vec-exact-vs-scalar. Scalar frozen code untouched (sed revert verified).
- Re-verified: selftest 27/27, freeze-check PASS.
- New/changed hashes: env.py 0096b5b8…, tasks.py 57c8c04e…, models.py b17e481b…,
  vec.py 85ead840… (new), selftest.py ae17d619…. Driver (train.py) unchanged.

## 2026-10-26 ~15:35 — REGISTERED TRAINING ABORTED PRE-RESULT (runtime feasibility)

- Master session terminated; completion monitor stopped. runs/ contains ZERO
  outputs (no JSON, no npz) — abort is pre-result by construction. No partial
  parameters, fitness curves, or outcomes were inspected (only file existence).
- Reason: projected ~5-day matrix runtime unviable for hackathon schedule.
  Operational correction before observing any registered result.
- Worker-death verification: ps/pgrep/pkill all blocked in sandbox. Tree-kill
  requested; runtime kills managed tree at session end regardless. Backstop:
  train.py is deterministic — any hypothetical survivor could only emit outputs
  identical to a frozen relaunch. RELAUNCH HYGIENE (binding): before any future
  relaunch, delete runs/*.json + runs/*.npz so no pre-optimization output can
  be picked up by skip-if-complete.
- Next: optimize execution on toys only (vectorization with equivalence proof).
  No registered configs until an optimization report lands. Training NOT resumed.
- From here: any implementation change = logged deviation + rerun affected runs.

## 2026-10-26 — COMPLIANCE CORRECTION: fixed equal budgets (pre-run, not result-driven)

- PREREG wording verified: section 2 requires "same per-phase step budgets"; section 8
  requires "same ... budgets" (A3) and "equal step budgets" (A4); H-meta is judged
  "at equal step budgets". Section 5's acquisition early-stop is incompatible with
  equal ACTUAL budgets (it shortens training unevenly across arms/seeds).
- Correction: every skill phase runs a FIXED 400 ES generations for all arms/seeds
  (pop 64, fitness 8 eps, identical check schedule every 10 gens incl. plastic arms
  and gate). Acquisition checks are record-only; FAIL = unacquired at gen 400.
  The 800-gen extension is removed (it cannot coexist with equal actual budgets).
  Phase output = end-of-budget theta. Savings = scratch first-pass gens /
  reacquire gens. Reacquisition = 40 gens with early stop (a measurement, not a
  comparative budget). Hypotheses, datasets, thresholds, batteries unchanged.
- Fixed budget/phase: 400 gens x 64 thetas x 8 eps x <=200 steps = <=41M env steps
  (plus 40 record-only checks x 40 eps). Measured 4040 steps/s (math-identical
  bound-net optimization, bit-verified, 1380 -> 4040). Projected ~2.5h/phase
  serial (assumes ~180-step mean episodes; first real phase calibrates);
  100 phases, 4-way parallel recommended at training time.
- Equality: single GEN_BUDGET constant, identical pop/fitness/checks/data for all
  arms (asserted by freeze-check plan test + fixed-budget selftest). Actual env
  steps vary only via behavior-dependent episode length (symmetric across arms)
  and are logged per run.
- No registered runs have occurred. This supersedes PREREG section 5's early-stop
  sentence and section 7's "FAIL after 800 gens" (now FAIL at gen 400).
- Re-verified: selftest 24/24 (incl. 2 new: bound-bit-identical,
  fixed-budget-ignores-passing-checks), freeze-check PASS.
- New executable hashes (sha256, src/):
  audit.py a88efe2f750d436f9e247826fb7378e1a117b07d88f4178da9e4c5167123aa20
  env.py 390b82f74778e00e0d437c91ed1e24e75f5dbeae85a4cb4478e6eb8e5bb4d968
  es.py 3778e1070beef5140454941844431660c87ce986d098654e870ced0f1a5c91a0
  evaluate.py 6efd31ad1d7ec6a40cfa66883138df0cccab51be47b74aa3d4a1256aaae0ec8f
  freeze_check.py 80912515b919a330fb5d3e70e30326e1a3781d8066cee0cebb7383956d6d8043
  gen.py b78df222de6950563c5dbedf3ccf0992c3ef733e2fcf838564be66423c1223b5
  mkdata.py a1b58eb32a802fcb93a4f534ae0f14e53c078d66894043b0ec7ebbb82af5be58
  models.py 276d8762b479dc122af88a71b280c3517872bfd89bdf5745df9bfc36beab3643
  policies.py 13ac7f7af80af9d9ddcd0303ebd68b23dc566e757f38f4f2dea49a819ac2c7e4
  reinforce.py 2fb977f4923c378b9ce6b3b14967eafb24348ffc53042ae725492826712750c8
  sanity.py 033786bde869cb3a7a62cebf076c2f2bf4a876b7878959e578736a9608575848
  selftest.py 14ab22e176da77f12c0711ca779b8dfa61293cf97346b58ce62a20c66ecdb475
  tasks.py 4fe0c74e7a69feb42eaa6da4def906b6cb500918bed039a011225289e1b9f495
  train.py 7f2bf51eec7c02001f6e6e92c9f5599ae7e3a9b91fb299e3617c98ec47d49fc4
