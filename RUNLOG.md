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

## 2026-10-26 — MODAL PREP BLOCKED (sandbox cannot reach Modal API)

- Found: modal 1.6.2 under py3.11 (Framework python); ~/.modal.toml has
  [lordtakeda] profile with token keys (values never printed). No `modal` on
  PATH; CLI works via `python3.11 -m modal`.
- `modal volume/app list` => "Could not connect to the Modal server" (both).
  curl via sandbox proxy: github.com/meta.com OK, but modal.com TLS returns
  nothing and api.modal.com fails entirely. Verdict: sandbox egress proxy does
  not allow Modal endpoints. Auth state UNTESTABLE from here (never connected).
- Prepared but UNVERIFIED: modal_app.py (env/package/selftest/toy/persist funcs
  + uncalled run_registered; numpy==2.5.4 pinned; 1 CPU/1GiB; volume
  tiny-learner-runs). Left UNCOMMITTED pending first verified run.
- Pricing gathered (2026 sources, for later costing): CPU $0.0000131/core-s,
  mem $0.00000222/GiB-s, per-second billing.
- No registered workloads touched. No installs, no new envs (used existing
  py3.11 for Modal driving per python-env guidance).

## 2026-10-26 — MODAL WRAPPER READY; HOST VERIFY PENDING (sandbox blocked)

- Host terminal confirmed working (`modal volume list` OK outside sandbox).
  Sandbox proxy remains the only blocker, so verification runs via user-pasted
  host output. Wrapper validated locally as far as possible without cloud:
  imports clean, dispatch works, image/volume defs lazy (no network at import).
  Fixed for modal 1.6.2 API: Mount.from_local_dir gone => code/data baked into
  image via add_local_dir(copy=True) = immutable snapshot (better for freeze).
- Local baselines captured BEFORE seeing any remote output: toy expectation
  (64 eps, seed_base 5000, saved); src+data hashes re-confirmed == frozen.
- PRE-STATED interpretation rules for the host verify paste: (1) remote file
  hashes MUST equal frozen exactly (copy check); (2) remote selftest MUST exit
  0 all-PASS; (3) Modal-vs-Modal toy reruns MUST match exactly (this is the
  determinism that counts); (4) local-vs-Modal toy: match preferred, ulp-level
  mismatch acceptable ONLY if (1)-(3) hold (OpenBLAS-vs-Accelerate note) — in
  that case ALL registered runs happen on Modal only (single environment);
  (5) volume write/read roundtrip must succeed; (6) throughput/cost projected
  from measured Modal steps/s at published rates, computed not assumed.
- Single host command (toy-only; run_registered NOT invoked by any action):
  cd /Users/satoru/Documents/hackathon/tiny-learner &&
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m modal run modal_app.py --action verify

## 2026-10-26 — MODAL WRAPPER FIX: Python 3.14 base image (pre-run)

- Host verify attempt FAILED at image build (no workload executed anywhere):
  base image inherited Python 3.11 from the local client, but pinned
  numpy==2.5.4 requires Python >=3.12.
- Fix (operational, wrapper-only): modal.Image.debian_slim(python_version="3.14"),
  matching the frozen local training interpreter (3.14). NumPy stays 2.5.4.
  Everything else unchanged. Import-validated locally (pin accepted).
- No registered workload has run anywhere. No budgets/seeds/thresholds touched.

## 2026-10-26 — MODAL VERIFY RESULTS + vec-exact DIAGNOSIS (pre-GO)

- Remote results (user-pasted summary): Python 3.14.2, NumPy 2.5.4, toy
  run1==run2 EXACTLY (Modal determinism HOLDS), 7218.4 steps/s, peak RSS
  103 MB, persistence round-trip passed, nothing registered launched.
  Remote selftest: 26/27 (ONLY vec-exact-vs-scalar FAILS; all scalar-path
  checks pass remotely).
- ISOLATION PROVEN (local grep): vec.py/VecArena/VecTaskRunner/run_batch/
  batched_* referenced ONLY in vec.py, class definitions, and selftest.
  train.py/evaluate.py/sanity.py/gen/mkdata/audit contain ZERO references:
  the registered driver CANNOT execute vec code.
- MECHANISM (local analysis): the ONLY batched-vs-scalar kernel-sensitive op
  in the vec path is norms (scalar np.linalg.norm vs vec sqrt(sum)). All else
  is elementwise-identical or shared scalar calls. Locally both agree (20k/20k
  micro-test); on Modal's BLAS they can differ ~1 ulp. The test asserts EXACT
  return equality, so ~1e-16 reward diffs fail it with NO flips/trajectory
  divergence needed (local probe: 1-ulp obs noise flips 0/20000 choices).
- PREDICTION (pre-stated, falsifiable): remote norm micro-test shows
  mismatches ~1e-16; lockstep first divergence at step 1 in obs/reward at
  ~1e-16 with identical actions/success. If norms are clean on Modal yet vec
  diverges, mechanism is elsewhere (tanh/libm?) and gets fresh diagnosis.
- vecdiag toy action added (per-family first-divergence + norm micro-test);
  lockstep logic validated locally (CLEAN). Host command below.
- PROJECTIONS from 7218.4 steps/s (1 CPU): per phase ~1.2h central [0.8-1.6h
  for 100-200-step mean episodes]; per job ~6h [4-8h]; 20 concurrent wall
  ~= one job ~6h [4-8h] + overhead. Cost: 432,000 CPU-s x ($0.00000655 CPU
  [ASSUMES 1 vCPU = 0.5 billed core — flag] + $0.00000222 mem) ~= $3.79
  central [$2.5-$5.1]; storage negligible. Keep 1 CPU + 1 GiB as-verified
  (103 MB peak = 10x headroom). Timeout 86400s >> 6h job.
- RECOMMENDATION (not executed): delete dead vec path (vec.py + VecArena/
  VecTaskRunner/batched fns + vec test) — documented dead-code removal, not
  test deletion (test expectation of cross-BLAS bit-identity is what's wrong);
  vec work stays in git history. Alternative: keep code, scope test local-only
  (weaker). Awaiting vecdiag confirmation + GO.

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

## 2026-10-26 — NORM HYPOTHESIS FALSIFIED; vecdiag2 READY (pre-GO)

- Remote vecdiag (user-pasted): norm_micro 0 mismatches/max_abs_diff 0.0;
  all tested families A/B/C/D/G1/G2/H1/H2/H3/I1 matched scalar vs vec exactly;
  all lockstep diverged=false. The norm/ULP mechanism is FALSIFIED; its
  prediction did not hold. That explanation is dropped.
- New candidate causes: (a) value-dependent divergence elsewhere in vec path
  (the failing case uses DIFFERENT theta/rng-seeds/family/cfgs than vecdiag's
  replication); (b) in-process order dependence (earlier full self-test ran 27
  tests before vec; standalone broadband/harness code paths may set state);
  (c) transient remote non-determinism/flake (check: does the full-suite FAIL
  reproduce at all?).
- vecdiag2 prepared (toy-only): new src/diag.py = exact replica of selftest's
  vec check (same theta seeds 31/32, fams A/B/H1, ep seeds 200-203, max 200
  steps, counters [0]), run 3x in the same Modal container, each failure
  followed by SAME-container first-divergence lockstep. NOT imported by
  train.py/other src (verified: diag unreferenced outside itself); selftest.py
  and vec code UNCHANGED.
- Local validation: exact_vec_pass 3/3 green; modal_app imports with vecdiag2
  action. No registered run launched. Awaiting host vecdiag2 output to decide:
  value-dependent (lockstep names quantity) vs context-dependent (isolated pass
  but full suite fails) vs flake (neither reproduces).

## 2026-10-26 — ISOLATED VEC REPRO PASSED 3/3; selftest2x READY (pre-GO)

- Remote vecdiag2 (user-pasted): exact selftest-vec replica passed 3/3 in one
  Modal container. The original vec-exact-vs-scalar FAIL is NOT reproducible in
  isolation. Value-dependent divergence is RULED OUT for the exact failing
  inputs. Remaining: full-suite context dependence vs transient.
- selftest2x prepared (toy-only): runs UNCHANGED selftest.main() twice in the
  same Modal process; only the harness resets the module-level PASS accumulator
  between runs (selftest.py, vec, train.py, budgets, seeds, thresholds all
  untouched). Run 0 replicates original verify conditions exactly; run 1 tests
  in-process repeatability. Full per-check lines captured per run.
- Local validation of the twice-harness: 27/27 + 27/27, single-run wall ~1m54s
  (remote expected similar or faster; 1800s timeout). Wrapper imports with the
  selftest2x action. No registered run launched.
- Decision rule: both runs 27/27 => original FAIL classified transient /
  non-reproducible, documented as such; vec FAIL in run 0 and/or run 1 =>
  context-dependent, next step is bisection over preceding tests (not yet
  authorized). GO assessment follows the pasted output.

## 2026-10-26 — ORIGINAL vec FAIL CLASSIFIED TRANSIENT (pre-GO)

- Remote selftest2x (user-pasted): run 0 27/27 exit 0, run 1 27/27 exit 0,
  vec-exact-vs-scalar PASSED both times in the same Modal process running the
  UNCHANGED full suite. Combined with vecdiag (norm exact, all fams match) and
  vecdiag2 (exact isolated repro 3/3 pass), the original single FAIL reproduced
  in 0 of 6 subsequent remote executions of equivalent-or-identical code.
- CLASSIFICATION: transient / non-reproducible. No code change is attributed as
  a fix (nothing in the vec path was modified at any point). The dead-vec
  cleanup proceeds as planned hygiene (the vec path was always unused by the
  registered driver), not as a fix for this FAIL.
- Registered path unaffected throughout: scalar rollout/train/eval verified
  bit-deterministic remotely (toy run1==run2 exactly) and 26/26 on all
  non-vec checks in every remote run.

## 2026-10-26 — FINAL CLEANUP: tree restored to frozen e8f3137 (pre-GO)

- Restored src/env.py, src/models.py, src/selftest.py, src/tasks.py to e8f3137
  blobs; deleted src/vec.py + src/diag.py (dead/diag-only; kept in history).
  Rewrote modal_app.py as launch-only layer: image py3.14 + numpy 2.5.4,
  run_registered(arm, seed) -> /vol/runs, launch_all spawns 20 jobs (optional
  arm/seed filter). All toy/diag/verify actions removed (kept in history).
- Verification: 22/22 src+data files blob-identical to e8f3137, none
  missing/extra; PREREG.md sha256 fd7de8cd...aa2c MATCHES frozen record;
  data/ 6/6 match MANIFEST.json; zero vec/backend refs in src+data; train.py
  rollout = scalar TaskRunner + HORIZON accounting (no forward pass in
  absorbing tail); launch manifest == train.ARMS/SEEDS; wrapper imports on
  host driver; selftest 26/26 + freeze-check PASS locally.
- Artifacts/concurrency: each job writes /vol/runs/{arm}_{seed}.json +
  {arm}_{seed}_theta.npz (distinct names, vol.commit); 20 spawn() calls, one
  container each, 1 CPU/1GiB, timeout 86400s. Local runs/ cleared (5 stale
  pre-result files from aborted launch); remote jobs read only /root/tl/data.
- Projections from 7218.4 steps/s: <=1.6h/phase, ~7-8h/job (5 phases + evals),
  ~8h concurrent wall; cost ~$5.05 flagged (0.5 core/vCPU) / ~$8.83 unflagged.
- NOT LAUNCHED. Awaiting final GO for the host launch_all command.

## 2026-10-26 — REGISTERED LAUNCH GO (Modal, 20 jobs, commit ab640ad)

- GO received. Pre-launch audit record written to runs/launch_ab640ad.json
  (14 src + 8 data sha256, modal_app 23b85478..., local py/numpy versions).
  Frozen experiment == e8f3137 blobs (verified); operational layer modal_app.py
  only. Curriculum/data manifests: data/MANIFEST.json (6 files) + frozen
  train_meta/train_control/test_battery (unchanged since snapshot).
- Launch command (host): modal run modal_app.py --action launch_all
  (20 spawn, 1 CPU/1 GiB each, timeout 86400s, out /vol/runs).
- Expected: ~7-8h/job compute, ~8h concurrent wall + image-build/startup
  overhead (~10-20 min); completion window ~9h from launch. Cost ~$5-9.
- NOTE: per-job artifacts + stdout summary appear only at job END
  (run_arm_seed writes once); /vol/runs stays empty until first completion.
  Live health = Modal dashboard / container status only. Job IDs + start/end
  timestamps to be recorded here when pasted back. Failures (if any) preserved
  exactly before any action.

## 2026-10-26 18:09 — ALL 20 JOBS LAUNCHED (Modal, commit ab640ad)

- Host launch completed 18:09 local (18.4s wall): image built in ~4s
  (im-shAOEcyuTTaXrwH3xcjvXb, im-EEoRBXIkYp4mTESxFqTFv9), app
  ap-CW89pkvUZOoZ1KwBc7H9S3, all 20 (arm, seed) pairs spawned with IDs
  fc-01M5XAZ1... through fc-01M5XAZ3... (full map in runs/launch_ab640ad.json,
  ignored). No errors at launch. Local entrypoint exited cleanly.
- Expected completion window: ~01:00-03:30 local 2026-10-27 (~7-9h wall).
  Next: user checks dashboard/volume; on completion, download /vol/runs (40
  files expected: 20 JSON + 20 npz) and paste `volume ls` for confirmation
  before scoring. Then PREREG analysis on raw results.

## 2026-10-27 — CLOUD LAUNCH FAILURE (operational, zero artifacts)

- Host evidence: local entrypoint exited after 18.4s; app logs show
  "Stopping app - local entrypoint completed" + "Runner terminated"; Volume
  tiny-learner-runs completely empty the following morning.
- Cause: plain `modal run` (no --detach), so the ephemeral app stopped with
  the launcher and all 20 spawned FunctionCalls died with it. No registered
  result could complete: completion without artifacts is impossible
  (run_arm_seed writes JSON+npz + vol.commit before returning), so empty
  volume <=> zero completions. No experimental outcomes exist from this
  attempt; no code/budget/seed/threshold issue involved.
- Relaunch plan (identical, from ab640ad): verified (a) src/data/modal_app
  byte-identical to ab640ad; (b) modal client 1.6.2 supports
  `run -d/--detach` ("Don't stop the app if the local process dies or
  disconnects"); (c) job list recomputed 20/20 unique (A1-A4 x 101-105).
  Host must confirm volume empty immediately before relaunch, then run the
  detached command. New FunctionCall IDs to be recorded on paste-back.f5599ae7e3a9b91fb299e3617c98ec47d49fc4

## 2026-10-27 — DETACHED RELAUNCH FAILURE (operational, zero results)

- Detached relaunch (same launcher + --detach; tree = ab640ad plus
  doc-only uncommitted changes, baked src/data identical to attempt 1)
  created all 20 FunctionCalls (A1-A4 x 101-105) under app
  ap-ArwriEWEAlk85lcRu5gV98. The CLI printed the detached-mode notice.
- ALL 20 failed within seconds at experiment entry: run_arm_seed() ->
  assert_no_net() raised because the Modal worker had already loaded
  socket, urllib, ssl, http before user code ran. Zero training steps,
  zero evaluations, zero artifacts from any job; /vol/runs empty, no
  registered outcomes. No code/budget/seed/threshold cause: purely the
  entry gate vs the cloud runtime.
- Post-checks from a fresh terminal (old terminal left streaming the
  failure logs, then abandoned; nothing launched from it afterwards):
  `modal container list` -> no active containers; `modal app list` ->
  ap-ArwriEWEAlk85lcRu5gV98 stopped, 0 tasks. Nothing running, nothing
  consuming compute. The blocked old terminal was draining failure logs,
  not running training.
- Why verify missed it: remote selftest 27/27 never executes the entry
  gate (selftest.py imports only constants/run_es_phase/rollout from
  train; it never calls run_arm_seed/run_battery/assert_no_net), so the
  Modal-preload clash was unobservable until the first registered call.
- Local mechanism reproduction (throwaway /tmp script, frozen src):
  clean interpreter imports train with forbidden-delta [] and the gate
  PASSES; preloading exactly socket/urllib/ssl/http reproduces the
  attempt-2 RuntimeError signature verbatim.
- FunctionCall IDs of this attempt were NOT captured (old terminal
  output not saved). Next launcher/probe run must tee its output to a
  file. Attempt-2 IDs stay unknown.
- Status: SECOND operational zero-result failure. NO relaunch until (a)
  the entry gate is fixed in a semantics-preserving way (pre-result
  correction, logged with hashes) and (b) a minimal detached probe
  proves the fixed gate passes remotely AND multi-spawn detached fan-out
  survives parent exit (Modal 1.6.2 warns it keeps only the last
  triggered function alive after disconnect; attempt 2 could not test
  this because all calls failed while the parent was still attached).

## 2026-10-27 — PREFLIGHT PASS (clean-subprocess gate + single-call survival)

- 3/3 toy inputs wrote done_{i}.json AFTER the local client was killed
  with Ctrl-C mid-run: F5 PASS. ONE experimental_spawn_map call survives
  a real disconnect. No further probes authorized or needed.
- done markers imply the gating asserts passed in every input (marker
  writes sit after them in code): frozen gate exits 0 in the clean
  subprocess (F2), frozen --plan reports exact budgets (F3), train.py /
  evaluate.py hashes match frozen values (F6). Code-gated inference,
  not pasted output.
- F1 parent-preload sets and F4 parent-gate outcomes were recorded in
  the markers but their contents were not pasted; values not registered.
- Verdict: launch pattern approved for the 20-job registered run.
  No src/ changes made or needed; tree stays byte-identical to ab640ad.

## 2026-10-27 — REGISTERED LAUNCH, ATTEMPT 3 (single batch call, 20 inputs)

- Pre-launch mechanical checks ONLY: tiny-learner-runs volume EMPTY
  (host `volume ls`); src/ + data/ 22/22 byte-identical to ab640ad
  (host confirm + local re-check, no diff vs ab640ad). Nothing else run.
- Launcher: modal_app.py a15420b8... (clean-subprocess workers) via
  experimental_spawn_map. App ap-KclxUNJF1x54xqqAdbIZ4Y. Single batch
  call fc-01M5ZCAF98TR6KNZDHHTFQRF0N with 20 inputs (A1-A4 x 101-105).
  Results target /vol/runs (40 files expected: 20 JSON + 20 npz).
- Local client exited normally after creating the call. No relaunch
  from here; nothing further launched.
- Next: await completion (~8h concurrent wall per prior projection, plus
  startup overhead). On completion paste `volume ls` for confirmation
  before scoring. Then PREREG analysis on raw results.

## 2026-10-27 — ATTEMPT-2 ID AUDIT CLOSURE (no new execution)

- The 20 FunctionCall IDs of the failed detached relaunch (app
  ap-ArwriEWEAlk85lcRu5gV98; all failed at assert_no_net, zero results)
  were recovered from the saved launcher output and are registered here,
  superseding the "IDs unknown" note in the failure entry above:
  A1_101 fc-01M5YVE1M2C4774AJM8F4V7C2Z
  A1_102 fc-01M5YVE1QKMKWQYCZBKMRA9BC0
  A1_103 fc-01M5YVE1VAHSS76G7HHQ2C6V1C
  A1_104 fc-01M5YVE1Z6B6ZCME43J8X6EC0Z
  A1_105 fc-01M5YVE23174M65N8HZECFZHJ2
  A2_101 fc-01M5YVE26T0FR3CDBKM8ZPZ8J3
  A2_102 fc-01M5YVE2AN2WKST13Z1M28Y1WX
  A2_103 fc-01M5YVE2EM1K8VD519WRVH6PZ3
  A2_104 fc-01M5YVE2JQHFVV7YS04WQ5JH4S
  A2_105 fc-01M5YVE2PNQFBJB852H3Z55WGV
  A3_101 fc-01M5YVE2TH4CP24NMHN412W4HC
  A3_102 fc-01M5YVE2YSH9879G4MXA5476CT
  A3_103 fc-01M5YVE32M8CEV65YW8ZEP09BD
  A3_104 fc-01M5YVE36FFTGDBE40YWG6JDEQ
  A3_105 fc-01M5YVE3AA36F8PR8V6DJYTH86
  A4_101 fc-01M5YVE3DY0XS8A315PSZNZTTF
  A4_102 fc-01M5YVE3HJ6T1A3HJN3241T8FD
  A4_103 fc-01M5YVE3NG2ACPYKAAVBR5RCMB
  A4_104 fc-01M5YVE3SG40B2BFKSTFSYWQTY
  A4_105 fc-01M5YVE3X79NPKNCQVPSETFF2V
- Same map preserved in runs/launch_attempt2.json (ignored dir).
- Attempt-2 audit CLOSED. Attempt 3 untouched and still running; nothing
  relaunched, nothing else changed.

## 2026-10-27 — REGISTERED RUN CLOSED (attempt 3): 20/20 jobs, 40/40 artifacts

- Host report: all 20 jobs finished; last result A2_103; no active Modal
  containers; 40 files downloaded to runs_modal/ (20 JSON + 20 _theta.npz,
  A1-A4 x 101-105 complete).
- Local integrity (runs/verify_artifacts.py, READ-ONLY on runs_modal/):
  PASS — exact 40-file set, JSON/NPZ pairs, filename<->content arm/seed
  match, 400 gens/phase, 41,280,000 steps/phase, 206,400,000 total per
  job, retention/reacquisition structure, theta shapes (7944 modular /
  8043 mono) finite. Raw artifacts untouched; manifest saved to
  runs/artifacts_manifest.json (ignored dir).
- sha256 of the 40 registered artifacts (BEGIN-HASHES):
  A1_101.json 4c9e46b178d2626b899ce5d96bf25a702e8db2bb6e57c5dbb147d540cbae68bf
  A1_101_theta.npz 4d5ef4603e53be15cea9ee8f7049bf2277ee25899f033248d4583a7c228b3127
  A1_102.json 4ada052fa2c4cca49d1d8583068b731872bb6560a39ecc1c46764ea48e4d92e4
  A1_102_theta.npz 6d16f5191b4e8a1a54a90e8615c0bc055ddee466d72848eb3cfb96b3057ccf66
  A1_103.json a9dd43b0d5b8f9be04c7f58835b714660c0dfb6b1c800e1b904f2deaa8655fe8
  A1_103_theta.npz 9f76940e6d505c9666428571745987be22824e38dceb74d1e36e3227a6cb0647
  A1_104.json c000cb58ab782bd829677d0dd8a452270fcfe5397125cc377a2b8f1377d7d2ba
  A1_104_theta.npz 901b6b4e52106191afd6f5c34e8ac8cf20ce88a4b161fe2e67a737653ed0fd2b
  A1_105.json 070e46b667e61316097a36075b857228ed91d6cfc144c2828cef40464b58ea60
  A1_105_theta.npz d33d0d51f8527dc206e5dc770dd360ec35d6853fe926f0a3bf86bfe7984a2573
  A2_101.json 8853a51c5e2a5a6e502f0252063af34bfcc920594a4fdafc8d29b59cffc7ec0c
  A2_101_theta.npz 78b3d42870fa00c29b74bc482b47d53b613722e68573867b7ebafc822c060241
  A2_102.json b3ac1ecb62061b613855955a56c7caed5c0beb866cb21f0859a1af1bc5de71a9
  A2_102_theta.npz 56779b3932fa4bf5732997b35b3222fbd212a3697fdc17a9ba69960e4a7ce18b
  A2_103.json 2ce94c494101dac4783af96c11c151260ea28ad5d5a23e2ce38488b1276657e6
  A2_103_theta.npz 29487208d30b73c4b45c3b8e6a6106643d094ae9f07297ad40ce044a7143d6d8
  A2_104.json 8273599a318666840adef341113346cc632c65ddb6a5fadae1f835cc8f10446d
  A2_104_theta.npz b8bb3a8b780655e8b65dab5fe3d83535ec39521192bac44fe436648218a7d9e5
  A2_105.json 5ee782ad813961555f2595441b404760862ee965f3bcabb3427a8a10b6a195fa
  A2_105_theta.npz 34bd23dcd686090b83b861dc814fd14a295acb59df0ede91aa6ee483f29fb003
  A3_101.json d6703f81d1e8af0beb995647e8bc4e09e3ffcbde08480411a07ba6310def6f2e
  A3_101_theta.npz efd146dce3954bc06949584baa87f1d1a2203d83d60665efbfe2b9db2379d934
  A3_102.json daca2124cec4a0be8095b83b5f7ca8feea12d63b710318c0f48f4cca82378c85
  A3_102_theta.npz 3030d614be76e2625fa45444eedddc3e16ee9791f762de18b87f6599cef94990
  A3_103.json fabf7700f7c2f6906d6d8782c9e8d1f1a2b6878f723e217ff96a3b8beba65b62
  A3_103_theta.npz 9529bf0c15a6758a7f6ab70b79c1303321f87c23eb496c887e328b7fc626132e
  A3_104.json 0262834e099a4289a99a594a9ce431ac9f97d26986780550fccff78e00056c89
  A3_104_theta.npz 72480939127ad5e439de3f505089c803801d818a648fbdb2c14c942569636ac8
  A3_105.json 986a52f9d8ee47c24c1b2d2aa2eb3a40fdf3da81e22ab39707eef327f6cce28f
  A3_105_theta.npz 69b159fe8e1c77711f68cc7e4f5e6d3d82e019d4cc232c7c4174c98a669abbbd
  A4_101.json d907995b24f33446c86a5b718ca0c51a5723ed642e41e9a5b7c08c25069aafb5
  A4_101_theta.npz 73242af7dd262cffa3b7c4053c312d84a31376e5068e44e19490a6ab72c1c2e2
  A4_102.json 12984b3bfe1599be6cbefc1f2835b2cd0c10e61f5b421a9d4899b5e0410ed73a
  A4_102_theta.npz bb137ffaf1cfb30a8253ed2a6fa4696be13f07113c5cde564facc9e38f39b922
  A4_103.json eb21db8333adaa335def413bfcabc087ce5398a19a0a397dd1468922217eabb8
  A4_103_theta.npz 6deb92f6d40e630d3048b5ccb47c02e2127e987be10c7a0135c32e5ddc807a5a
  A4_104.json bb95e341ed51397307cf0b57e21250ae7853d1736f0a8bb90bbd821c6b68c43c
  A4_104_theta.npz 2385b2f817f47d33701101c823cf6e3353155da247e3ec86b7bc3cc9f3319517
  A4_105.json d66bd3e2d10e16baaa84ca130f37d0169739859785048ca17ae70be88428ac10
  A4_105_theta.npz 17e147a15e8d17327cfe887409a974ba7cac5333e29d548a6252c30f0cdc91bc
  (END-HASHES, transcription verified against the manifest programmatically)
- Acquisition flags observed in logs (driver FAIL definition; PREREG N/A
  rules apply at scoring): A1/A4 acquired A,D only (B,C FAIL all 5 seeds);
  A2/A3 acquired A,B,D (C FAIL all 5 seeds). C unacquired in 20/20 runs.
  Reacquisition events: A2/A in 3 seeds; A3/B,C in most seeds + D once;
  none in A1/A4. NO verdicts here — scoring next with the same frozen
  code, no threshold/seed changes, no reruns.

## 2026-10-27 — REGISTERED SCORING COMPLETE (250 batteries, verdicts below)

- Method: runs/score_registered.py (sha 826aba0ff794) post-hoc, READ-ONLY
  on runs_modal/ + data/; frozen src/evaluate.py make_policy/run_battery;
  sampled actions (matches frozen retention_eval); seed_xor=0; final
  thetas. 20 thetas x 8 batteries mode normal + A1 ablations (4 skills x
  H1-H3, shuffle x H1-H3, mask x H1/H2/C). Raw: runs/scores_raw.json.
  Verdicts: runs/verdict.py (sha 90a6cd721e69), rules in header:
  driver-flag FAIL/N/A (correction), thresholds on means over seeds, CI =
  mean +- 1.96*SD/sqrt(5) (codebase convention), H-iso "measurably" =
  CI-distinguishable (no invented margin). Full tables: runs/SCORES.md.
- Per-seed final full-net sr, A B C D I1 H1 H2 H3 (* = N/A per 7):
  A1_101 1.000 0.000 0.280 1.000 0.055 0.000* 0.000* 0.165*
  A1_102 1.000 0.000 0.345 1.000 0.095 0.000* 0.000* 0.285*
  A1_103 1.000 0.000 0.100 1.000 0.155 0.000* 0.000* 0.175*
  A1_104 1.000 0.000 0.230 1.000 0.065 0.000* 0.000* 0.200*
  A1_105 1.000 0.000 0.140 1.000 0.095 0.000* 0.000* 0.140*
  A2_101 1.000 0.000 0.185 1.000 0.145 0.000* 0.000* 0.175*
  A2_102 1.000 0.000 0.185 0.675 0.120 0.000* 0.000* 0.235*
  A2_103 1.000 0.000 0.400 1.000 0.060 0.000* 0.000* 0.320*
  A2_104 1.000 0.000 0.200 1.000 0.150 0.005* 0.000* 0.175*
  A2_105 1.000 0.000 0.290 1.000 0.100 0.000* 0.000* 0.235*
  A3_101 1.000 0.125 0.330 1.000 0.185 0.000* 0.000* 0.265*
  A3_102 1.000 0.000 0.335 1.000 0.065 0.000* 0.000* 0.280*
  A3_103 1.000 0.000 0.220 1.000 0.115 0.005* 0.000* 0.145*
  A3_104 1.000 0.000 0.355 0.970 0.105 0.000* 0.000* 0.250*
  A3_105 1.000 0.000 0.375 1.000 0.125 0.000* 0.000* 0.250*
  A4_101 1.000 0.000 0.345 1.000 0.130 0.000* 0.000* 0.190*
  A4_102 1.000 0.000 0.325 0.990 0.105 0.000* 0.000* 0.320*
  A4_103 1.000 0.000 0.240 1.000 0.100 0.000* 0.000* 0.150*
  A4_104 1.000 0.000 0.200 1.000 0.100 0.000* 0.000* 0.165*
  A4_105 1.000 0.000 0.050 1.000 0.175 0.000* 0.000* 0.085*
- Acquisition (flag count / phase-end heldout mean, bar >= 0.80): A1 A 5/5
  1.000 PASS, B 0/5 n/e FAIL, C 0/5 n/e FAIL, D 5/5 0.920 PASS; A2 A 5/5
  0.479 FAIL, B 5/5 0.005 FAIL, C 0/5 0.031 FAIL, D 5/5 0.198 FAIL; A3 A
  5/5 1.000 PASS, B 5/5 0.816 PASS, C 0/5 0.569 FAIL, D 5/5 1.000 PASS;
  A4 A 5/5 1.000 PASS, B 0/5 n/e FAIL, C 0/5 n/e FAIL, D 5/5 0.890 PASS.
  A2 train-flag/heldout divergence: flags A,B,D true but heldouts ~0-0.48
  (fixed-budget end-of-budget theta vs mid-phase checks; fitness=return).
  N/A stays flag-based per the correction; disagreements reported as-is.
- H: ALL cells N/A in all 20 runs (C missing everywhere, gate blocked per
  7). I1 means: A1 0.093, A2 0.115, A3 0.119, A4 0.122 (bar 0.70 missed
  everywhere; full CIs in SCORES.md).
- Retention drops (means): A1/A4 +0.000 structural (asserted); A2 A
  +0.033+-0.453 (per-seed +0.18,+0.725,-0.62,+0.19,-0.31) D -0.013; A3 B
  +0.816+-0.055 (all seeds >= 0.745: wipeout) C +0.198 D +0.090.
  Retention criterion (A1 <= 0.10): MET structural. Sensitivity (>= 0.20):
  MET via A3/B.
- Reacquisition: 13 events, 0 pass (restore 1/13, savings 5/13, both
  0/13; A3/C savings undefined). Criterion NOT MET.
- Ablations A1 (descriptive, H N/A): single-skill H1/H2 ~0 (near-chance
  as expected), H3 0.056-0.112; shuffle H3 0.110 vs full 0.193 (partial);
  mask H1/H2 = 0 (floor, uninformative); mask C = full C (no-op).
- H-based A1vA2/A2vA3/A1vA4 comparisons VOID (all N/A). Descriptive: I1
  ~0.1 all arms; A1 = A4 on acquisition (no curriculum signal); A3
  acquired most at threshold (A,B,D).
- VERDICTS. H-main: INCONCLUSIVE (UNTESTABLE: all H N/A per 7;
  independently UNINTERPRETABLE: A1 I1 0.093 < 0.70; bars never engage).
  H-iso: FAIL (sensitivity met via A3/B +0.816, but ablation comparison
  A1-vs-A2 shows no measurable reduction: pooled +0.010+-0.216, CI
  includes 0; per-skill A null, D reversed; measurability = mandated
  CIs, no invented margin). H-meta: INCONCLUSIVE (UNTESTABLE: H N/A in
  A1+A4; adversarial-subset disjunct undefined, not filled in).
- Bottom line: prerequisites failed (C 0/20, B heldout only A3) so the
  compositional-transfer question is UNTESTABLE as preregistered.
  Positive registered evidence: frozen retention perfect (structural);
  monolith catastrophically forgot B (+0.816) while modular-plastic did
  not systematically; A2 train-check/heldout divergence documented. No
  thresholds changed, no seeds dropped, no reruns.

## 2026-10-30 - W1 GO + FREEZE (redesigned follow-up post-I1, Windows host)

- User GO with methodological correction: accept diagnosis A (I1 out of
  G1 geometric support, retired permanently as blind gate/test), but the
  new generator must NOT use I1-derived bands. Wide-G1 covers the
  physically valid G1 space procedurally from the task definition.
- Design frozen in w1/PREREG_W1.md: guarded-mirror generator; agent-haz
  [35,70] (union of registered B-family pursuit spawn bands);
  beacon-haz >= MIN_SEP=10 with NO upper bound; speed [3.0,3.5] (G1
  support); no angular constraint (full range by construction); no
  regions (single general distribution). Same physics/obs/rewards.
- Recipe verbatim N1b (imitate) + N2 (hard-ES a=.01, 25 gens, tied
  ranks); only the G1 distribution changes. G2 demos (54) + singles
  pairs + ES G2/singles pools reused frozen (sha-pinned). Tiny
  original: gate 612 + pilot3 skills = 7944 (asserted).
- Freeze (gen_wide --freeze): VC==P3 bit-exact re-verified 4/4 x3;
  pins match prior records (train_meta 7ebd5b9b, n1_demos 0ea76674,
  VC shas == n2_freeze). Code shas (BEGIN-HASHES):
  gen_wide.py 4fde8c6a9e3b4bbe277e7e9ec0ed5d663175344957f71c56939447f07f64ce21
  w1_collect.py 4071657448a56088fa6d4991f41861670f42d971b5026901b734e97c4c4cefc7
  w1_imitate.py 9be75ed465cf027cbb3302f29f81af0e3aeb35f9d6b1806a2c69b73af4813d88
  w1_es.py a79c35e03e1c44b518c1be1194d476459631784bd474d0e8e4cb5cfa6cc60583
  w1_gate.py cc133dfbffde8b15ffb8d7d83409cb4f117b558bb6ae67055768648da60c0d59
  w1_h.py 37ebc77bb89e834847cb079b42ad0d0f435e2b487aaafabe5d716efc40385e4c
  PREREG_W1.md 2cbbbb1eab0476ec6f6776bffb5ad97e6d6ab2ed3c459e00e3c2982c44edf9ad
  w1_freeze.json b695381fd5e5 (full in MANIFEST.json)
  (END-HASHES)
- Pools (gen_wide --pools): demo 18 / espool 12 / gate-G1 200 /
  gate-G2 200 / sing-gate 80; 0 collisions vs 10648 prior entries
  (19 files) + internal 0.
  w1_pools.json 6050b1536fb250c511289bec8dc60934e8b10a21e4fa84980f138597b1e0856c
  w1_coverage.json e032ca4d98236ca48425e64fbbde4806be1981729ba128fc19d7afabb99933b7
- Coverage (descriptive, pre-train): gate-G1 d_gh [10.84,101.19]
  med 48.15; ang [0.0,3.13] med 1.06, sextiles [56,42,37,29,17,19]
  (all non-empty); demo d_gh [11.31,87.67]; espool [10.88,77.52].
  Near-to-far + full angular range confirmed by construction+report.
- Gate bar (frozen): W1-final mean >= 0.70 on gate-G1 AND gate-G2
  (means over 3 thetas) else STOP. H refs (descriptive follow-up,
  not preregistered): each H >= 0.60; H-mean >= best-single + 0.40.
- Pre-freeze mechanics: selftest T1 PASS (2000 throwaway, ss 21999);
  planner smoke 1 throwaway ep (ok=True, T=5, 0.3 s); py_compile 6/6.
  One pre-freeze bugfix (set->str in freeze dict); nothing frozen
  before the fix. From here w1/*.py + PREREG + freeze immutable
  (MANIFEST-enforced).

## 2026-10-30 - W1 TRAIN + GATE FAIL + STOP (H sealed)

- Collect: 54/54 P-AB planner on wide-G1 demos x3 th, 105 s of
  1800 s, 0 incomplete; 52 ok + 2 fails (D00/302, D05/302)
  included per protocol.
  w1_demos_g1.json 312009649e4b2210a69bb62cbd517f993c7d55baf74f476480516dcb6d88c866
- Imitate (N1b verbatim, 54 wide-G1 + 54 reused G2 + reused
  singles), 119.1 s: train acc .929/.904/.941; eval acc
  .769/.778/.849 vs const ~.34; singles sel A .87/.805/1,
  B 1/1/1, C .695/.95/.96, D .97/.97/1; D-mass 220/233/200.
  w1_imit_models.json 6f763e2140bea2e72326ee4ed712abf468b201124afce2ccd1a62342bfa37825
  w1_imit_results.json 011437f1d055e0a8f1157ea793bfdfca1e0b6d824e82d2f5fb47b170374cc0b4
- ES (N2 verbatim, 8 pairs/.1/.01/25 gens/tied ranks, calendar
  2 wide-G1+2G2+ABCD), 266.5 s of 2400 s, 3x25 gens, 0
  incomplete; P3 skill shas intact pre+post.
  w1_es_models.json d7710053799426318912c51dc28f371ff054f08521453aa08d2ad47bded94623
- GATE (single opening, 2640 eps, 41 s): W1-final G1 0.543
  (0.56/0.51/0.56) vs imit 0.508; G2 0.678 (0.745/0.755/0.535)
  vs imit 0.608; singles spot A 59/60 B 45/60 C 38/60 D 60/60.
  Bar 0.70 each family: FAIL both.
  w1_gate.json 169d2930cffd90cd6e03c2df6ba3bc37ab3528f970c21e1d8e023038d03df86c
- STOP per protocol: no retune. H1/H2/H3 NEVER opened
  (w1_h.py never ran; zero H episodes); H seal intact for
  future lines (mechanical dup-check loads only, N-series
  pattern). W1 line closed pending user-authorized redesign.
  Full record: w1/README.md.
