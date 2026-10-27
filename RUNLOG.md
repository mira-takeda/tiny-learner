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
  before scoring. Then PREREG analysis on raw results.f5599ae7e3a9b91fb299e3617c98ec47d49fc4
