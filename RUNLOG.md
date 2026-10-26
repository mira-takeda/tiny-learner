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
- From here: any implementation change = logged deviation + rerun affected runs.
