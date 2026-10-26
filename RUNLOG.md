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
