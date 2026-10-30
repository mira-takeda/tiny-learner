# tiny-learner

One small falsifiable step toward tiny non-transformer systems that learn
reusable behavior from interaction. Source of truth: `CROSSCHECK.md`.

## Current state (2026-10-30; S1–S16 closed, A01 frozen)

- pilot3: A=1.00, B=0.83, C=0.843, D=1.00; frozen retention intact.
- Composition: v2 I1=0.047 (uninterpretable), v2views I1N=0.093,
  v2conf ACQ=0.102 (FAIL), mean B-weight ≈0.001.
- S1–S16: coordination-bottleneck diagnostics (`v2conf_diag/`).
- A01 (fixed): 0.647 on ACQ18 vs A-only 0.145 / B-only 0.246 —
  useful learned selection, NOT general composition or I1/H transfer.
- Corrected monolith still pending. Four hackathon days left.

4-day rule (binding): no new probe chains. At most ONE last scientific
fork, only after the user picks (A) one last pre-frozen composition
test or (B) closing for demo/submission.

## Minimal repro (A01 vs A/B on ACQ18)

Real local CPU execution (3600 episodes, deterministic). Detail in
[repro_s16/](repro_s16/README.md); provenance in
[repro_s16/PROVENANCE.md](repro_s16/PROVENANCE.md).

Install (once):

    python3.14 -m venv .venv
    .venv/bin/pip install -r repro_s16/requirements.txt

Run:

    .venv/bin/python repro_s16/repro.py

Checks 10 hashes; ~60 s on Intel Mac. Expected output (verified
2026-10-30, incl. clean copy with fresh venv):

    hashes ok: 10 files
      th arm      arr  col  tmo     ret
     301 A01      236  164    0   4.433
     301 A-ONLY    79  321    0  -3.142
     301 B-ONLY   117   68  215   1.152
     302 A01      303   97    0   8.538
     302 A-ONLY    55  345    0  -4.895
     302 B-ONLY    98   33  269   1.570
     303 A01      237  163    0   4.524
     303 A-ONLY    40  360    0  -5.694
     303 B-ONLY    80   48  272   0.244
    mean arrivals over 1200:
      A01     0.6467
      A-ONLY  0.1450
      B-ONLY  0.2458
    REAL LOCAL EXECUTION wall=~60s episodes=3600 (deterministic, CPU-only)

## Docs map

- `CROSSCHECK.md` — operative source of truth (state, blocks, valid
  claims, pending, next authorized step).
- `HANDOFF.md` — history / technical continuity detail (per-probe log).
- `CONTEXT.md` — living project context; `CROSSCHECK.md` — state snapshot.
- `PREREG.md` (frozen), `RUNLOG.md` (append-only history).
- `v2conf_diag/README.md` — diagnostics archive; `repro_s16/` — repro package.
