# CROSSCHECK — tiny-learner current-state snapshot

Living summary for cross-checking only. Updated after every milestone, result,
deviation, correction, freeze, or next-step change. Details live in CONTEXT.md /
RUNLOG.md / PREREG.md — this file never replaces them.

- **Goal:** one small falsifiable step toward tiny non-transformer systems that learn
  reusable behavior from interaction. No AGI claims.
- **Hypothesis (H-main):** a tiny modular net acquires atomic skills A–D from scalar
  reward alone, then solves never-seen conjunctions H1–H3 by combining them, beating
  a matched monolith. Secondaries: H-iso (isolation reduces interference,
  validity-conditioned), H-meta (Meta curriculum adds value at equal budgets).
- **Git HEAD:** see `git log` (code frozen at `e8f3137`; docs/ops commits after).
- **Preregistration:** FROZEN 2026-10-26 (`PREREG.md` sha `fd7de8cd…`, unaltered).
- **Datasets:** FROZEN + audited (leakage 0; counts/seeds/subranges PASS; hashes in
  `data/MANIFEST.json`). Code: FROZEN at `e8f3137` (hashes in RUNLOG).
- **Architecture:** 4 MLP skills (14→32→32→9, 1,833 each) + softmax gate (14→32→4,
  612) = **7,944 params**; monolith 14→78→78→9 = **8,043 (+1.25%)**. Discrete 9-action.
  Gate is observation-only; cannot emit actions directly.
- **Optimizer / budget:** ES (antithetic 32 pairs, σ=0.1, α=0.05, fitness 8 eps).
  Fixed **41,280,000 env steps/phase/arm/seed** (400 gens×64×8×200 + 40 checks×40×200),
  absorbing-padded, asserted in code. REINFORCE implemented, unused.
- **Arms:** A1 modular-frozen+Meta, A2 modular-plastic+Meta, A3 monolith+Meta,
  A4 modular-frozen+control. Sequential A→B→C→D→gate. Seeds 101–105.
- **Tests:** atomic-heldout A–D (200 eps each); interpolation probe I1 (seen combo,
  new params); held-out conjunctions H1=B+C, H2=A+B+C, H3=D+C (200 eps each).
- **Thresholds:** acq ≥80%/skill; I1 ≥70% (else H UNINTERPRETABLE); each H ≥60% AND
  H-mean ≥15pp over monolith AND ≥40pp over best single skill; frozen drop ≤10pp;
  savings ≥2×; Meta ≥10pp over control. 5 seeds, 95% CI.
- **Latest milestone:** W1 GATE FAIL 2026-10-30 (Windows host):
  wide-G1 redesigned follow-up (no I1-derived numbers; agent-haz
  [35,70] from B-family support, beacon-haz >=10 unbounded, full
  angular range; frozen+hashed, pools disjoint, coverage
  descriptive) trained with verbatim N1b+N2 recipes (Tiny
  original 7944). Fresh gate (single opening): W1-final G1
  0.543 / G2 0.678 vs bar 0.70 each -> FAIL both (imit
  0.508/0.608; ES improved +3.5/+7.0pp but not functioning).
  STOP per protocol, no retune; H1/H2/H3 NEVER opened, seal
  intact. Full record: w1/README.md + RUNLOG. Before:
  I1-GAP DIAG 2026-10-30 (offline, no H/train):
  train d_gh ≥35.6 vs I1 12–24.9 (ZERO overlap, 100% OOS) + aligned
  hazard (ang 80% OOS); heldouts in-support (3% OOS, 0.68–0.78);
  gate degenerate on I1. DECISION A: I1 off G1 geometric support →
  widen G1 procedurally (d_gh ~10–35), Tiny original kept. I1 can
  never be a blind gate again for this line; H still sealed. Before:
  N5 DONE — I1 GATE FAILED. Frozen
  N3b: 0.105/0.250/0.090 (0.148 global) vs A-only 0.147 / B-only
  0.220. Train-manifest coordination does NOT interpolate to I1
  geometry. Verdict: H UNINTERPRETABLE. H1/H2/H3 NOT opened; no
  retunes into I1. Before: N4 DONE (joint recipe from N3b:
  N4 0.678/0.772 fails joint bar; CONTROL N3b meets it 0.711/0.744
  on fresh manifest; D 60/60 intact). Decision recorded: frozen
  N3b-finals advance as THE I1 candidate (heldout-3 burned; I1 still
  sealed for single opening). Before: N3 DONE (per-family peaks:
  0.722, G2-only 0.778; N3b best joint 0.689/0.778; G1 peak marginal).
  Decision: joint-optimization problem, not capacity wall. Before: N2
  DONE (hard-ES α=.01 from N1b:
  G1 123 vs 136, G2 122 vs 94 — traded G1 for G2; 0.683/0.678 both
  <0.70, G1 regressed; NO clean pass to I1). Before: N1b DONE (eval
  acc 0.77/0.79/0.84, D-mass>0; pre-N2 G1 50/49/37, G2 35/33/26).
  Before: N1 DONE (108 planner demos G1+G2 →
  14→32→4 intact 5000ep; eval acc 0.59/0.51/0.60, S5-like overfit;
  B rare in train, D absent by design). Before: N0 DONE (A/C planner
  on G2: 88/90 vs
  C-only 71/90 vs A-only 0/90; 19 scenarios need both; exact G2 mirror,
  V1 bit-exact). POSITIVE branch: frozen A+C contain a G2 solution;
  next N1 = teach it to the 14→32→4 gate, no arch change. Before that:
  S1–S16 CLOSED, A01 FROZEN. Registered campaign
  (2026-10-27): 20/20 jobs, 40/40 artifacts, 250 post-hoc batteries. Verdicts:
  H-main INCONCLUSIVE (UNTESTABLE, C unacquired 20/20; I1 0.093 < 0.70),
  H-iso FAIL (no measurable A1-vs-A2 reduction), H-meta INCONCLUSIVE
  (UNTESTABLE). Follow-up line: pilot3 A=1.00/B=0.83/C=0.843/D=1.00;
  v2 I1=0.047 (uninterpretable); v2views I1N=0.093; v2conf ACQ=0.102,
  mean B-weight ≈0.001, B-ablation ~no effect; S1–S16 coordination-
  bottleneck diagnostics; A01 fixed at 0.647 on ACQ18 vs A-only 0.145 /
  B-only 0.246 (useful learned selection, NOT general composition or
  I1/H transfer). Full record in RUNLOG / HANDOFF / v2conf_diag/README.
- **Active caveats:** shared masked obs schema supplies factorization (modularity claim,
  not emergence); frozen retention is structural (3-arm comparison carries the claim);
  skill ontology is experimenter-carved; gate saw compositeness (only combos are novel);
  generator and implementer share weights (procedural blinding + audit mitigate).
- **Pre-run corrections (no runs yet, not result-driven):** (1) fixed 400-gen budgets,
  record-only checks, FAIL at gen 400 (supersedes PREREG early-stop/800-extension);
  (2) absorbing padding to exact 41.28M steps/phase. Gap-fills: CONTACT_R=5.0,
  G1-train beacon–haz≥35, H2 haz–agent sep≥10, shared train manifests, gate fixed 400.
- **Unresolved:** corrected monolith still pending. Composition verdicts stand
  (v2/v2views uninterpretable, v2conf ACQ-FAIL); A01 (0.647 vs 0.145/0.246)
  shows useful learned selection only. Modal CPU-billing assumption (1 vCPU =
  0.5 billed core) unconfirmed — cost reported with and without it.
- **W1 status:** RUN 2026-10-30 -> GATE FAIL (see Latest
  milestone). The "d_gh ~10-35" sketch from Decision A was
  SUPERSEDED by the W1 methodological correction (no I1-derived
  numbers; PREREG_W1.md). H1/H2/H3 still sealed (never opened).
  Rule: no microtuning chains without a decision-changing
  question; no retune on this line (gate FAIL is terminal).
- **Next authorized step:** NONE. STOPPED after W1 gate FAIL.
  Awaiting user decision: (A) authorize a NEW redesigned line
  (new generator/gate design; H still sealed and usable), or
  (B) close experimentation -> demo/submission on the honest
  claim only. Do not design or run anything until then. Cloud
  still in reserve. Demo deferred.
  Demo (whenever built) stays on the HONEST claim only: registered H
  untestable as preregistered, monolith forgot B while frozen arms retained
  (structural), A01 = useful learned selection (0.647 vs 0.145/0.246).
  No emergence/transfer claims.
- **Canonical paths (all under `/Users/satoru/Documents/hackathon/tiny-learner/`):**
  `PREREG.md`, `RUNLOG.md`, `CONTEXT.md`, `CROSSCHECK.md`, `src/train.py`,
  `src/{models,tasks,gen,evaluate,audit,es}.py`, `data/MANIFEST.json`,
  `data/{test_battery,train_meta,train_control}_v1.json`. Runs → `runs/` (ignored).
