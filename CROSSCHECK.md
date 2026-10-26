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
- **Latest milestone:** Modal verify HEALTHY except vec-exact (26/27 remote;
  determinism+throughput+persist pass). Isolation proven; vecdiag ready.
- **Active caveats:** shared masked obs schema supplies factorization (modularity claim,
  not emergence); frozen retention is structural (3-arm comparison carries the claim);
  skill ontology is experimenter-carved; gate saw compositeness (only combos are novel);
  generator and implementer share weights (procedural blinding + audit mitigate).
- **Pre-run corrections (no runs yet, not result-driven):** (1) fixed 400-gen budgets,
  record-only checks, FAIL at gen 400 (supersedes PREREG early-stop/800-extension);
  (2) absorbing padding to exact 41.28M steps/phase. Gap-fills: CONTACT_R=5.0,
  G1-train beacon–haz≥35, H2 haz–agent sep≥10, shared train manifests, gate fixed 400.
- **Unresolved:** runtime amendment decision (shrink budget and/or tolerate batched
  divergence?) — must be made pre-result with explicit authorization. B-phase
  acquisition difficulty still unknown (N/A rules ready).
- **NOT run yet:** everything registered (launched 15:06, aborted ~15:35 pre-result,
  zero outputs). Relaunch requires hygiene (clear runs/*.json+*.npz) + explicit GO.
- **Next authorized step:** USER runs host vecdiag (toy-only) and pastes output;
  then GO decision on registered Modal runs (dead-vec cleanup first). No training yet.
- **Canonical paths (all under `/Users/satoru/Documents/hackathon/tiny-learner/`):**
  `PREREG.md`, `RUNLOG.md`, `CONTEXT.md`, `CROSSCHECK.md`, `src/train.py`,
  `src/{models,tasks,gen,evaluate,audit,es}.py`, `data/MANIFEST.json`,
  `data/{test_battery,train_meta,train_control}_v1.json`. Runs → `runs/` (ignored).
