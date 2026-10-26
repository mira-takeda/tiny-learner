# tiny-learner — Project Context (living document)

> Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). Theme: Push the Frontier. Track: indie.
> This file is the running context for the small non-transformer project.
> Rule: keep it updated whenever a decision is made, a result lands, or scope changes.

## 1. Goal

- Long-term: a very small, cheap artificial system that learns reusable behavior from experience
  without storing world knowledge in a giant language model.
- Hackathon scope: ONE small falsifiable experiment testing one step toward that. No AGI claims.

## 2. Hard constraints

- Core learner is NOT a transformer; must run comfortably on this Intel Mac CPU.
- Learning happens in the small system — never faked by having Muse answer the task.
- Meta/Muse may be: teacher, experience generator, evaluator, adversarial tester, specialist.
- Must compare before vs after learning; must test transfer (not memorization).
- Must test whether new learning damages old learning (forgetting) + reacquisition.
- Quantitative, reproducible, pre-registered thresholds.
- Demo must be visually understandable in ~3 minutes.
- No chatbot wrapper, no simple RAG, no predetermined-answer experiment.

## 3. Prior-work separation (binding)

- TAVI, FOCO, Semilla predate the hackathon = prior research, NOT hackathon work.
- Do not use their implementations; do not assume their architecture.
- Inspect as reference ONLY if the user explicitly authorizes it; never copy code into hackathon work.
- All new work lives under `/Users/satoru/Documents/hackathon/`:
  - this project: `tiny-learner/` ONLY
  - Ray-Ban project: `rayban/` ONLY
  - never in TAVI/FOCO/Semilla/Desktop/Downloads.

## 4. Environment snapshot (audited 2026-10-26, read-only)

- Muse Code 1.4.13; provider `meta`, model `muse-spark-1.3-contributor` (CLI login not done).
- Mac Intel i5-7360U, 4 threads, 8 GB RAM, macOS 13.7.8 Ventura, x86_64.
- Disk 94% full (~6.5 GB free) — no room for Xcode/Android toolchains or local LLM weights.
- Python 3.11 (bare) + 3.14 (numpy/matplotlib/pillow). No torch/JAX/TF. No Node, Docker, Go, Rust.
- Command Line Tools only (no Xcode.app, no Java runtime, no Android SDK).
- Git 2.39 present but unconfigured (no identity); no `gh`, no SSH keys → GitHub NOT usable yet.
- No Meta SDKs/keys/MCP servers configured. Network to github.com / api.github.com / meta.com OK (proxy).
- Run interpreter: `/usr/local/bin/python3.14` (3.14, numpy 2.5.4). NOTE: bare `python3`
  resolves to 3.11 WITHOUT numpy — always invoke `python3.14` explicitly. Verified 2026-10-26.

## 5. Chosen design: B-modified ("Toolkit", interaction-learned skills) — REVISED + FROZEN 2026-10-26

Hypothesis: a tiny modular non-transformer net acquires atomic skills purely from interaction
(scalar reward, NO teacher demonstrations), then solves NEVER-SEEN skill combinations by
combining those skills — beating an equal-parameter monolith — while parameter isolation
protects old skills during sequential training.

- Env: 2D continuous arena, NumPy physics, seeded. Obs: fixed 14-dim (velocity + egocentric
  vectors/distances to target/hazard/object/destination, absent = zero-masked).
- Actions: DISCRETE, 9 actions (8 compass accel impulses at fixed magnitude + coast).
- Algorithm: Evolution Strategies — FROZEN 2026-10-26 (formal LineWorld sanity PASS:
  sr 1.00, 10/150 gens, 8s wall). Discrete REINFORCE implemented as fallback, unused.
- Skills: 4 MLP modules (14→32→32→9 softmax, ~1833 each) + softmax gate MLP (~612).
  Total ≈ 7944 params. Gate is observation-only (no task ID). Action = gate-weighted blend.
- Atomic families: A=approach beacon; B=avoid PURSUING hazard + net-displacement requirement
  (stillness/corner/spin all fail by construction); C=transport object with friction + 5-step
  zone dwell; D=confined transit through gap in full-width wall. All shaping potential-based.
- Sequential protocol A→B→C→D with THREE arms: (1) modular-frozen, (2) modular-plastic,
  (3) equal-param monolith (±5%, asserted in code) + (4) modular-frozen + control schedule.
- Gate training (after D, skills frozen): ONLY on G1=A+B guarded approach and G2=A+C fetch.
- Held-out composite tests: H1=B+C hazardous transport; H2=A+B+C guarded fetch (higher-order);
  H3=D+C corridor delivery (zero-shot D use). Plus I1 interpolation probe (seen combo A+B).
- Meta role: ONE snapshotted curriculum spec over shared generator G (schema + skill defs only,
  blinded to test configs); local seeded generator instantiates all episodes. Zero per-episode
  API calls. Curriculum snapshot hashed BEFORE test configs are sampled (order enforced).
- No-Meta control: SAME generator G, SAME step budgets, fixed stratified-uniform schedule
  (seed 31337).
- Test-time guarantee: evaluation runs with network disabled; harness calls only the small system.
- Preregistered thresholds (frozen in PREREG.md §5): acquisition ≥80%/skill; I1 ≥70% (else
  H-main UNINTERPRETABLE); each H ≥60% AND mean ≥15pp over monolith AND ≥40pp over best
  single skill; frozen drop ≤10pp; reacquisition savings ≥2×; Meta ≥10pp over control.
  200 eps/battery, 5 training seeds (101–105), mean ± 95% CI.
- CLARIFICATION 1 (validity vs success): the ≥20pp forgetting condition in a non-isolated arm
  is a VALIDITY condition for H-iso ONLY. If unmet → H-iso INCONCLUSIVE, no isolation claim,
  and H-main judged independently (not failed). Recorded in PREREG §7.
- CLARIFICATION 2 (optimizer fallback): formal sanity check on separate LineWorld env with
  pre-registered PASS bar (≥90%/100eps, ≤150 gens/14.4k eps, ≤20 min). ES first; if FAIL →
  REINFORCE re-run; if both FAIL → STOP and report. Debugging runs are unregistered; the
  formal run is the first clean run of final code. Optimizer frozen before registered training.
- Falsifiers: H ≤ monolith; single skill solves H; gate-shuffle harmless; leakage found;
  no-Meta matches Meta; I1 fails (gate failure, uninterpretable).
- Demo (3 min): bet slide → before/after playback → H1 money shot with live skill-activation
  bar ("Meta offline") → bars (modular vs monolith vs single-skill vs chance) + retention
  matrix → honest claim + one control result.
- Allowed/forbidden claims: per PREREG §1 (scoped to this protocol/family; no emergence or
  forgetting-solved claims; no gate-retention claims).

## 6. Status

- [x] Environment audit (2026-10-26)
- [x] 3 designs proposed + ranked (B > A > C)
- [x] B-modified experiment plan specified (15 points)
- [x] Adversarial review (held-out combos, forgetting framing, matched control, red-team, ES)
- [x] Plan approved with 2 clarifications → PREREG.md FROZEN (2026-10-26)
- [x] Learner + training system implemented and FROZEN (2026-10-26, no runs yet)
- [x] Optimizer formal sanity check: ES PASS → optimizer frozen (2026-10-26)
- [x] Curriculum snapshot (Meta) + control schedule + test battery, audits PASS (2026-10-26)
- [ ] Registered training + evaluation
- [ ] Results + demo

## 7. Next steps

1. Implement: env + generator + policies + ES/REINFORCE + sanity runner.
2. Debug freely (unregistered), then ONE formal sanity run → freeze optimizer.
3. Report sanity result; await confirmation before registered training.
4. Snapshot Meta curriculum → control schedule → test battery (order + hashes in RUNLOG).
5. Registered training (4 arms × 5 seeds), evaluation, demo.

## 8. Changelog

- 2026-10-26: created; recorded audit, constraints, paths, B-modified design, status.
- 2026-10-26: adversarial revision — discrete 9-action + ES; held-out combos H1/H2/H3 + I1;
  3-arm forgetting design; matched no-Meta control; reward red-team fixes; N/A rules.
- 2026-10-26: approval clarifications recorded; PREREG.md frozen (hash in RUNLOG at freeze).
  Optimizer gated on formal LineWorld sanity; forgetting-sensitivity demoted to validity-only.
- 2026-10-26: formal optimizer sanity PASS (ES, sr=1.00, 10 gens, 8s wall); optimizer frozen as ES.
- 2026-10-26: registered datasets frozen + audits PASS (leakage 0, counts/seeds/subranges verified);
  chance baselines measured (A .40, B .00, C .045, D .405, I1/H1/H2 .00, H3 .095).
- 2026-10-26: training system frozen (22/22 toy selftests, freeze-check PASS, hashes in RUNLOG).
- 2026-10-26: compliance correction — fixed 400-gen/phase budgets for all arms, record-only
  checks, FAIL at gen 400 (supersedes PREREG early-stop/800-extension); re-frozen (24/24 tests).
- 2026-10-26: compliance correction 2 — exact 41,280,000 steps/phase via absorbing padding;
  4 workers chosen (toy benchmark: 2.03x, deterministic); re-frozen (26/26 tests).
