"""N4 secondary: singles-heldout D/C selection check (frozen).
N4 + N3b gates on 20 A/B/C/D control-manifest singles x3 thetas.
One-shot with heldout-3. Budget 120 s.
"""
import hashlib
import json
import sys

sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/src")
sys.path.insert(0, "/Users/satoru/Documents/hackathon/tiny-learner/v2views")

import numpy as np

import s4_closed as S4
from models import GATE_SHAPES, mlp_forward, split_modular
from tasks import TaskRunner

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
SEEDS = (301, 302, 303)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    fr = json.load(open(f"{ROOT}/v2conf_diag/n4_freeze.json"))
    assert sha(f"{ROOT}/data/train_control_v1.json") == fr["heldout3_sha"]
    ctrl = {e["id"]: e for e in json.load(
        open(f"{ROOT}/data/train_control_v1.json"))}
    m4 = json.load(open(f"{ROOT}/v2conf_diag/n4_models.json"))
    m3 = json.load(open(f"{ROOT}/v2conf_diag/n3b_models.json"))
    skills = {s: split_modular(np.load(
        f"{ROOT}/v2conf_runs/VC_{s}_theta.npz")["theta"])[0] for s in SEEDS}
    out = []
    for arm, m, k in (("N4", m4, "N4"), ("N3b", m3, "N3b")):
        for s in SEEDS:
            g = np.array(m[f"{k}_{s}"])
            for f in "ABCD":
                for eid in fr["singles_heldout"][f]:
                    e = ctrl[eid]
                    r = TaskRunner()
                    obs = r.reset(e["family"], e["cfg"], int(e["seed"]))
                    uses, done = [0, 0, 0, 0], False
                    while not done:
                        kk = int(np.argmax(mlp_forward(g, obs, GATE_SHAPES)))
                        a = S4.greedy_skill(skills[s], obs, kk)
                        uses[kk] += 1
                        obs, _, done, info = r.step(a)
                    out.append({"arm": arm, "theta": s, "fam": f, "id": eid,
                                "success": bool(info["success"]),
                                "uses": uses})
    json.dump(out, open(f"{ROOT}/v2conf_diag/n4_singles.json", "w"))
    for arm in ("N4", "N3b"):
        for f in "ABCD":
            rs = [r for r in out if r["arm"] == arm and r["fam"] == f]
            ok = sum(1 for r in rs if r["success"])
            u = np.mean([r["uses"] for r in rs], axis=0).round(1)
            print(f"{arm} {f}: ok={ok}/{len(rs)} uses={u}", flush=True)


if __name__ == "__main__":
    main()
