"""Modal wrapper for tiny-learner. TOY/UNREGISTERED workloads only until GO.

Registered entrypoint run_registered() is implemented but MUST NOT be invoked
without explicit user authorization. Registered training stays scalar and
otherwise exactly as frozen; this file only ships code/data and runs it remotely.

Usage (local, turbulently networked): /Library/.../python3.11 -m modal run \
  modal_app.py --action env|package|selftest|toy|persist|throughput
"""

import modal

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
APP = "tiny-learner"
VOLUME = "tiny-learner-runs"

# copy=True bakes an immutable snapshot into the image: remote code/data are
# pinned at build time and immune to later local edits.
image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)
BASE = dict(image=image, cpu=1.0, memory=1024)


@app.function(**BASE, timeout=300)
def env_info():
    import os
    import platform
    import sys

    import numpy
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count()}


@app.function(**BASE, timeout=300)
def packaging_check():
    import glob
    import hashlib

    out = {}
    for path in sorted(glob.glob("/root/tl/src/*.py")
                       + glob.glob("/root/tl/data/*.json")):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        out[path.replace("/root/tl/", "")] = h.hexdigest()
    return out


@app.function(**BASE, timeout=2400)
def run_selftest():
    import io
    import sys
    from contextlib import redirect_stdout

    sys.path.insert(0, "/root/tl/src")
    import selftest

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            code = selftest.main()
    except SystemExit as e:
        code = e.code
    return {"exit": code, "output": buf.getvalue()[-4000:]}


TOY_CFGS = {
    "A": {"agent": [50, 50], "goal": {"pos": [70, 60]}},
    "B": {"agent": [50, 50], "hazard": {"pos": [20, 20], "speed": 3.5}},
    "C": {"agent": [30, 30], "obj": {"pos": [40, 40]},
          "zone": {"center": [50, 92]}},
    "D": {"agent": [50, 20], "goal": {"pos": [50, 65]},
          "wall": {"y": 50, "gap_x": 50, "gap_w": 16}},
}
TOY_FAMS = ("A", "B", "C", "D")

DIAG_CFGS = dict(TOY_CFGS, **{
    "G1": {"agent": [50, 50], "goal": {"pos": [70, 60]},
           "hazard": {"pos": [20, 20], "speed": 3.2}},
    "G2": {"agent": [30, 30], "obj": {"pos": [40, 40]},
           "zone": {"center": [50, 92]}},
    "H1": {"agent": [30, 30], "obj": {"pos": [40, 40]},
           "zone": {"center": [50, 92]},
           "hazard": {"pos": [80, 80], "speed": 4.5}},
    "H2": {"agent": [30, 30], "obj": {"pos": [40, 40]},
           "zone": {"center": [50, 92]},
           "hazard": {"pos": [45, 50], "speed": 4.5}},
    "H3": {"agent": [30, 30], "obj": {"pos": [40, 35]},
           "zone": {"center": [50, 92]},
           "wall": {"y": 55, "gap_x": 45, "gap_w": 11}},
    "I1": {"agent": [50, 50], "goal": {"pos": [70, 60]},
           "hazard": {"pos": [62, 55], "speed": 3.2}},
})


@app.function(**BASE, timeout=1800)
def toy_episodes(n_eps=64, seed_base=5000):
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/root/tl/src")
    from models import N_SKILLS, SKILL_SHAPES, GATE_SHAPES, ModularNet
    from policies import xavier_init
    from tasks import TaskRunner

    rng = np.random.default_rng(0)
    theta = np.concatenate(
        [xavier_init(rng, SKILL_SHAPES) for _ in range(N_SKILLS)]
        + [xavier_init(rng, GATE_SHAPES)])
    net = ModularNet(theta)
    tr = TaskRunner()
    out, steps = [], 0
    t0 = time.time()
    for k in range(n_eps):
        fam = TOY_FAMS[k % 4]
        cfg, seed = TOY_CFGS[fam], seed_base + k
        r2 = np.random.default_rng(seed)
        o = tr.reset(fam, cfg, seed)
        tot, done = 0.0, False
        while not done:
            p, _ = net.probs(o)
            o, r, done, info = tr.step(int(r2.choice(len(p), p=p)))
            tot += r
            steps += 1
        out.append([round(float(tot), 9), bool(info["success"])])
    wall = time.time() - t0
    import resource

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return {"episodes": out, "steps": steps, "wall_s": wall,
            "steps_per_s": steps / wall, "peak_rss_mb": peak_kb}


@app.function(**BASE, timeout=1800)
def vecdiag():
    """Toy-only: pinpoint scalar-vs-vec first divergence + norm kernels."""
    import sys

    import numpy as np

    sys.path.insert(0, "/root/tl/src")
    from models import (N_SKILLS, SKILL_SHAPES, GATE_SHAPES, MONO_SHAPES,
                        ModularNet)
    from policies import xavier_init
    from tasks import TaskRunner, VecTaskRunner
    from train import rollout
    from vec import run_batch

    rep = {}
    rng = np.random.default_rng(0)
    D = rng.normal(size=(50000, 2)) * 50
    s = np.array([np.linalg.norm(d) for d in D])
    b = np.sqrt((D * D).sum(axis=1))
    rep["norm_micro"] = {"n": 50000,
                         "mismatches": int((s != b).sum()),
                         "max_abs_diff": float(np.abs(s - b).max())}

    rng = np.random.default_rng(0)
    mtheta = np.concatenate(
        [xavier_init(rng, SKILL_SHAPES) for _ in range(N_SKILLS)]
        + [xavier_init(rng, GATE_SHAPES)])
    tmono = xavier_init(rng, MONO_SHAPES)
    net = ModularNet(mtheta)
    fams = {}
    for fam, cfg in DIAG_CFGS.items():
        fe = {"full": {}, "lockstep": {}}
        for arch, th in (("modular", mtheta), ("mono", tmono)):
            items = [(cfg, 300 + k) for k in range(4)]
            vr, vs = run_batch(th, arch, fam, items, backend="scalar")
            det = []
            for k, (c, sd) in enumerate(items):
                r0, s0 = rollout(th, arch, fam, c, sd, [0])
                det.append({"ep": k, "seed": sd,
                            "ret_equal": bool(r0 == vr[k]),
                            "suc_equal": bool(s0 == vs[k]),
                            "ret_scalar": float(r0), "ret_vec": float(vr[k]),
                            "ret_abs_diff": float(abs(r0 - vr[k]))})
            fe["full"][arch] = det
        d = {"diverged": False}
        for k in range(3):
            seed = 300 + k
            s_tr, v_tr = TaskRunner(), VecTaskRunner(1)
            r2 = np.random.default_rng(seed)
            o1 = s_tr.reset(fam, cfg, seed)
            o2 = v_tr.reset(fam, [cfg], [seed])
            t = 0
            first = None
            while first is None and t < 200:
                p, _ = net.probs(o1)
                a = int(r2.choice(len(p), p=p))
                o1, r1, done1, i1 = s_tr.step(a)
                oo, rr, dd, ss = v_tr.step(np.array([a]))
                t += 1
                cands = [("obs", o1, oo[0]),
                         ("agent_pos", s_tr.env.agent_pos,
                          v_tr.env.agent_pos[0]),
                         ("agent_vel", s_tr.env.agent_vel,
                          v_tr.env.agent_vel[0]),
                         ("reward", np.array(r1), np.array(float(rr[0])))]
                if s_tr.env.haz_pos is not None:
                    cands.append(("haz_pos", s_tr.env.haz_pos,
                                  v_tr.env.haz_pos[0]))
                if s_tr.env.obj_pos is not None:
                    cands.append(("obj_pos", s_tr.env.obj_pos,
                                  v_tr.env.obj_pos[0]))
                    cands.append(("obj_vel", s_tr.env.obj_vel,
                                  v_tr.env.obj_vel[0]))
                for name, x, y in cands:
                    if not np.array_equal(x, y):
                        first = {"step": t, "quantity": name,
                                 "max_abs_diff": float(np.abs(x - y).max())}
                        break
                if first is None and (done1 != bool(dd[0])
                                      or i1["success"] != bool(ss[0])):
                    first = {"step": t, "quantity": "done/success",
                             "max_abs_diff": 1.0}
                if done1:
                    break
            if first is not None:
                first.update({"ep": k, "seed": seed})
                d = {"diverged": True, "first": first}
                break
        fe["lockstep"] = d
        fams[fam] = fe
    rep["families"] = fams
    return rep


@app.function(**BASE, timeout=1800)
def run_selftest_twice():
    """Toy-only: full selftest.main() twice, same process, unchanged code.

    Resets only the module-level PASS accumulator between runs (harness-side;
    selftest.py itself is untouched) so each run's verdict is independent.
    Run 0 replicates the original verify conditions exactly; run 1 tests
    in-process repeatability.
    """
    import io
    import sys
    from contextlib import redirect_stdout

    sys.path.insert(0, "/root/tl/src")
    import selftest

    runs = []
    for i in range(2):
        selftest.PASS.clear()
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                code = selftest.main()
        except SystemExit as e:
            code = e.code
        out = buf.getvalue()
        checks = [ln for ln in out.splitlines()
                  if ln.startswith("[PASS]") or ln.startswith("[FAIL]")]
        runs.append({
            "run": i,
            "exit": code,
            "n_pass": sum(ln.startswith("[PASS]") for ln in checks),
            "n_fail": sum(ln.startswith("[FAIL]") for ln in checks),
            "fails": [ln[7:] for ln in checks if ln.startswith("[FAIL]")],
            "checks": checks,
        })
    return {"runs": runs}


@app.function(**BASE, timeout=1800)
def vecdiag2():
    """Toy-only: exact selftest-vec repro x3 + lockstep on first mismatch."""
    import sys

    sys.path.insert(0, "/root/tl/src")
    from diag import exact_vec_pass, lockstep_first_div, build_thetas
    from selftest import TOY_CFGS

    rep = {"isolated_passes": []}
    for p in range(3):
        ok, mm = exact_vec_pass()
        entry = {"pass": p, "ok": bool(ok)}
        if not ok:
            entry["mismatch"] = mm
            th = build_thetas()[mm["arch"]]
            entry["lockstep"] = lockstep_first_div(
                mm["family"], mm["arch"], TOY_CFGS[mm["family"]],
                mm["seed"], th)
        rep["isolated_passes"].append(entry)
    return rep


@app.function(image=image, volumes={"/vol": vol}, cpu=1.0, memory=1024,
             timeout=300)
def persistence_write(tag):
    import json
    import os

    path = f"/vol/probe_{tag}.json"
    payload = {"tag": tag, "n": 3}
    with open(path, "w") as f:
        json.dump(payload, f)
    vol.commit()
    return {"path": path, "bytes": os.path.getsize(path)}


@app.function(image=image, volumes={"/vol": vol}, cpu=1.0, memory=1024,
             timeout=300)
def persistence_read(tag):
    import json
    import os

    path = f"/vol/probe_{tag}.json"
    with open(path) as f:
        payload = json.load(f)
    os.remove(path)
    vol.commit()
    return payload


@app.function(**BASE, volumes={"/vol": vol}, timeout=86400)
def run_registered(arm, seed):
    """DO NOT INVOKE without explicit user GO. Runs one frozen arm/seed job."""
    import json
    import os
    import sys

    sys.path.insert(0, "/root/tl/src")
    import train

    assert arm in train.ARMS and seed in train.SEEDS
    log = train.run_arm_seed(arm, seed, "/root/tl/data", "/vol/runs")
    vol.commit()
    return {"arm": arm, "seed": seed,
            "phases": {k: {"gens": v["gens"], "acquired": v.get("acquired")}
                       for k, v in log["phases"].items()},
            "total_steps": log["total_steps"]}


@app.local_entrypoint()
def main(action="env", n_eps=64, seed_base=5000, tag="probe"):
    if action == "env":
        print(env_info.remote())
    elif action == "package":
        for k, v in sorted(packaging_check.remote().items()):
            print(f"{k} {v}")
    elif action == "selftest":
        r = run_selftest.remote()
        print(r["output"])
        print("SELFTEST-REMOTE-EXIT:", r["exit"])
    elif action == "toy":
        import json as _j

        print(_j.dumps(toy_episodes.remote(n_eps=n_eps,
                                           seed_base=seed_base)))
    elif action == "persist":
        print(persistence_write.remote(tag))
        print(persistence_read.remote(tag))
    elif action == "verify":
        import json as _j

        print("### ENV")
        print(_j.dumps(env_info.remote(), indent=1))
        print("### PACKAGE")
        pkg = packaging_check.remote()
        for k in sorted(pkg):
            print(f"{k} {pkg[k]}")
        print("### SELFTEST")
        r = run_selftest.remote()
        print(r["output"])
        print("SELFTEST-REMOTE-EXIT:", r["exit"])
        print("### TOY-RUN-1")
        t1 = toy_episodes.remote(n_eps=64, seed_base=5000)
        print(_j.dumps(t1))
        print("### TOY-RUN-2")
        t2 = toy_episodes.remote(n_eps=64, seed_base=5000)
        print(_j.dumps(t2["episodes"]))
        print("MATCH-RUN1:", t1["episodes"] == t2["episodes"])
        print("### THROUGHPUT")
        t3 = toy_episodes.remote(n_eps=256, seed_base=9000)
        print(_j.dumps({k: t3[k] for k in ("steps", "wall_s", "steps_per_s",
                                           "peak_rss_mb")}))
        print("### PERSIST")
        print(persistence_write.remote("verify1"))
        print(persistence_read.remote("verify1"))
        print("### VERIFY-DONE")
    elif action == "vecdiag":
        import json as _j

        print(_j.dumps(vecdiag.remote(), indent=1))
    elif action == "vecdiag2":
        import json as _j

        print(_j.dumps(vecdiag2.remote(), indent=1))
    elif action == "selftest2x":
        import json as _j

        print(_j.dumps(run_selftest_twice.remote(), indent=1))
    else:
        raise SystemExit(f"unknown action: {action}")
