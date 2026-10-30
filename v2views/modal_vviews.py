"""Modal launcher for v2views (v2views-only).

Operational wrapper only: bakes frozen src/data + v2views snapshot
(incl. new eval battery) + v2 snapshot (G1/G2 heldout, read-only) +
the 3 pilot3 post_D thetas (read-only starts) into the image
(copy=True) and launches the 3 views-gate
trainings (seeds 301-303) as ONE batch call via experimental_spawn_map
(validated in client 1.6.2). Each seed runs the views trainer in a
CLEAN SUBPROCESS (still required: frozen run_battery asserts no-net
and the Modal worker preloads net modules). Results go to a SEPARATE
volume (tiny-learner-v2views); all other records untouched.

DO NOT INVOKE without explicit user GO.
"""

import modal

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
APP = "tiny-learner-v2views"
VOLUME = "tiny-learner-v2views"

VV_SEEDS = [301, 302, 303]

image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True)
         .add_local_dir(f"{ROOT}/v2views", remote_path="/root/tl/v2views",
                        copy=True)
         .add_local_dir(f"{ROOT}/v2", remote_path="/root/tl/v2", copy=True)
         .add_local_file(f"{ROOT}/pilot3_runs/P3_301_theta.npz",
                         remote_path="/root/tl/start/P3_301_theta.npz",
                         copy=True)
         .add_local_file(f"{ROOT}/pilot3_runs/P3_302_theta.npz",
                         remote_path="/root/tl/start/P3_302_theta.npz",
                         copy=True)
         .add_local_file(f"{ROOT}/pilot3_runs/P3_303_theta.npz",
                         remote_path="/root/tl/start/P3_303_theta.npz",
                         copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)
BASE = dict(image=image, cpu=1.0, memory=1024)


@app.function(**BASE, volumes={"/vol": vol}, timeout=86400)
def run_vviews(seed):
    """DO NOT INVOKE without explicit user GO. Runs one v2views seed
    via the views trainer in a clean subprocess (see module docstring)."""
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root/tl/v2views")
    import views_train

    assert seed in views_train.VV_SEEDS
    out_json = f"/vol/v2views/runs/VV_{seed}.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, "/root/tl/v2views/views_train.py",
           "--seed", str(seed),
           "--data", "/root/tl/data",
           "--start", f"/root/tl/start/P3_{seed}_theta.npz",
           "--g12", "/root/tl/v2/g1g2_heldout_v1.json",
           "--newbatt", "/root/tl/v2views/eval_new_v1.json",
           "--out", "/vol/v2views/runs"]
    subprocess.run(cmd, check=True, env=env, timeout=86000)
    vol.commit()
    log = json.load(open(out_json))
    return {"seed": seed, "steps": log["steps"],
            "eval": {b: log["eval"][b]["normal"]["sr"]
                     for b in ["I1N", "H1N", "H2N", "H3N"]}}


@app.local_entrypoint()
def main(only=""):
    seeds = [int(s) for s in only.split(",") if s] if only else list(VV_SEEDS)
    assert all(s in VV_SEEDS for s in seeds), seeds
    print(f"creating ONE v2views batch call with {len(seeds)} input(s) "
          f"via experimental_spawn_map: {seeds}")
    fc = run_vviews.experimental_spawn_map(seeds)
    print(f"v2views batch call created: {fc.object_id} "
          f"({len(seeds)} inputs running detached; results land in "
          f"/vol/v2views/runs on volume {VOLUME})")
