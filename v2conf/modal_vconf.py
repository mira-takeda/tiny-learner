"""Modal launcher for v2conf (v2conf-only).

Operational wrapper only: bakes frozen src/data + v2conf snapshot
(incl. conflict train set + eval batteries) + v2 snapshot (G1/G2
heldout, read-only) + v2views snapshot (views code, read-only) + the
3 pilot3 post_D thetas (read-only starts) into the image (copy=True)
and launches the 3 conflict-gate trainings (seeds 301-303) as ONE
batch call via experimental_spawn_map (validated in client 1.6.2).
Each seed runs the conf trainer in a CLEAN SUBPROCESS (still
required: frozen run_battery asserts no-net and the Modal worker
preloads net modules). Results go to a SEPARATE volume
(tiny-learner-v2conf); all other records untouched.

DO NOT INVOKE without explicit user GO.
"""

import modal

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
APP = "tiny-learner-v2conf"
VOLUME = "tiny-learner-v2conf"

VC_SEEDS = [301, 302, 303]

image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True)
         .add_local_dir(f"{ROOT}/v2conf", remote_path="/root/tl/v2conf",
                        copy=True)
         .add_local_dir(f"{ROOT}/v2", remote_path="/root/tl/v2", copy=True)
         .add_local_dir(f"{ROOT}/v2views", remote_path="/root/tl/v2views",
                        copy=True)
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
def run_vconf(seed):
    """DO NOT INVOKE without explicit user GO. Runs one v2conf seed
    via the conf trainer in a clean subprocess (see module docstring)."""
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root/tl/v2conf")
    import conf_train

    assert seed in conf_train.VC_SEEDS
    out_json = f"/vol/v2conf/runs/VC_{seed}.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, "/root/tl/v2conf/conf_train.py",
           "--seed", str(seed),
           "--data", "/root/tl/data",
           "--start", f"/root/tl/start/P3_{seed}_theta.npz",
           "--tconf", "/root/tl/v2conf/train_conflict_v1.json",
           "--evconf", "/root/tl/v2conf/eval_conf_v1.json",
           "--g12", "/root/tl/v2/g1g2_heldout_v1.json",
           "--out", "/vol/v2conf/runs"]
    subprocess.run(cmd, check=True, env=env, timeout=86000)
    vol.commit()
    log = json.load(open(out_json))
    return {"seed": seed, "steps": log["steps"],
            "acq": log["eval"]["ACQ"]["normal"]["sr"]}


@app.local_entrypoint()
def main(only=""):
    seeds = [int(s) for s in only.split(",") if s] if only else list(VC_SEEDS)
    assert all(s in VC_SEEDS for s in seeds), seeds
    print(f"creating ONE v2conf batch call with {len(seeds)} input(s) "
          f"via experimental_spawn_map: {seeds}")
    fc = run_vconf.experimental_spawn_map(seeds)
    print(f"v2conf batch call created: {fc.object_id} "
          f"({len(seeds)} inputs running detached; results land in "
          f"/vol/v2conf/runs on volume {VOLUME})")
