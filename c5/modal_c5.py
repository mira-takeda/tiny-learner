"""Modal launcher for C5 (c5-only).

Operational wrapper only: bakes frozen src/data + diag snapshot (reused
read-only) + c5 snapshot into the image (copy=True) and launches the 3
C5 jobs (seeds 401-403) as ONE batch call via experimental_spawn_map.
Each job runs the C5 trainer in a CLEAN SUBPROCESS (still required:
frozen run_battery asserts no-net and the Modal worker preloads net
modules). Results go to a SEPARATE volume (tiny-learner-c5);
runs_modal/, v1 records, diag/ and pilot/ untouched.

DO NOT INVOKE without explicit user GO.
"""

import modal

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
APP = "tiny-learner-c5"
VOLUME = "tiny-learner-c5"

C5_SEEDS = [401, 402, 403]

image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True)
         .add_local_dir(f"{ROOT}/diag", remote_path="/root/tl/diag", copy=True)
         .add_local_dir(f"{ROOT}/c5", remote_path="/root/tl/c5", copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)
BASE = dict(image=image, cpu=1.0, memory=1024)


@app.function(**BASE, volumes={"/vol": vol}, timeout=86400)
def run_c5(seed):
    """DO NOT INVOKE without explicit user GO. Runs one C5 seed via the
    C5 trainer in a clean subprocess (see module docstring)."""
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root/tl/c5")
    import c5_train

    assert seed in c5_train.C5_SEEDS
    out_json = f"/vol/c5/runs/C5_{seed}.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, "/root/tl/c5/c5_train.py",
           "--seed", str(seed),
           "--data", "/root/tl/data", "--out", "/vol/c5/runs"]
    subprocess.run(cmd, check=True, env=env, timeout=86000)
    vol.commit()
    log = json.load(open(out_json))
    return {"seed": seed, "steps": log["steps"],
            "pure_sampled": log["heldout_pure"]["sampled"]["sr"],
            "pure_greedy": log["heldout_pure"]["greedy"]["sr"]}


@app.local_entrypoint()
def main(only=""):
    seeds = [int(s) for s in only.split(",") if s] if only else list(C5_SEEDS)
    assert all(s in C5_SEEDS for s in seeds), seeds
    print(f"creating ONE C5 batch call with {len(seeds)} input(s) "
          f"via experimental_spawn_map: {seeds}")
    fc = run_c5.experimental_spawn_map(seeds)
    print(f"C5 batch call created: {fc.object_id} "
          f"({len(seeds)} inputs running detached; results land in "
          f"/vol/c5/runs on volume {VOLUME})")
