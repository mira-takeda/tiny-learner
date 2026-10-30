"""Modal launcher for the isolated diagnostic campaign (diag-only).

Operational wrapper only: bakes frozen src/data + diag snapshot into the
image (copy=True) and launches the 24 diagnostic jobs (C0-C4/B0-B2 x seeds
201-203) as ONE batch call via experimental_spawn_map (validated mechanism
in client 1.6.2; public spawn_map fires per-input calls). C4/B2 are main
campaign (direct test of the fitness/checks mismatch hypothesis). Each job
runs the diag trainer in a CLEAN SUBPROCESS (still required: frozen
run_battery asserts no-net and the Modal worker preloads net modules).
Results go to a SEPARATE volume (tiny-learner-diag); runs_modal/ and v1
records untouched.

DO NOT INVOKE without explicit user GO.
"""

import modal

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
APP = "tiny-learner-diag"
VOLUME = "tiny-learner-diag"

DIAG_JOBS = ["C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2"]
DIAG_SEEDS = [201, 202, 203]

image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True)
         .add_local_dir(f"{ROOT}/diag", remote_path="/root/tl/diag", copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)
BASE = dict(image=image, cpu=1.0, memory=1024)


@app.function(**BASE, volumes={"/vol": vol}, timeout=86400)
def run_diag(job, seed):
    """DO NOT INVOKE without explicit user GO. Runs one diag job via the
    diag trainer in a clean subprocess (see module docstring)."""
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root/tl/diag")
    import diag_train

    assert job in diag_train.JOBS and seed in diag_train.DIAG_SEEDS
    out_json = f"/vol/diag/runs/{job}_{seed}.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, "/root/tl/diag/diag_train.py",
           "--job", str(job), "--seed", str(seed),
           "--data", "/root/tl/data", "--out", "/vol/diag/runs"]
    subprocess.run(cmd, check=True, env=env, timeout=86000)
    vol.commit()
    log = json.load(open(out_json))
    return {"job": job, "seed": seed, "gens": log["gens"],
            "acquired": log["acquired"], "steps": log["steps"],
            "heldout": {m: v["sr"] for m, v in log["heldout"].items()}}


@app.local_entrypoint()
def main(only=""):
    jobs = [j for j in only.split(",") if j] if only else list(DIAG_JOBS)
    assert all(j in DIAG_JOBS for j in jobs), jobs
    assert all(s in DIAG_SEEDS for s in DIAG_SEEDS)
    pairs = [(j, s) for j in jobs for s in DIAG_SEEDS]
    print(f"creating ONE diagnostic batch call with {len(pairs)} input(s) "
          f"via experimental_spawn_map: {pairs}")
    fc = run_diag.experimental_spawn_map(
        [j for j, _ in pairs], [s for _, s in pairs])
    print(f"diagnostic batch call created: {fc.object_id} "
          f"({len(pairs)} inputs running detached; results land in "
          f"/vol/diag/runs on volume {VOLUME})")
