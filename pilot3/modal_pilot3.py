"""Modal launcher for pilot3 (pilot3-only).

Operational wrapper only: bakes frozen src/data + pilot snapshot
(reused read-only) + pilot3 snapshot into the image (copy=True) and
launches the 3 paired seeds (301-303, sequential A->B->C(1200)->D) as
ONE batch call via experimental_spawn_map (validated in client 1.6.2).
Each seed runs the pilot3 trainer in a CLEAN SUBPROCESS (still
required: frozen run_battery asserts no-net and the Modal worker
preloads net modules). Results go to a SEPARATE volume
(tiny-learner-pilot3); pilot/, pilot2/ and all other records untouched.

DO NOT INVOKE without explicit user GO.
"""

import modal

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
APP = "tiny-learner-pilot3"
VOLUME = "tiny-learner-pilot3"

PILOT3_SEEDS = [301, 302, 303]

image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True)
         .add_local_dir(f"{ROOT}/pilot", remote_path="/root/tl/pilot",
                        copy=True)
         .add_local_dir(f"{ROOT}/pilot3", remote_path="/root/tl/pilot3",
                        copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)
BASE = dict(image=image, cpu=1.0, memory=1024)


@app.function(**BASE, volumes={"/vol": vol}, timeout=86400)
def run_pilot3(seed):
    """DO NOT INVOKE without explicit user GO. Runs one pilot3 seed via
    the pilot3 trainer in a clean subprocess (see module docstring)."""
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root/tl/pilot3")
    import pilot3_train

    assert seed in pilot3_train.PILOT3_SEEDS
    out_json = f"/vol/pilot3/runs/P3_{seed}.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, "/root/tl/pilot3/pilot3_train.py",
           "--seed", str(seed),
           "--data", "/root/tl/data", "--out", "/vol/pilot3/runs"]
    subprocess.run(cmd, check=True, env=env, timeout=86000)
    vol.commit()
    log = json.load(open(out_json))
    return {"seed": seed, "steps": log["total_steps"],
            "phase_end": {f: log["heldout_after"][f"post_{f}"][f]["sampled"]["sr"]
                          for f in ["A", "B", "C", "D"]}}


@app.local_entrypoint()
def main(only=""):
    seeds = [int(s) for s in only.split(",") if s] if only else list(PILOT3_SEEDS)
    assert all(s in PILOT3_SEEDS for s in seeds), seeds
    print(f"creating ONE pilot3 batch call with {len(seeds)} input(s) "
          f"via experimental_spawn_map: {seeds}")
    fc = run_pilot3.experimental_spawn_map(seeds)
    print(f"pilot3 batch call created: {fc.object_id} "
          f"({len(seeds)} inputs running detached; results land in "
          f"/vol/pilot3/runs on volume {VOLUME})")
