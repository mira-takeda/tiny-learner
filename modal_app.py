"""Modal launch layer for the frozen tiny-learner experiment.

Operational cloud-execution wrapper only: ships the frozen src/data snapshot
into the image (copy=True) and launches the 20 registered (arm, seed) jobs
concurrently on the Modal volume. All toy/diag/verify actions were removed
after successful verification; they remain in git history. This file launches
registered training and nothing else.

DO NOT INVOKE without explicit user GO.
"""

import modal

ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"
APP = "tiny-learner"
VOLUME = "tiny-learner-runs"

# Launch manifest. Source of truth is src/train.py (ARMS/SEEDS); duplicated as
# literals here because the host-side Modal driver interpreter has no numpy to
# import train. run_registered re-asserts membership remotely before any work.
ARMS = ["A1", "A2", "A3", "A4"]
SEEDS = [101, 102, 103, 104, 105]

# copy=True bakes an immutable snapshot into the image: remote code/data are
# pinned at build time and immune to later local edits.
image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)
BASE = dict(image=image, cpu=1.0, memory=1024)


@app.function(**BASE, volumes={"/vol": vol}, timeout=86400)
def run_registered(arm, seed):
    """DO NOT INVOKE without explicit user GO. Runs one frozen arm/seed job."""
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
def main(action="launch_all", arm=None, seed=None):
    if action != "launch_all":
        raise SystemExit(f"unknown action: {action}")
    arms = [arm] if arm else list(ARMS)
    seeds = [int(seed)] if seed else list(SEEDS)
    assert all(a in ARMS for a in arms), arms
    assert all(s in SEEDS for s in seeds), seeds
    jobs = [(a, s) for a in arms for s in seeds]
    print(f"launching {len(jobs)} registered job(s): {jobs}")
    for a, s in jobs:
        call = run_registered.spawn(a, s)
        print(f"spawned {a} {s}: {call.object_id}")
