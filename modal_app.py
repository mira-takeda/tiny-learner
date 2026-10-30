"""Modal launch layer for the frozen tiny-learner experiment.

Operational cloud-execution wrapper only: ships the frozen src/data snapshot
into the image (copy=True) and launches the 20 registered (arm, seed) jobs
on the Modal volume. All toy/diag/verify actions were removed after
successful verification; they remain in git history. This file launches
registered training and nothing else.

Two operational fixes over the attempt-1/2 launcher (frozen src/ untouched):
  1. Each job runs the frozen `train.py` CLI in a CLEAN SUBPROCESS (fresh
     interpreter): the Modal worker preloads socket/urllib/ssl/http, which
     trips the frozen assert_no_net() gate in-process (attempt-2 failure).
     The frozen bytes run unmodified in the child; the parent only launches,
     commits the volume, and reads back the summary artifact.
  2. All jobs launch as ONE batch call via experimental_spawn_map (a single
     FunctionCall with N inputs), not 20 independent .spawn() calls. Note:
     public .spawn_map() in client 1.6.2 fires one async call PER INPUT and
     returns None (see parallel_map.py), so it has the same detached-mode
     exposure as 20x .spawn(); only experimental_spawn_map creates the
     single multi-input call.

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
    """DO NOT INVOKE without explicit user GO. Runs one frozen arm/seed job
    via the frozen CLI in a clean subprocess (see module docstring)."""
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root/tl/src")
    import train  # import-only in the parent; the gate fires solely in the child

    assert arm in train.ARMS and seed in train.SEEDS
    out_json = f"/vol/runs/{arm}_{seed}.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, "/root/tl/src/train.py",
           "--arm", str(arm), "--seed", str(seed),
           "--data", "/root/tl/data", "--out", "/vol/runs"]
    # Inherits stdout/stderr: progress stays in the container logs.
    # timeout < function timeout so a hang fails cleanly before the kill.
    subprocess.run(cmd, check=True, env=env, timeout=86000)
    vol.commit()
    log = json.load(open(out_json))
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
    print(f"creating ONE batch call with {len(jobs)} input(s) "
          f"via experimental_spawn_map: {jobs}")
    fc = run_registered.experimental_spawn_map(
        [a for a, _ in jobs], [s for _, s in jobs])
    print(f"batch call created: {fc.object_id} "
          f"({len(jobs)} inputs running detached; results land in /vol/runs)")
