"""Toy-only Modal execution-mechanism probe (UNREGISTERED debugging).

DO NOT INVOKE without explicit user GO.

Purpose: establish ground truth about the Modal execution environment BEFORE
any third registered relaunch, after two operational zero-result attempts:
  attempt 1: plain `modal run` (no --detach) -> app died with the launcher.
  attempt 2: detached relaunch -> all 20 jobs failed at entry in
             run_arm_seed() -> assert_no_net() because the Modal worker had
             socket/urllib/ssl/http preloaded before user code ran.

This probe NEVER calls run_arm_seed() and NEVER trains. Each toy call only:
  1. snapshots sys.modules vs FORBIDDEN at worker entry (before tl imports);
  2. imports the frozen train module (import-only, same baked-image path);
  3. snapshots again and reports the delta (did OUR code load net modules?);
  4. reproduces the current gate outcome (expected: raises remotely);
  5. writes a marker JSON to the PROBE volume (separate from the registered
     volume) and commits, sleeps, re-snapshots (late/lazy loads?), writes a
     done marker and commits.

The local entrypoint spawns N toy calls and returns IMMEDIATELY without
.get(): the local parent exits while the calls sleep, which directly tests
Modal 1.6.2's detached-mode warning ("only keeps the last triggered Modal
function alive after the parent process has been killed or disconnected").

Pre-stated interpretation (recorded before running):
  P1: every marker shows the same entry preload set (expected to include
      socket, urllib, ssl, http) -> establishes the environmental preload.
  P2: delta after importing frozen train == [] -> frozen code loads no
      network modules itself; the attempt-2 clash is purely environmental,
      which justifies a baseline/delta gate as semantics-preserving.
  P3: current assert_no_net() raises remotely -> reproduces attempt 2 in
      isolation (mechanism confirmed, no other cause needed).
  P4: N/N done markers present after parent exit -> multi-spawn detached
      fan-out survives; fewer (e.g. only the last) -> the warning is real
      and the 20-job launch pattern must change (blocking parent or
      remote fan-out on a deployed app).
  P5: late snapshot == entry snapshot -> no lazy mid-run loads by the
      worker (else the fixed gate must tolerate them too).

Cost: N=3 x ~4 min x 1 CPU/1 GiB ~ cents. Wall ~5 min.
"""

import modal

APP = "tiny-learner-probe"
VOLUME = "tiny-learner-probe"
ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"

image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)

FORBIDDEN = ("socket", "urllib", "requests", "ssl", "http", "httpx")


def _forbidden_present():
    import sys

    return sorted(m for m in sys.modules
                  if m.split(".")[0] in FORBIDDEN)


@app.function(image=image, cpu=1.0, memory=1024,
              volumes={"/vol": vol}, timeout=1800)
def probe_env(i, sleep_s):
    """DO NOT INVOKE without explicit user GO. Toy-only, unregistered."""
    import hashlib
    import json
    import os
    import sys
    import time

    entry = _forbidden_present()
    entry_top = sorted(set(m.split(".")[0] for m in entry))

    sys.path.insert(0, "/root/tl/src")
    import train  # noqa: F401  (import-only; run_arm_seed NOT called)
    from evaluate import assert_no_net

    after_import = _forbidden_present()
    delta = sorted(set(after_import) - set(entry))

    try:
        assert_no_net()
        gate_raised: list = []
    except RuntimeError as e:
        gate_raised = [str(e)]

    hashes = {}
    for fn in ("train.py", "evaluate.py"):
        h = hashlib.sha256()
        with open(f"/root/tl/src/{fn}", "rb") as f:
            h.update(f.read())
        hashes[fn] = h.hexdigest()

    os.makedirs("/vol/probe", exist_ok=True)
    marker = {"i": i, "entry_top": entry_top, "entry_detail": entry,
              "after_import": after_import, "import_delta": delta,
              "gate_raised": gate_raised, "src_hashes": hashes}
    with open(f"/vol/probe/marker_{i}.json", "w") as f:
        json.dump(marker, f, indent=1)
    vol.commit()

    time.sleep(sleep_s)

    late = _forbidden_present()
    with open(f"/vol/probe/done_{i}.json", "w") as f:
        json.dump({"i": i, "late": late,
                   "late_delta": sorted(set(late) - set(entry))}, f, indent=1)
    vol.commit()
    marker["late"] = late
    return marker


@app.local_entrypoint()
def main(n=3, sleep_s=180):
    n, sleep_s = int(n), int(sleep_s)
    print(f"spawning {n} toy probe call(s), sleep {sleep_s}s each")
    for i in range(n):
        call = probe_env.spawn(i, sleep_s)
        print(f"spawned probe {i}: {call.object_id}")
    print("parent returning immediately WITHOUT .get() -- check survival "
          "from another terminal, then stop the app.")
