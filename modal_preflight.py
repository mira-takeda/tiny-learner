"""Toy-only preflight for the clean-subprocess + single-call launch pattern.

DO NOT INVOKE without explicit user GO. UNREGISTERED debugging, never training.

Proves, BEFORE any third registered attempt:
  1. a clean subprocess on the Modal worker passes the FROZEN assert_no_net()
     gate (defeating the attempt-2 environmental preload) with src/ UNMODIFIED;
  2. the frozen `train.py --plan` CLI runs in that clean env with exact budgets;
  3. ONE experimental_spawn_map call with N inputs survives a REAL client
     disconnect (the open question 20x .spawn() / public .spawn_map() cannot
     answer: public .spawn_map() fires one async call per input, so only the
     experimental single-call variant is tested here).

Protocol: run detached; as soon as the call ID prints, KILL the local client
(Ctrl-C) to force a real disconnect; minutes later verify markers + call state
from another terminal, then stop the app.

Pre-stated pass criteria:
  F1: every worker enters dirty (socket/urllib/ssl/http preloaded) -> control ok.
  F2: clean-subprocess gate snippet exits 0 with GATE-PASS in every input.
  F3: frozen --plan in a clean subprocess reports phase_steps_exact 41280000,
      gen_budget 400, early_stop_comparative False.
  F4: the same gate still RAISES in the dirty parent -> F2's pass comes from
      the clean env, not from a neutered gate.
  F5: N/N done markers exist after the client was killed -> the single-call
      batch survives real disconnect. Fewer -> pattern unsafe, rethink launch.
  F6: clean-subprocess hashes of train.py/evaluate.py match the frozen values
      (train a1977d12dd18, evaluate 6efd31ad1d7e).

Cost: N=3 x ~4 min x 1 CPU/1 GiB ~ cents. Touches ONLY the probe volume:
no run_arm_seed(), no training, nothing under /vol/runs.
"""

import modal

APP = "tiny-learner-preflight"
VOLUME = "tiny-learner-probe"
ROOT = "/Users/satoru/Documents/hackathon/tiny-learner"

# Same image contents as the registered launcher (src+data baked, copy=True):
# the only differences vs the real launch are the command run and the volume.
image = (modal.Image.debian_slim(python_version="3.14")
         .pip_install("numpy==2.5.4")
         .add_local_dir(f"{ROOT}/src", remote_path="/root/tl/src", copy=True)
         .add_local_dir(f"{ROOT}/data", remote_path="/root/tl/data", copy=True))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

app = modal.App(APP)

FORBIDDEN = ("socket", "urllib", "requests", "ssl", "http", "httpx")
FROZEN_SHORT = {"train.py": "a1977d12dd18", "evaluate.py": "6efd31ad1d7e"}

GATE_SNIPPET = (
    "import sys, hashlib; sys.path.insert(0, '/root/tl/src'); "
    "import train; from evaluate import assert_no_net; assert_no_net(); "
    "th = hashlib.sha256(open('/root/tl/src/train.py','rb').read()).hexdigest()[:12]; "
    "eh = hashlib.sha256(open('/root/tl/src/evaluate.py','rb').read()).hexdigest()[:12]; "
    "print('CLEAN-SUBPROCESS-GATE-PASS', th, eh)")


@app.function(image=image, cpu=1.0, memory=1024,
              volumes={"/vol": vol}, timeout=900)
def preflight_gate(i, sleep_s):
    """DO NOT INVOKE without explicit user GO. Toy-only, unregistered."""
    import json
    import os
    import subprocess
    import sys
    import time

    parent = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN)
    parent_top = sorted(set(m.split(".")[0] for m in parent))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    g = subprocess.run([sys.executable, "-c", GATE_SNIPPET],
                       capture_output=True, text=True, env=env, timeout=300)
    assert g.returncode == 0, f"clean gate child failed: {g.stderr[-2000:]}"
    assert "CLEAN-SUBPROCESS-GATE-PASS" in g.stdout, g.stdout[-500:]
    _, th, eh = g.stdout.strip().split()
    assert {"train.py": th, "evaluate.py": eh} == FROZEN_SHORT, (th, eh)

    p = subprocess.run([sys.executable, "/root/tl/src/train.py", "--plan"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == 0, f"clean --plan child failed: {p.stderr[-2000:]}"
    plan = json.loads(p.stdout)
    assert plan["phase_steps_exact"] == 41280000, plan
    assert plan["gen_budget_per_phase"] == 400, plan
    assert plan["early_stop_comparative"] is False, plan

    sys.path.insert(0, "/root/tl/src")
    from evaluate import assert_no_net
    try:
        assert_no_net()
        parent_gate = "PASS-UNEXPECTED"
    except RuntimeError as e:
        parent_gate = f"RAISED-expected: {e}"

    os.makedirs("/vol/preflight", exist_ok=True)
    marker = {"i": i, "parent_preloads": parent_top,
              "gate_stdout": g.stdout.strip(),
              "plan_steps": plan["phase_steps_exact"],
              "parent_gate": parent_gate}
    with open(f"/vol/preflight/marker_{i}.json", "w") as f:
        json.dump(marker, f, indent=1)
    vol.commit()

    time.sleep(sleep_s)

    with open(f"/vol/preflight/done_{i}.json", "w") as f:
        json.dump({"i": i, "survived_disconnect": True}, f, indent=1)
    vol.commit()
    return marker


@app.local_entrypoint()
def main(n=3, sleep_s=180):
    n, sleep_s = int(n), int(sleep_s)
    print(f"creating ONE batch call with {n} toy input(s) "
          f"via experimental_spawn_map (sleep {sleep_s}s each)")
    fc = preflight_gate.experimental_spawn_map(list(range(n)), [sleep_s] * n)
    print(f"batch call created: {fc.object_id}. NOW KILL THIS CLIENT "
          f"(Ctrl-C) to force a REAL disconnect, then verify from another "
          f"terminal and stop the app.")
