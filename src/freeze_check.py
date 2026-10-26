"""Static freeze checks. Reads dataset files for hashes/counts ONLY.

Runs no policies on registered configs. Exit 0 = PASS.
"""

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from gen import sha256_file

ROOT = os.path.join(__file__.rsplit("/", 1)[0], "..")
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(ROOT, "src")
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def main():
    manifest = json.load(open(f"{DATA}/MANIFEST.json"))
    for fn, h in manifest.items():
        check(f"frozen-{fn}", sha256_file(f"{DATA}/{fn}") == h)

    prereg_hash = sha256_file(f"{ROOT}/PREREG.md")
    runlog = open(f"{ROOT}/RUNLOG.md").read()
    check("prereg-unaltered", prereg_hash in runlog, prereg_hash[:12])

    import train as T
    check("train-seeds", T.SEEDS == [101, 102, 103, 104, 105])
    check("train-arms", T.ARMS == ["A1", "A2", "A3", "A4"])
    check("battery-file", T.BATTERY_FILE == "test_battery_v1.json")
    check("es-hypers", (T.POP_PAIRS, T.SIGMA, T.ALPHA, T.FIT_EPS,
                        T.GEN_BUDGET) == (32, 0.1, 0.05, 8, 400))
    r = subprocess.run([sys.executable, f"{SRC}/train.py", "--plan"],
                       capture_output=True, text=True)
    plan = json.loads(r.stdout)
    check("plan-fixed-budget", plan["gen_budget_per_phase"] == 400
          and plan["early_stop_comparative"] is False
          and plan["phase_steps_exact"] == 41280000
          and T.PHASE_STEPS == 41280000)
    n_meta = sum(1 for _ in open(f"{DATA}/train_meta_v1.json"))
    check("manifests-nonempty", n_meta > 100, n_meta)

    pat = re.compile(r"^\s*(import|from)\s+(socket|urllib|requests|ssl|http|httpx)\b")
    bad = []
    for fn in sorted(os.listdir(SRC)):
        if fn.endswith(".py"):
            for i, line in enumerate(open(f"{SRC}/{fn}"), 1):
                if pat.match(line):
                    bad.append(f"{fn}:{i}")
    check("no-network-imports", not bad, bad[:3])

    for arm in T.ARMS:
        r = subprocess.run(
            [sys.executable, f"{SRC}/train.py", "--plan"],
            capture_output=True, text=True)
        check(f"plan-{arm}", r.returncode == 0)

    print("src hashes:")
    for fn in sorted(os.listdir(SRC)):
        if fn.endswith(".py"):
            print(f"  {fn} {sha256_file(f'{SRC}/{fn}')}")
    print("FREEZE-CHECK", "PASS" if all(PASS) else "FAIL")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
