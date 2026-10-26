"""Task families: rewards, termination, success. Wraps env.Arena.

Families A,B,C,D (atomic), G1,G2 (gate-train), I1,H1,H2,H3 (held-out tests).
Rewards are potential-based shaping + terminal bonuses (PREREG 3).
CONTACT_R = 5.0 is an implementation constant (PREREG gap-fill, logged).
"""

import numpy as np

from env import Arena

STEP_PENALTY = -0.01
ARRIVE_R = 6.0
CONTACT_R = 5.0
TOUCH_R = 6.0
ZONE_R = 8.0
DWELL_N = 5
MARGIN_CAP = 30.0
DISP_WINDOW = 40
DISP_MIN = 6.0


class TaskRunner:
    def __init__(self):
        self.env = Arena()

    def reset(self, family, cfg, seed):
        self.family = family
        self.cfg = cfg
        obs = self.env.reset(cfg, seed)
        e = self.env
        self.prev_goal = self._d(e.agent_pos, e.goal)
        self.prev_oz = self._d(e.obj_pos, e.zone)
        self.prev_ao = self._d(e.agent_pos, e.obj_pos)
        self.dwell = 0
        self.hist = [e.agent_pos.copy()]
        self.touched = False
        return obs

    @staticmethod
    def _d(a, b):
        if a is None or b is None:
            return None
        return float(np.linalg.norm(a - b))

    def _contact(self):
        e = self.env
        return e.haz_pos is not None and self._d(e.agent_pos, e.haz_pos) < CONTACT_R

    def step(self, a):
        e = self.env
        obs = e.step(a)
        f = self.family
        r = STEP_PENALTY
        done, success = False, False

        if f in ("B", "G1", "H1", "H2", "I1") and self._contact():
            return obs, -10.0, True, {"success": False}

        if f in ("A", "G1", "I1"):
            cur = self._d(e.agent_pos, e.goal)
            r += 0.1 * (self.prev_goal - cur)
            self.prev_goal = cur
            if cur < ARRIVE_R:
                r += 10.0
                done, success = True, True
        elif f == "B":
            self.hist.append(e.agent_pos.copy())
            ref = self.hist[0] if len(self.hist) < DISP_WINDOW else self.hist[-DISP_WINDOW]
            if float(np.linalg.norm(e.agent_pos - ref)) >= DISP_MIN:
                r += 0.05
            r += min(self._d(e.agent_pos, e.haz_pos), MARGIN_CAP) / MARGIN_CAP * 0.05
            if e.t >= e.MAX_STEPS:
                done, success = True, True
        elif f in ("C", "H1", "H3"):
            oz = self._d(e.obj_pos, e.zone)
            ao = self._d(e.agent_pos, e.obj_pos)
            r += 0.1 * (self.prev_oz - oz) + 0.02 * (self.prev_ao - ao)
            self.prev_oz, self.prev_ao = oz, ao
            self.dwell = self.dwell + 1 if oz < ZONE_R else 0
            if self.dwell >= DWELL_N:
                r += 10.0
                done, success = True, True
        elif f == "D":
            cur = self._d(e.agent_pos, e.goal)
            r += 0.1 * (self.prev_goal - cur)
            self.prev_goal = cur
            if e.agent_pos[1] > e.wall["y"] + 8.0:
                r += 10.0
                done, success = True, True
        elif f in ("G2", "H2"):
            if self._d(e.agent_pos, e.obj_pos) < TOUCH_R:
                self.touched = True
            if not self.touched:
                ao = self._d(e.agent_pos, e.obj_pos)
                r += 0.1 * (self.prev_ao - ao)
                self.prev_ao = ao
            else:
                oz = self._d(e.obj_pos, e.zone)
                r += 0.1 * (self.prev_oz - oz)
                self.dwell = self.dwell + 1 if oz < ZONE_R else 0
                if self.dwell >= DWELL_N:
                    r += 10.0
                    done, success = True, True
            self.prev_oz = self._d(e.obj_pos, e.zone)

        if e.t >= e.MAX_STEPS and not done:
            done = True
        return obs, float(r), done, {"success": success}
