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


class VecTaskRunner:
    """Batched TaskRunner, one family per instance. Row n must equal the
    scalar runner on the same cfg/seed. Done rows freeze (absorbing)."""

    CONTACT_FAMS = ("B", "G1", "H1", "H2", "I1")

    def __init__(self, n):
        from env import VecArena
        self.env = VecArena(n)
        self.n = n

    def reset(self, family, cfgs, seeds):
        self.family = family
        obs = self.env.reset(cfgs, seeds)
        e = self.env
        self.prev_goal = self._d(e.agent_pos, e.goal)
        self.prev_oz = self._d(e.obj_pos, e.zone)
        self.prev_ao = self._d(e.agent_pos, e.obj_pos)
        self.dwell = np.zeros(self.n, dtype=int)
        self.hist = np.zeros((self.n, DISP_WINDOW, 2))
        self.hist[:, 0] = e.agent_pos
        self.hcount = np.ones(self.n, dtype=int)
        self.touched = np.zeros(self.n, dtype=bool)
        self.rets = np.zeros(self.n)
        self.done = np.zeros(self.n, dtype=bool)
        self.success = np.zeros(self.n, dtype=bool)
        self.active = np.ones(self.n, dtype=bool)
        return obs

    @staticmethod
    def _d(A, B):
        return np.sqrt(((A - B) ** 2).sum(axis=1))

    def _contact(self):
        e = self.env
        return e.haz_mask & (self._d(e.agent_pos, e.haz_pos) < CONTACT_R)

    def step(self, actions):
        e = self.env
        f = self.family
        obs = e.step(actions)
        r = np.full(self.n, STEP_PENALTY)
        done = np.zeros(self.n, dtype=bool)
        succ = np.zeros(self.n, dtype=bool)

        if f in self.CONTACT_FAMS:
            hit = self._contact()
            r[hit] = -10.0
            done[hit] = True
            live = ~hit
        else:
            live = np.ones(self.n, dtype=bool)

        if f in ("A", "G1", "I1"):
            cur = self._d(e.agent_pos, e.goal)
            r[live] += 0.1 * (self.prev_goal[live] - cur[live])
            self.prev_goal = cur
            arr = live & (cur < ARRIVE_R)
            r[arr] += 10.0
            done[arr] = True
            succ[arr] = True
        elif f == "B":
            pos = (self.hcount % DISP_WINDOW)
            self.hist[np.arange(self.n), pos] = e.agent_pos
            self.hcount += 1
            ref = np.where((self.hcount < DISP_WINDOW)[:, None],
                           self.hist[:, 0],
                           self.hist[np.arange(self.n),
                                     (self.hcount - DISP_WINDOW) % DISP_WINDOW])
            disp = self._d(e.agent_pos, ref)
            r[live & (disp >= DISP_MIN)] += 0.05
            r[live] += (np.minimum(self._d(e.agent_pos, e.haz_pos),
                                   MARGIN_CAP)[live] / MARGIN_CAP * 0.05)
            tout = live & (e.t >= Arena.MAX_STEPS)
            done[tout] = True
            succ[tout] = True
        elif f in ("C", "H1", "H3"):
            oz = self._d(e.obj_pos, e.zone)
            ao = self._d(e.agent_pos, e.obj_pos)
            r[live] += (0.1 * (self.prev_oz[live] - oz[live])
                        + 0.02 * (self.prev_ao[live] - ao[live]))
            self.prev_oz, self.prev_ao = oz, ao
            inside = oz < ZONE_R
            self.dwell = np.where(inside, self.dwell + 1, 0)
            got = live & (self.dwell >= DWELL_N)
            r[got] += 10.0
            done[got] = True
            succ[got] = True
        elif f == "D":
            cur = self._d(e.agent_pos, e.goal)
            r[live] += 0.1 * (self.prev_goal[live] - cur[live])
            self.prev_goal = cur
            out = live & (e.agent_pos[:, 1] > e.wall_y + 8.0)
            r[out] += 10.0
            done[out] = True
            succ[out] = True
        elif f in ("G2", "H2"):
            touch = self._d(e.agent_pos, e.obj_pos) < TOUCH_R
            self.touched[live & touch] = True
            pre = live & ~self.touched
            ao = self._d(e.agent_pos, e.obj_pos)
            r[pre] += 0.1 * (self.prev_ao[pre] - ao[pre])
            self.prev_ao = ao
            post = live & self.touched
            oz = self._d(e.obj_pos, e.zone)
            r[post] += 0.1 * (self.prev_oz[post] - oz[post])
            inside = oz < ZONE_R
            self.dwell = np.where(live & inside, self.dwell + 1, 0)
            got = post & (self.dwell >= DWELL_N)
            r[got] += 10.0
            done[got] = True
            succ[got] = True
            self.prev_oz = self._d(e.obj_pos, e.zone)

        if f != "B":
            tout = live & ~done & (e.t >= Arena.MAX_STEPS)
            done[tout] = True

        newly = done & self.active
        self.rets[self.active] += r[self.active]
        self.success[newly] |= succ[newly]
        self.done |= done
        self.active &= ~self.done
        return obs, r, self.done.copy(), self.success.copy()
