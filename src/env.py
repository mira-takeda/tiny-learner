"""Environments for tiny-learner. Numpy only. Seeded. No rendering.

LineWorld: 1D optimizer sanity env (PREREG 6). NEVER used in registered evals.
Arena: 2D continuous arena (gen-v1 constants, PREREG 3). Task rewards live in tasks.py.
"""

import numpy as np


class LineWorld:
    """1D track. Start 50, target 100. Actions {0:left, 1:stay, 2:right}."""

    TRACK = 100.0
    START = 50.0
    TARGET = 100.0
    ARRIVE_R = 2.0
    MAX_STEPS = 100
    STEP = 2.0
    N_ACTIONS = 3
    OBS_DIM = 2

    def reset(self):
        self.pos = float(self.START)
        self.t = 0
        return self._obs()

    def _obs(self):
        return np.array(
            [self.pos / self.TRACK, (self.TARGET - self.pos) / self.TRACK],
            dtype=np.float64,
        )

    def step(self, a):
        prev = abs(self.TARGET - self.pos)
        move = (int(a) - 1) * self.STEP
        self.pos = float(np.clip(self.pos + move, 0.0, self.TRACK))
        self.t += 1
        cur = abs(self.TARGET - self.pos)
        reward = (prev - cur) * 0.1 - 0.01
        done, success = False, False
        if cur <= self.ARRIVE_R:
            reward += 10.0
            done, success = True, True
        elif self.t >= self.MAX_STEPS:
            done = True
        return self._obs(), float(reward), done, {"success": success}


class Arena:
    """2D point-mass arena. Seeded. Physics + observations only.

    cfg keys (agent required, rest optional):
      agent: [x, y]
      goal: {"pos": [x, y]}            -> obs slot S1
      hazard: {"pos": [x, y], "speed": float} -> obs slot S2
      obj: {"pos": [x, y]}             -> obs slot S3
      zone: {"center": [x, y]}         -> obs slot S4
      wall: {"y": float, "gap_x": float, "gap_w": float}
    """

    SIZE = 100.0
    ACC_MAG = 2.0
    DAMP = 0.90
    VMAX = 5.0
    MAX_STEPS = 200
    OBS_DIM = 14
    N_ACTIONS = 9
    DMAX = 100.0 * np.sqrt(2.0)

    _DIRS = tuple(
        (np.cos(k * np.pi / 4.0), np.sin(k * np.pi / 4.0)) for k in range(8)
    ) + ((0.0, 0.0),)

    def __init__(self):
        self.ACTIONS = np.array(
            [(dx * self.ACC_MAG, dy * self.ACC_MAG) for dx, dy in self._DIRS]
        )
        self.rng = np.random.default_rng(0)

    def reset(self, cfg, seed):
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.agent_pos = np.array(cfg["agent"], dtype=np.float64)
        self.agent_vel = np.zeros(2)
        hz = cfg.get("hazard")
        self.haz_pos = np.array(hz["pos"], dtype=np.float64) if hz else None
        self.haz_speed = float(hz.get("speed", 4.5)) if hz else 0.0
        ob = cfg.get("obj")
        self.obj_pos = np.array(ob["pos"], dtype=np.float64) if ob else None
        self.obj_vel = np.zeros(2)
        zn = cfg.get("zone")
        self.zone = np.array(zn["center"], dtype=np.float64) if zn else None
        gl = cfg.get("goal")
        self.goal = np.array(gl["pos"], dtype=np.float64) if gl else None
        self.wall = cfg.get("wall")
        self.t = 0
        return self._obs()

    def _slot(self, p):
        if p is None:
            return np.zeros(3)
        d = p - self.agent_pos
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return np.zeros(3)
        return np.array([d[0] / dist, d[1] / dist, dist / self.DMAX])

    def _obs(self):
        o = np.zeros(self.OBS_DIM)
        o[0:2] = self.agent_vel / self.VMAX
        o[2:5] = self._slot(self.goal)
        o[5:8] = self._slot(self.haz_pos)
        o[8:11] = self._slot(self.obj_pos)
        o[11:14] = self._slot(self.zone)
        return o

    @staticmethod
    def _bounce(p, v):
        for i in (0, 1):
            if p[i] < 0.0:
                p[i] = 0.0
                v[i] = -0.5 * v[i]
            elif p[i] > Arena.SIZE:
                p[i] = Arena.SIZE
                v[i] = -0.5 * v[i]
        return p, v

    def _wall_block(self, prev_y):
        w = self.wall
        if w is None:
            return
        crossed = (prev_y - w["y"]) * (self.agent_pos[1] - w["y"]) <= 0.0
        if not crossed:
            return
        in_gap = abs(self.agent_pos[0] - w["gap_x"]) <= w["gap_w"] / 2.0
        if not in_gap:
            self.agent_pos[1] = prev_y
            self.agent_vel[1] = 0.0

    def step(self, a):
        prev_y = self.agent_pos[1]
        self.agent_vel = (self.agent_vel + self.ACTIONS[int(a)]) * self.DAMP
        sp = float(np.linalg.norm(self.agent_vel))
        if sp > self.VMAX:
            self.agent_vel *= self.VMAX / sp
        self.agent_pos = self.agent_pos + self.agent_vel
        self._wall_block(prev_y)
        self.agent_pos, self.agent_vel = self._bounce(self.agent_pos, self.agent_vel)
        if self.haz_pos is not None:
            d = self.agent_pos - self.haz_pos
            n = float(np.linalg.norm(d))
            hv = np.zeros(2) if n < 1e-9 else d / n * self.haz_speed
            hv = hv + self.rng.normal(0.0, 0.5, 2)
            self.haz_pos = self.haz_pos + hv
            self.haz_pos, _ = self._bounce(self.haz_pos, np.zeros(2))
        if self.obj_pos is not None:
            if float(np.linalg.norm(self.obj_pos - self.agent_pos)) < 6.0:
                self.obj_vel = self.obj_vel + self.agent_vel * 0.35
            self.obj_vel = self.obj_vel * 0.92
            self.obj_pos = self.obj_pos + self.obj_vel
            self.obj_pos, self.obj_vel = self._bounce(self.obj_pos, self.obj_vel)
        self.t += 1
        return self._obs()


def random_arena_config(rng):
    def pt():
        return [float(rng.uniform(10, 90)), float(rng.uniform(10, 90))]

    return {
        "agent": pt(),
        "goal": {"pos": pt()},
        "hazard": {"pos": pt(), "speed": 4.0},
        "obj": {"pos": pt()},
        "zone": {"center": pt()},
    }
