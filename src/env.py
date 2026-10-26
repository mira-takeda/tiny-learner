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


class VecArena:
    """Batched Arena. Row n must equal scalar Arena on the same cfg/seed."""

    def __init__(self, n):
        self.n = n
        self.ACTIONS = Arena().ACTIONS
        self.rngs = [np.random.default_rng(0) for _ in range(n)]
        self.agent_pos = np.zeros((n, 2))
        self.agent_vel = np.zeros((n, 2))
        self.haz_pos = np.zeros((n, 2))
        self.haz_speed = np.zeros(n)
        self.haz_mask = np.zeros(n, dtype=bool)
        self.obj_pos = np.zeros((n, 2))
        self.obj_vel = np.zeros((n, 2))
        self.obj_mask = np.zeros(n, dtype=bool)
        self.zone = np.zeros((n, 2))
        self.zone_mask = np.zeros(n, dtype=bool)
        self.goal = np.zeros((n, 2))
        self.goal_mask = np.zeros(n, dtype=bool)
        self.wall_y = np.zeros(n)
        self.wall_gap_x = np.zeros(n)
        self.wall_gap_w = np.zeros(n)
        self.wall_mask = np.zeros(n, dtype=bool)
        self.t = np.zeros(n, dtype=int)

    def reset(self, cfgs, seeds):
        for i, (cfg, seed) in enumerate(zip(cfgs, seeds)):
            self.rngs[i] = np.random.default_rng(seed)
            self.agent_pos[i] = cfg["agent"]
            self.agent_vel[i] = 0.0
            hz = cfg.get("hazard")
            self.haz_mask[i] = hz is not None
            if hz:
                self.haz_pos[i] = hz["pos"]
                self.haz_speed[i] = float(hz.get("speed", 4.5))
            ob = cfg.get("obj")
            self.obj_mask[i] = ob is not None
            if ob:
                self.obj_pos[i] = ob["pos"]
                self.obj_vel[i] = 0.0
            zn = cfg.get("zone")
            self.zone_mask[i] = zn is not None
            if zn:
                self.zone[i] = zn["center"]
            gl = cfg.get("goal")
            self.goal_mask[i] = gl is not None
            if gl:
                self.goal[i] = gl["pos"]
            w = cfg.get("wall")
            self.wall_mask[i] = w is not None
            if w:
                self.wall_y[i] = w["y"]
                self.wall_gap_x[i] = w["gap_x"]
                self.wall_gap_w[i] = w["gap_w"]
            self.t[i] = 0
        return self._obs()

    def _slot(self, P, mask, out):
        d = P - self.agent_pos
        dist = np.sqrt((d * d).sum(axis=1))
        valid = mask & (dist >= 1e-9)
        out[valid, 0:2] = d[valid] / dist[valid, None]
        out[valid, 2] = dist[valid] / Arena.DMAX
        return out

    def _obs(self):
        o = np.zeros((self.n, Arena.OBS_DIM))
        o[:, 0:2] = self.agent_vel / Arena.VMAX
        self._slot(self.goal, self.goal_mask, o[:, 2:5])
        self._slot(self.haz_pos, self.haz_mask, o[:, 5:8])
        self._slot(self.obj_pos, self.obj_mask, o[:, 8:11])
        self._slot(self.zone, self.zone_mask, o[:, 11:14])
        return o

    @staticmethod
    def _bounce(P, V):
        for i in (0, 1):
            lo = P[:, i] < 0.0
            P[lo, i] = 0.0
            V[lo, i] = -0.5 * V[lo, i]
            hi = P[:, i] > Arena.SIZE
            P[hi, i] = Arena.SIZE
            V[hi, i] = -0.5 * V[hi, i]

    def step(self, actions):
        prev_y = self.agent_pos[:, 1].copy()
        self.agent_vel = (self.agent_vel + self.ACTIONS[actions]) * Arena.DAMP
        sp = np.sqrt((self.agent_vel**2).sum(axis=1))
        over = sp > Arena.VMAX
        self.agent_vel[over] *= (Arena.VMAX / sp[over])[:, None]
        self.agent_pos = self.agent_pos + self.agent_vel
        crossed = (prev_y - self.wall_y) * (self.agent_pos[:, 1] - self.wall_y) <= 0.0
        in_gap = (np.abs(self.agent_pos[:, 0] - self.wall_gap_x)
                  <= self.wall_gap_w / 2.0)
        blocked = self.wall_mask & crossed & ~in_gap
        self.agent_pos[blocked, 1] = prev_y[blocked]
        self.agent_vel[blocked, 1] = 0.0
        self._bounce(self.agent_pos, self.agent_vel)
        if self.haz_mask.any():
            d = self.agent_pos - self.haz_pos
            dist = np.sqrt((d * d).sum(axis=1))
            hv = np.zeros((self.n, 2))
            safe = self.haz_mask & (dist >= 1e-9)
            hv[safe] = (d[safe] / dist[safe, None]
                        * self.haz_speed[safe, None])
            for i in range(self.n):
                if self.haz_mask[i]:
                    hv[i] = hv[i] + self.rngs[i].normal(0.0, 0.5, 2)
            self.haz_pos[self.haz_mask] += hv[self.haz_mask]
            P, V = self.haz_pos.copy(), np.zeros((self.n, 2))
            self._bounce(P, V)
            self.haz_pos[self.haz_mask] = P[self.haz_mask]
        if self.obj_mask.any():
            d = self.obj_pos - self.agent_pos
            contact = self.obj_mask & (np.sqrt((d * d).sum(axis=1)) < 6.0)
            self.obj_vel[contact] = (self.obj_vel[contact]
                                     + self.agent_vel[contact] * 0.35)
            self.obj_vel[self.obj_mask] *= 0.92
            self.obj_pos[self.obj_mask] += self.obj_vel[self.obj_mask]
            P = self.obj_pos.copy()
            V = self.obj_vel.copy()
            self._bounce(P, V)
            self.obj_pos[self.obj_mask] = P[self.obj_mask]
            self.obj_vel[self.obj_mask] = V[self.obj_mask]
        self.t += 1
        return self._obs()
