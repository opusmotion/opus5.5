"""02 — REASON. The line is revealed as the chosen path of a larger search:
trajectories TRY, hit constraints and terminate, survivors converge, one is VERIFIED."""
import math

import numpy as np

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..camera import Cam3
from ..timeline import bt

T0, T1 = bt(1), bt(2) + 0.02

ROOT, GOAL = -450.0, 450.0
PULL0, PULL1 = bt(1, 0.2), bt(1, 2.5)
COMPARE = bt(1, 2.35)
SELECT = bt(1, 2.75)
VERIFY0, VERIFY1 = bt(1, 2.85), bt(1, 3.45)
FADE0, FADE1 = bt(1, 3.3), bt(1, 3.85)
CHECKS = [0.14, 0.32, 0.5, 0.68, 0.86, 1.0]
LABELS = [("reason", 0.20, bt(1, 0.75), 0.30), ("plan", 0.50, bt(1, 1.5), 0.62), ("verify", 0.86, bt(1, 2.85), 0.95)]

NS = 160
_S = np.linspace(0, 1, NS)


def _y(s):
    return ROOT + (GOAL - ROOT) * s


# constraints: horizontal "rules" the search must respect (never across the centre line)
CONSTRAINTS = [(-300, 70, 330), (-205, -470, -120), (-95, 150, 520), (10, -360, -60),
               (120, 60, 280), (210, -540, -230), (300, 250, 560), (-360, -300, -80), (360, -200, -40)]


def _build():
    rng = np.random.default_rng(C.SEED + 2)
    trajs = []

    def make(parent, s0, t0, depth):
        side = rng.choice([-1.0, 1.0])
        amp = side * rng.uniform(35, 560) ** 1.0
        p = rng.uniform(0.75, 1.5)
        f1, ph, w1 = rng.uniform(0.6, 2.2), rng.uniform(0, 2 * math.pi), rng.uniform(0.0, 0.22)
        z = rng.uniform(-1, 1) * 240 * min(1.0, abs(amp) / 400)
        env = np.sin(math.pi * _S) ** p * (1 + w1 * np.sin(2 * math.pi * f1 * _S + ph))
        if parent is None:
            x = amp * env
            zz = np.full(NS, z)
        else:
            k = E.smooth(0) if False else np.clip((_S - s0) / 0.16, 0, 1)
            k = k * k * (3 - 2 * k)
            x = parent["x"] + (amp - parent["amp"]) * env * k
            zz = parent["z"] + (z - parent["z"][0]) * k
        y = _y(_S)
        # first constraint crossing after s0 terminates the trajectory
        s_end = 1.0
        for cy, a, b in CONSTRAINTS:
            idx = np.nonzero((np.abs(y - cy) < (GOAL - ROOT) / NS * 0.6) & (_S > s0 + 0.02))[0]
            for i in idx:
                if a <= x[i] <= b:
                    s_end = min(s_end, _S[i])
                    break
        if s_end >= 1.0 and rng.random() < 0.55:
            s_end = rng.uniform(max(s0 + 0.12, 0.3), 0.93)
        speed = rng.uniform(0.55, 0.95)
        tr = dict(x=x, z=zz, amp=amp, s0=s0, t0=t0, s_end=s_end, speed=speed, depth=depth,
                  survivor=s_end >= 1.0)
        trajs.append(tr)
        if depth < 2:
            for _ in range(int(rng.integers(1, 4 if depth == 0 else 3))):
                cs0 = rng.uniform(s0 + 0.08, max(s0 + 0.1, min(s_end, 0.85)))
                if cs0 < s_end:
                    ct0 = t0 + (cs0 - s0) / speed
                    make(tr, cs0, ct0, depth + 1)

    for i in range(26):
        make(None, 0.0, bt(1, 0.12) + rng.uniform(0, 1.3) ** 1.3, 0)
    return trajs


TRAJ = _build()


def camera(t):
    D = E.expi(560.0, 1650.0, E.discover(E.seg(t, PULL0, PULL1)))
    yaw = 0.20 * E.smooth(E.seg(t, bt(1, 0.6), bt(1, 2.6))) * (1 - E.smooth(E.seg(t, bt(1, 3.0), bt(2))))
    pitch = -0.07 * E.smooth(E.seg(t, bt(1, 0.6), bt(1, 2.6))) * (1 - E.smooth(E.seg(t, bt(1, 3.0), bt(2))))
    pos = np.array([-math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)]) * D
    return Cam3.look_at(pos, (0.0, 0.0, 0.0))


def head(tr, t):
    return min(tr["s_end"], tr["s0"] + max(0.0, t - tr["t0"]) * tr["speed"])


def draw(c, t):
    cam = camera(t)
    fade = 1.0 - E.smooth(E.seg(t, FADE0, FADE1))
    sel = E.smooth(E.seg(t, SELECT, SELECT + 0.25))
    # constraints — dotted rules
    if fade > 0:
        for cy, a, b in CONSTRAINTS:
            appear = E.smooth(E.seg(t, bt(1, 0.3), bt(1, 1.4)))
            xs = np.arange(a, b, 7.0)
            pts, z = cam.project(np.stack([xs, np.full_like(xs, cy), np.zeros_like(xs)], -1))
            pa = gfx.fill(P.GRAY, 0.55 * appear * fade)
            for (px, py) in pts:
                c.drawCircle(float(px), float(py), 0.9, pa)
    # trajectories
    ends = []
    for tr in TRAJ:
        h = head(tr, t)
        if h <= tr["s0"] + 1e-4 or fade <= 0:
            continue
        i0 = int(tr["s0"] * (NS - 1))
        i1 = int(math.ceil(h * (NS - 1)))
        P3 = np.stack([tr["x"][i0:i1 + 1], _y(_S[i0:i1 + 1]), tr["z"][i0:i1 + 1]], -1)
        if len(P3) < 2:
            continue
        # exact head point
        xh = np.interp(h, _S, tr["x"])
        zh = np.interp(h, _S, tr["z"])
        P3[-1] = (xh, _y(h), zh)
        pts, z = cam.project(P3)
        done = h >= tr["s_end"] - 1e-6
        t_done = tr["t0"] + (tr["s_end"] - tr["s0"]) / tr["speed"]
        if tr["survivor"]:
            cmp = E.smooth(E.seg(t, COMPARE, COMPARE + 0.12)) * (1 - sel)
            col = P.mix(P.SOFT, P.INK, cmp)
            a = (0.75 + 0.25 * cmp) * (1 - 0.72 * sel)
        else:
            dim = E.smooth(E.seg(t, t_done, t_done + 0.35)) if done else 0.0
            col = P.mix(P.SOFT, P.GRAPHITE, dim)
            a = (0.8 - 0.25 * dim) * (1 - 0.6 * sel)
        depth_a = float(np.clip(1.25 - (z.mean() - 900) / 2600, 0.35, 1.0))
        c.drawPath(gfx.poly(pts), gfx.stroke(col, a * fade * depth_a, 1.05))
        if not done:
            gfx.dot(c, pts[-1][0], pts[-1][1], 1.9, P.INK, 0.95 * fade)
        elif not tr["survivor"]:
            ends.append((pts, t_done, a * fade * depth_a))
    # terminations: a perpendicular tick, a brief flash when it happens
    for pts, td, a in ends:
        p1, p0 = pts[-1], pts[-2] if len(pts) > 1 else pts[-1]
        d = p1 - p0
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        nx, ny = -d[1] / n, d[0] / n
        flash = math.exp(-max(0.0, t - td) / 0.08)
        gfx.line(c, p1[0] - nx * 4.5, p1[1] - ny * 4.5, p1[0] + nx * 4.5, p1[1] + ny * 4.5,
                 gfx.stroke(P.mix(P.SOFT, P.INK, flash), min(1.0, a + 0.4 * flash), 1.1))
    # trunk — the original line, the chosen path
    trunk_dim = E.smooth(E.seg(t, bt(1, 0.3), bt(1, 1.2))) * (1 - sel)
    pts, _ = cam.project(np.array([[0, ROOT, 0], [0, GOAL, 0]], float))
    ext = np.array([[0, -4000, 0], [0, 4000, 0]], float)
    pre = 1.0 - E.smooth(E.seg(t, PULL0, PULL0 + 0.6))  # before the reveal the line runs off-frame
    e_pts, _ = cam.project(ext)
    top = pts[0] + (e_pts[0] - pts[0]) * pre
    bot = pts[1] + (e_pts[1] - pts[1]) * pre
    gfx.line(c, top[0], top[1], bot[0], bot[1], gfx.stroke(P.mix(P.INK, P.SOFT, trunk_dim), 1.0, 1.6))
    # root and goal
    ra = E.smooth(E.seg(t, PULL0 + 0.3, PULL0 + 0.7))
    gfx.ring(c, pts[0][0], pts[0][1], 3.5, P.INK, ra, 1.2)
    gfx.dot(c, pts[1][0], pts[1][1], 3.0, P.INK, ra)
    # verification pulse and checkpoints
    if t >= VERIFY0:
        u = E.traverse(E.seg(t, VERIFY0, VERIFY1))
        for s in CHECKS:
            ts = VERIFY0 + (VERIFY1 - VERIFY0) * s
            if t >= ts:
                q, _ = cam.project(np.array([0.0, _y(s), 0.0]))
                k = E.settle(t - ts, 26.0)
                gfx.ring(c, q[0], q[1], 5.0 * k, P.INK, 0.9, 1.1)
                gfx.dot(c, q[0], q[1], 1.8 * k, P.INK, 0.9)
        if u < 1.0:
            q, _ = cam.project(np.array([0.0, _y(u), 0.0]))
            gfx.dot(c, q[0], q[1], 2.6, P.INK, 1.0)
    # annotations
    for word, s, ta, tb in LABELS:
        a = E.smooth(E.seg(t, ta, ta + 0.18)) * (1 - E.smooth(E.seg(t, FADE0 + 0.1, FADE1)))
        if a <= 0:
            continue
        q, _ = cam.project(np.array([0.0, _y(s), 0.0]))
        gfx.line(c, q[0] + 6, q[1], q[0] + 20, q[1], gfx.stroke(P.SOFT, 0.7 * a, 1.0))
        typo.draw(c, word, q[0] + 27, q[1] + 6, "reg", 19, P.SOFT, a)


def events():
    ev = []
    for tr in TRAJ:
        if tr["depth"] == 0 or tr["s0"] > 0:
            ev.append(dict(t=tr["t0"], kind="spawn", pan=float(np.clip(tr["amp"] / 900, -1, 1))))
        if not tr["survivor"]:
            td = tr["t0"] + (tr["s_end"] - tr["s0"]) / tr["speed"]
            if td < FADE0:
                ev.append(dict(t=td, kind="terminate", pan=float(np.clip(tr["amp"] / 900, -1, 1))))
    for s in CHECKS:
        ev.append(dict(t=VERIFY0 + (VERIFY1 - VERIFY0) * s, kind="verify", i=CHECKS.index(s)))
    ev.append(dict(t=SELECT, kind="select"))
    return ev
