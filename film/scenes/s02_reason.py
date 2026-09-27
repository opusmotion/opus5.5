"""02 — REASON. The camera pulls back and rolls a quarter turn: the vertical line becomes
the horizontal axis of a search. Trajectories TRY, hit rules and terminate, survivors
converge, one path is SELECTED and VERIFIED."""
import math

import numpy as np

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..camera import Cam3
from ..timeline import bt

T0, T1 = bt(1), bt(2) + 0.02

ROOT, GOAL = -560.0, 560.0  # the axis of reasoning runs along world x
PULL0, PULL1 = bt(1, 0.15), bt(1, 2.4)
ROLL0, ROLL1 = bt(1, 0.1), bt(1, 1.9)
COMPARE = bt(1, 2.35)
SELECT = bt(1, 2.75)
VERIFY0, VERIFY1 = bt(1, 2.85), bt(1, 3.45)
FADE0, FADE1 = bt(1, 3.3), bt(1, 3.85)
CHECKS = [0.14, 0.32, 0.5, 0.68, 0.86, 1.0]
LABELS = [("reason", 0.22, bt(1, 1.0)), ("plan", 0.52, bt(1, 1.7)), ("verify", 0.86, bt(1, 2.85))]

NS = 180
_S = np.linspace(0, 1, NS)


def _x(s):
    return ROOT + (GOAL - ROOT) * s


# rules: gates across the search space, never across the axis itself
GATES = [(-330, 60, 300), (-245, -320, -90), (-130, 100, 380), (-30, -270, -50), (70, 55, 230),
         (165, -380, -150), (255, 140, 360), (345, -240, -40), (430, 40, 210), (-400, -200, -60)]


def _build():
    rng = np.random.default_rng(C.SEED + 2)
    trajs = []
    dx = (GOAL - ROOT) / NS

    def make(parent, s0, t0, depth):
        side = rng.choice([-1.0, 1.0])
        amp = side * rng.uniform(26, 380)
        p = rng.uniform(1.25, 1.7)  # >1: leaves the root tangent to the axis → a lens
        f1, ph, w1 = rng.uniform(0.5, 1.4), rng.uniform(0, 2 * math.pi), rng.uniform(0.0, 0.12)
        z = rng.uniform(-1, 1) * 60 * min(1.0, abs(amp) / 300)
        env = np.sin(math.pi * _S) ** p * (1 + w1 * np.sin(2 * math.pi * f1 * _S + ph))
        if parent is None:
            y = amp * env
            zz = np.full(NS, z)
        else:
            k = np.clip((_S - s0) / 0.16, 0, 1)
            k = k * k * (3 - 2 * k)
            y = parent["y"] + (amp - parent["amp"]) * env * k
            zz = parent["z"] + (z - parent["z"][0]) * k
        x = _x(_S)
        s_end = 1.0
        for gx, a, b in GATES:
            idx = np.nonzero((np.abs(x - gx) < dx * 0.6) & (_S > s0 + 0.02))[0]
            for i in idx:
                if a <= y[i] <= b:
                    s_end = min(s_end, _S[i])
                    break
        if s_end >= 1.0 and rng.random() < 0.5:
            s_end = rng.uniform(max(s0 + 0.12, 0.3), 0.93)
        speed = rng.uniform(1.05, 1.55)
        tr = dict(y=y, z=zz, amp=amp, s0=s0, t0=t0, s_end=s_end, speed=speed, depth=depth, survivor=s_end >= 1.0)
        trajs.append(tr)
        if depth < 2:
            for _ in range(int(rng.integers(1, 4 if depth == 0 else 3))):
                cs0 = rng.uniform(s0 + 0.08, max(s0 + 0.1, min(s_end, 0.85)))
                if cs0 < s_end:
                    make(tr, cs0, t0 + (cs0 - s0) / speed, depth + 1)

    for _ in range(34):
        make(None, 0.0, bt(1, 0.4) + rng.uniform(0, 1.0) ** 1.4, 0)
    return trajs


TRAJ = _build()


def camera(t):
    D = E.expi(560.0, 1650.0, E.discover(E.seg(t, PULL0, PULL1)))
    roll = -0.5 * math.pi * (1.0 - E.discover(E.seg(t, ROLL0, ROLL1)))
    return Cam3((0.0, 0.0, -D), 0.0, 0.0, roll)


def head(tr, t):
    return min(tr["s_end"], tr["s0"] + max(0.0, t - tr["t0"]) * tr["speed"])


def trunk_screen(t):
    pts, _ = camera(t).project(np.array([[ROOT, 0, 0], [GOAL, 0, 0]], float))
    return pts


def draw(c, t):
    cam = camera(t)
    fade = 1.0 - E.smooth(E.seg(t, FADE0, FADE1))
    sel = E.smooth(E.seg(t, SELECT, SELECT + 0.25))
    # rules — dotted gates
    if fade > 0:
        appear = E.smooth(E.seg(t, bt(1, 0.6), bt(1, 1.6)))
        pa = gfx.fill(P.GRAY, 0.6 * appear * fade)
        for gx, a, b in GATES:
            ys = np.arange(a, b, 7.0)
            pts, _ = cam.project(np.stack([np.full_like(ys, gx), ys, np.zeros_like(ys)], -1))
            for (px, py) in pts:
                c.drawCircle(float(px), float(py), 0.9, pa)
    ends = []
    for tr in TRAJ:
        h = head(tr, t)
        if h <= tr["s0"] + 1e-4 or fade <= 0:
            continue
        i0 = int(tr["s0"] * (NS - 1))
        i1 = int(math.ceil(h * (NS - 1)))
        P3 = np.stack([_x(_S[i0:i1 + 1]), tr["y"][i0:i1 + 1], tr["z"][i0:i1 + 1]], -1)
        if len(P3) < 2:
            continue
        P3[-1] = (_x(h), np.interp(h, _S, tr["y"]), np.interp(h, _S, tr["z"]))
        pts, z = cam.project(P3)
        done = h >= tr["s_end"] - 1e-6
        t_done = tr["t0"] + (tr["s_end"] - tr["s0"]) / tr["speed"]
        if tr["survivor"]:
            cmp = E.smooth(E.seg(t, COMPARE, COMPARE + 0.12)) * (1 - sel)
            col = P.mix(P.SOFT, P.INK, cmp)
            a = (0.62 + 0.38 * cmp) * (1 - 0.72 * sel)
        else:
            dim = E.smooth(E.seg(t, t_done, t_done + 0.35)) if done else 0.0
            col = P.mix(P.SOFT, P.GRAPHITE, dim)
            a = (0.58 - 0.2 * dim) * (1 - 0.6 * sel)
        c.drawPath(gfx.poly(pts), gfx.stroke(col, a * fade, 1.0))
        if not done:
            gfx.dot(c, pts[-1][0], pts[-1][1], 1.9, P.INK, 0.95 * fade)
        elif not tr["survivor"]:
            ends.append((pts, t_done, a * fade))
    for pts, td, a in ends:
        p1, p0 = pts[-1], pts[-2]
        d = p1 - p0
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        nx, ny = -d[1] / n, d[0] / n
        flash = math.exp(-max(0.0, t - td) / 0.08)
        gfx.line(c, p1[0] - nx * 4.5, p1[1] - ny * 4.5, p1[0] + nx * 4.5, p1[1] + ny * 4.5,
                 gfx.stroke(P.mix(P.SOFT, P.INK, flash), min(1.0, a + 0.4 * flash), 1.1))
    # the axis — the original line, dims while alternatives are explored, then is chosen
    trunk_dim = E.smooth(E.seg(t, bt(1, 0.6), bt(1, 1.4))) * (1 - sel)
    pts = trunk_screen(t)
    gfx.line(c, pts[0][0], pts[0][1], pts[1][0], pts[1][1], gfx.stroke(P.mix(P.INK, P.SOFT, trunk_dim), 1.0, 1.6))
    ra = E.smooth(E.seg(t, bt(1, 1.2), bt(1, 1.6)))
    gfx.ring(c, pts[0][0], pts[0][1], 3.6, P.INK, ra, 1.2)
    gfx.dot(c, pts[1][0], pts[1][1], 3.0, P.INK, ra)
    if t >= VERIFY0:
        u = E.traverse(E.seg(t, VERIFY0, VERIFY1))
        for s in CHECKS:
            ts = VERIFY0 + (VERIFY1 - VERIFY0) * s
            if t >= ts:
                q, _ = cam.project(np.array([_x(s), 0.0, 0.0]))
                k = E.settle(t - ts, 26.0)
                gfx.ring(c, q[0], q[1], 5.0 * k, P.INK, 0.9, 1.1)
                gfx.dot(c, q[0], q[1], 1.8 * k, P.INK, 0.9)
        if u < 1.0:
            q, _ = cam.project(np.array([_x(u), 0.0, 0.0]))
            gfx.dot(c, q[0], q[1], 2.6, P.INK, 1.0)
    for word, s, ta in LABELS:
        a = E.smooth(E.seg(t, ta, ta + 0.18)) * (1 - E.smooth(E.seg(t, FADE0 + 0.1, FADE1)))
        if a <= 0:
            continue
        q0, _ = cam.project(np.array([_x(s), -8.0, 0.0]))
        q1, _ = cam.project(np.array([_x(s), -24.0, 0.0]))
        qa, _ = cam.project(np.array([_x(s), -44.0, 0.0]))
        gfx.line(c, q0[0], q0[1], q1[0], q1[1], gfx.stroke(P.SOFT, 0.7 * a, 1.0))
        typo.draw(c, word, qa[0], qa[1] + 6, "reg", 19, P.SOFT, a, align="center")


def events():
    ev = []
    for tr in TRAJ:
        pan = float(np.clip(_x(0.1) / 900, -1, 1))
        ev.append(dict(t=tr["t0"], kind="spawn", pan=-0.6 + 0.2 * tr["s0"] * 6 if False else float(np.clip(-0.7 + 1.4 * tr["s0"], -1, 1))))
        if not tr["survivor"]:
            td = tr["t0"] + (tr["s_end"] - tr["s0"]) / tr["speed"]
            if td < FADE0:
                ev.append(dict(t=td, kind="terminate", pan=float(np.clip(-0.7 + 1.4 * tr["s_end"], -1, 1))))
    for i, s in enumerate(CHECKS):
        ev.append(dict(t=VERIFY0 + (VERIFY1 - VERIFY0) * s, kind="verify", i=i))
    ev.append(dict(t=SELECT, kind="select"))
    return ev
