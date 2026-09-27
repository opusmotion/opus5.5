"""09 — THE FAMILY. The two timelines curl into orbits; a third appears.
Sol: deep, deliberate rings. Astra: complexity organising into stepped action.
Luna: one fast, efficient arc. For one beat they align — the only colour in the film."""
import math

import numpy as np

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..timeline import bt
from . import s01_question as S01
from . import s08_time as S08

T0, T1 = bt(8), bt(9) + 0.02

R = 110.0
CY_ = 520.0
CEN = {"sol": np.array([560.0, CY_]), "astra": np.array([960.0, CY_]), "luna": np.array([1360.0, CY_])}
LC = 2 * math.pi * R

CURL0, CURL1 = bt(8, 0.1), bt(8, 1.0)
LUNA0 = bt(8, 0.55)
BEH0 = bt(8, 1.0)
SYNC = bt(8, 3.0)
MERGE0, MERGE1 = bt(8, 3.28), bt(8, 3.62)
FLAT0, FLAT1 = bt(8, 3.66), bt(8, 3.92)
STEP = C.BEAT / 4

HUE = {"sol": P.SOL, "astra": P.ASTRA, "luna": P.LUNA}


def colour(t):
    return E.pulse(t, SYNC - 0.05, 0.05, 0.30)


def tint(name, t, base=P.INK):
    return P.mix(base, HUE[name], colour(t))


def _curl(x_left, y0, L_from, mid_to, t, delay):
    """A timeline (left end fixed while it trims to one circumference) bending into a
    circle whose lowest point is mid_to."""
    u = E.discover(E.seg(t, CURL0 + delay, CURL1 + delay))
    L = E.lerp(L_from, LC, E.smooth(E.seg(t, CURL0 + delay - 0.15, CURL0 + delay + 0.2)))
    k = u / R
    s = np.linspace(-L / 2, L / 2, 180)
    if k < 1e-6:
        x, y = s, np.zeros_like(s)
    else:
        x, y = np.sin(k * s) / k, -(1 - np.cos(k * s)) / k
    mid_from = np.array([x_left + L / 2, y0])
    mid = mid_from + (mid_to - mid_from) * E.settle(t - CURL0 - delay + 0.05, 9.0)
    return np.stack([mid[0] + x, mid[1] + y], -1), u


def merge(t):
    return E.settle(t - MERGE0, 21.0) if t >= MERGE0 else 0.0


def centre(name, t):
    m = merge(t)
    c = CEN[name] + (CEN["astra"] - CEN[name]) * m
    return c + np.array([0.0, (C.CY - CY_) * m])


def flat(t):
    return E.reveal(E.seg(t, FLAT0, FLAT1))


def _circ(c, name, t, r, paint):
    ctr = centre(name, t)
    fx = 1.0 - flat(t)
    if fx > 0.999:
        c.drawCircle(float(ctr[0]), float(ctr[1]), float(r), paint)
    else:
        a = np.linspace(0, 2 * math.pi, 120)
        ry = E.lerp(r, S01.CUR_H / 2, E.reveal(E.seg(t, FLAT0 + 0.12, bt(9))))  # edge-on → the cursor
        pts = np.stack([ctr[0] + r * np.cos(a) * max(fx, 0.0), ctr[1] + ry * np.sin(a)], -1)
        gfx.draw_poly(c, pts, paint, closed=True)


def draw(c, t):
    inner = 1.0 - E.smooth(E.seg(t, MERGE0 - 0.25, MERGE0 + 0.05))
    # orbits: before the curl completes, draw the bending lines
    if t < CURL1 + 0.34:
        for name, y0, L, d in (("sol", S08.SOL_Y, 75 * S08.PXM, 0.0), ("astra", S08.AST_Y, 40 * S08.PXM, 0.2)):
            pts, u = _curl(S08.XL, y0, L, CEN[name] + np.array([0, R]), t, d)
            if u < 0.999:
                col = P.mix(P.GRAY if name == "sol" else P.INK, P.INK, u)
                gfx.draw_poly(c, pts, gfx.stroke(col, 1.0, 1.6))
    ready = t >= CURL1 + 0.32
    stroke_w = 1.6
    if ready:
        for name in ("sol", "astra"):
            _circ(c, name, t, R, gfx.stroke(tint(name, t), 1.0, stroke_w))
    # Luna enters as one fast sweep
    ul = E.reveal(E.seg(t, LUNA0, LUNA0 + 0.3))
    if ul > 0:
        if ul < 1:
            ctr = CEN["luna"]
            gfx.arc(c, ctr[0], ctr[1], R, math.pi / 2, 2 * math.pi * ul, gfx.stroke(P.INK, 1.0, stroke_w))
        else:
            _circ(c, "luna", t, R, gfx.stroke(tint("luna", t), 1.0, stroke_w))
    if t < BEH0 - 0.3:
        return
    ba = E.smooth(E.seg(t, BEH0 - 0.3, BEH0)) * inner
    col = colour(t)
    # SOL — concentric rings, gaps turning slowly, all aligned at the sync
    ctr = centre("sol", t)
    breath = 1 + 0.018 * math.sin(2 * math.pi * (t - SYNC) / (2 * C.BEAT))
    for k, (rf, w) in enumerate(((0.80, 1.5), (0.62, 1.3), (0.44, 1.2), (0.26, 1.1))):
        dr = E.seg(t, BEH0 - 0.3 + 0.08 * k, BEH0 + 0.1 + 0.08 * k)
        if dr <= 0:
            continue
        om = (0.55, -0.42, 0.33, -0.26)[k]
        gap = math.radians(46)
        g0 = -math.pi / 2 + om * (t - SYNC) + gap / 2
        gfx.arc(c, ctr[0], ctr[1], R * rf * breath, g0, (2 * math.pi - gap) * E.traverse(dr),
                gfx.stroke(tint("sol", t, P.SOFT), ba, w))
    # ASTRA — 48 marks: chaos → order → discrete steps
    ctr = centre("astra", t)
    rng = np.random.default_rng(C.SEED + 9)
    n = 48
    ra = rng.uniform(0, 2 * math.pi, n)
    rr = rng.uniform(0.5, 1.05, n)
    rl = rng.uniform(0.05, 0.16, n)
    rt = rng.uniform(-1.2, 1.2, n)
    steps = math.floor((t - SYNC) / STEP + 1e-6)
    frac = (t - SYNC) / STEP - steps
    snap = E.settle(frac * STEP, 60.0)
    rot = 2 * math.pi / n * (steps + snap)
    pa = gfx.stroke(tint("astra", t, P.INK), ba, 1.3)
    import skia

    path = skia.Path()
    for j in range(n):
        o = E.settle(t - BEH0 - 0.012 * j, 11.0)
        ang_o = -math.pi / 2 + 2 * math.pi * j / n + rot
        r0o, r1o = R * (0.70 if j % 4 == 0 else 0.76), R * 0.88
        mid_c = R * rr[j]
        a_c = ra[j]
        l_c = R * rl[j]
        # chaotic state: a short stroke at random place and tilt
        cx_c = ctr[0] + mid_c * math.cos(a_c)
        cy_c = ctr[1] + mid_c * math.sin(a_c)
        dx_c, dy_c = math.cos(a_c + rt[j]) * l_c, math.sin(a_c + rt[j]) * l_c
        p0c = np.array([cx_c - dx_c / 2, cy_c - dy_c / 2])
        p1c = np.array([cx_c + dx_c / 2, cy_c + dy_c / 2])
        p0o = ctr + r0o * np.array([math.cos(ang_o), math.sin(ang_o)])
        p1o = ctr + r1o * np.array([math.cos(ang_o), math.sin(ang_o)])
        p0 = p0c + (p0o - p0c) * o
        p1 = p1c + (p1o - p1c) * o
        path.moveTo(*p0)
        path.lineTo(*p1)
    c.drawPath(path, pa)
    # LUNA — a fast arc with a fading wake
    ctr = centre("luna", t)
    om = 2 * math.pi * 1.6
    head = -math.pi / 2 + om * (t - SYNC)
    for k in range(12):
        a1 = head - k * math.radians(7)
        gfx.arc(c, ctr[0], ctr[1], R, a1 - math.radians(7), math.radians(7.2),
                gfx.stroke(tint("luna", t), ba * (1 - k / 12) ** 1.6, 3.0))
    hp = ctr + R * np.array([math.cos(head), math.sin(head)])
    gfx.dot(c, hp[0], hp[1], 3.4, tint("luna", t), ba)
    gfx.ring(c, ctr[0], ctr[1], R * 0.62, tint("luna", t, P.GRAY), 0.7 * ba, 1.0)
    # the alignment — a single line through three summits
    if col > 0.01:
        y = CY_ - R
        xs = np.linspace(CEN["sol"][0], CEN["luna"][0], 60)
        for i in range(len(xs) - 1):
            u = i / (len(xs) - 2)
            hc = P.mix(P.SOL, P.ASTRA, u * 2) if u < 0.5 else P.mix(P.ASTRA, P.LUNA, (u - 0.5) * 2)
            gfx.line(c, xs[i], y, xs[i + 1] + 0.5, y, gfx.stroke(hc, 0.9 * col, 1.3))
        for name in ("sol", "astra", "luna"):
            gfx.dot(c, CEN[name][0], y, 4.0, HUE[name], col)
    # names
    na = E.smooth(E.seg(t, BEH0, BEH0 + 0.25)) * (1 - E.smooth(E.seg(t, MERGE0 - 0.3, MERGE0)))
    for name, label in (("sol", "Sol"), ("astra", "Astra"), ("luna", "Luna")):
        typo.draw(c, label, CEN[name][0], CY_ + R + 58, "reg", 24, tint(name, t, P.SOFT), na, align="center")


def events():
    ev = [dict(t=CURL0, kind="curl"), dict(t=LUNA0, kind="luna_in"), dict(t=BEH0, kind="family")]
    # Astra steps on 16ths before the sync
    k = 0
    tt = SYNC - 8 * STEP
    while tt < MERGE0:
        ev.append(dict(t=tt, kind="astra_step", k=k))
        tt += STEP
        k += 1
    ev.append(dict(t=SYNC, kind="sync"))
    ev += [dict(t=MERGE0, kind="merge"), dict(t=FLAT0, kind="flatten")]
    return ev
