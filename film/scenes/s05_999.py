"""05 — 99.9 / ARC-AGI-3. One curve becomes hundreds of attempted trajectories; they
converge, and their negative space reveals the number before it resolves."""
import math

import numpy as np
import skia

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..camera import Cam2
from ..timeline import bt
from . import s04_build as S04

T0, T1 = bt(4), bt(5) + 0.05

SIZE = 440.0
KEY = "dlight"
NUM = "99.9"
PCT_SIZE = SIZE * 0.34
GAP = 22.0

_numw = typo.width(NUM, KEY, SIZE)
_pctw = typo.width("%", KEY, PCT_SIZE)
TOTAL_W = _numw + GAP + _pctw
NX = C.CX - TOTAL_W / 2
BASE = C.CY + typo.CAP * SIZE / 2
PX = NX + _numw + GAP
PCT_BASE = BASE - typo.CAP * SIZE + typo.CAP * PCT_SIZE

GLYPHS = typo.glyphs(NUM, KEY, SIZE, NX, BASE)
NUM_PATH = skia.Path()
for ch, gp, _, _ in GLYPHS:
    if ch != ".":
        NUM_PATH.addPath(gp)
DOT_PATH = [gp for ch, gp, _, _ in GLYPHS if ch == "."][0]
_db = DOT_PATH.getBounds()
DOT = np.array([(_db.left() + _db.right()) / 2, (_db.top() + _db.bottom()) / 2])
DOT_R = (_db.right() - _db.left()) / 2
ALL_PATH = skia.Path()
ALL_PATH.addPath(NUM_PATH)
ALL_PATH.addPath(DOT_PATH)

SPREAD0, SPREAD1 = bt(4, 0.0), bt(4, 0.9)
CONV0, CONV1 = bt(4, 0.8), bt(4, 1.9)
KNOCK0, KNOCK1 = bt(4, 0.9), bt(4, 1.85)
RESOLVE = bt(4, 2.0)
LABEL = bt(4, 2.3)
ZOOM0, ZOOM1 = bt(4, 3.15), bt(5, 0.0)

NL = 170
Y_TOP, Y_BOT = 120.0, 960.0


def _lines():
    rng = np.random.default_rng(C.SEED + 5)
    ys = np.linspace(Y_TOP, Y_BOT, NL)
    return dict(y=ys, f1=rng.uniform(1.5, 4.5, NL), f2=rng.uniform(4, 9, NL), p1=rng.uniform(0, 6.28, NL),
                p2=rng.uniform(0, 6.28, NL), a=rng.uniform(20, 70, NL), t0=rng.uniform(0, 0.38, NL),
                sp=rng.uniform(2600, 4200, NL))


LN = _lines()
XS = np.linspace(-30, C.W + 30, 200)


def camera(t):
    u = E.accel(E.seg(t, ZOOM0, ZOOM1))
    z = E.expi(1.0, 40.0, u)
    pan = E.smooth(E.seg(t, ZOOM0, ZOOM1 - 0.1))
    cx = E.lerp(C.CX, DOT[0], pan)
    cy = E.lerp(C.CY, DOT[1], pan)
    # keep the decimal point on its screen position while zooming, drifting it to centre
    return Cam2(cx, cy, z) if z > 1.0001 else Cam2()


def zoom_state(t):
    u = E.accel(E.seg(t, ZOOM0, ZOOM1))
    return E.expi(1.0, 40.0, u), E.smooth(E.seg(t, ZOOM0, ZOOM1 - 0.1))


def dot_screen(t):
    z, pan = zoom_state(t)
    # screen position of the decimal point: glides to the centre
    sx = E.lerp(DOT[0], C.CX, pan)
    sy = E.lerp(DOT[1], C.CY, pan)
    return sx, sy, z


def draw(c, t):
    seed = S04.curve_pts(bt(4))
    seed_y = np.interp(XS, seed[:, 0], seed[:, 1])
    spread = E.reveal(E.seg(t, SPREAD0, SPREAD1))
    conv = E.smoother(E.seg(t, CONV0, CONV1))
    fade_lines = E.smooth(E.seg(t, RESOLVE, RESOLVE + 0.22))
    sx, sy, z = dot_screen(t)
    # zoom transform: world point W → screen (sx,sy) + (W - DOT) * z
    c.save()
    c.translate(sx, sy)
    c.scale(z, z)
    c.translate(-DOT[0], -DOT[1])
    if fade_lines < 1:
        ph = t * 2.2
        for i in range(NL):
            yi = LN["y"][i]
            wob = (1 - conv) * LN["a"][i] * (np.sin(XS / 1920 * 6.28 * LN["f1"][i] + LN["p1"][i] + ph)
                                             + 0.35 * np.sin(XS / 1920 * 6.28 * LN["f2"][i] + LN["p2"][i] - ph * 1.7))
            y = seed_y + (yi - seed_y) * spread + wob * spread
            head = (t - SPREAD0 - LN["t0"][i]) * LN["sp"][i] - 30
            if head <= -30:
                continue
            tail = -30 + (C.W + 60) * E.accel(fade_lines)
            m = (XS >= tail) & (XS <= head)
            if m.sum() < 2:
                continue
            pts = np.stack([XS[m], y[m]], -1)
            a = (0.36 + 0.3 * conv) * (1 - fade_lines)
            c.drawPath(gfx.poly(pts), gfx.stroke(P.SOFT, a, 1.0))
            if head < C.W:
                gfx.dot(c, pts[-1][0], pts[-1][1], 1.4, P.INK, 0.8 * (1 - fade_lines))
    # knockout: the number exists first as absence
    k = E.smooth(E.seg(t, KNOCK0, KNOCK1))
    if k > 0 and t < RESOLVE + 0.2:
        c.drawPath(ALL_PATH, gfx.fill(P.BLACK, k))
    # resolve
    f = E.reveal(E.seg(t, RESOLVE, RESOLVE + 0.16))
    if f > 0:
        c.drawPath(NUM_PATH, gfx.fill(P.INK, f))
        pa = E.smooth(E.seg(t, RESOLVE + 0.12, RESOLVE + 0.35))
        typo.draw(c, "%", PX, PCT_BASE, KEY, PCT_SIZE, P.SOFT, pa)
        la = E.smooth(E.seg(t, LABEL, LABEL + 0.25)) * (1 - E.smooth(E.seg(t, ZOOM0, ZOOM0 + 0.15)))
        typo.draw(c, "ARC-AGI-3", NX + 8, BASE + 64, "med", 22, P.SOFT, la, tracking=0.06)
    c.restore()
    # decimal point: drawn in screen space; it becomes the origin of the next scene
    if f > 0:
        r = DOT_R * min(z, 1.0 + 0.0 * z)
        r = E.lerp(DOT_R, 5.0, E.smooth(E.seg(t, ZOOM0 + 0.05, ZOOM1 - 0.05)))
        gfx.dot(c, sx, sy, r, P.INK, f)


def events():
    ev = [dict(t=SPREAD0, kind="spread")]
    for i in range(0, NL, 9):
        ev.append(dict(t=SPREAD0 + LN["t0"][i], kind="traj", pan=(LN["y"][i] - C.CY) / C.H))
    ev += [dict(t=KNOCK0, kind="knock"), dict(t=RESOLVE, kind="bench", value="99.9"),
           dict(t=LABEL, kind="label"), dict(t=ZOOM0, kind="zoom")]
    return ev
