"""06 — 98 / FrontierMath Tier 4. The decimal point becomes an origin; strokes become
axes; compass-and-straightedge constructions grow dense, then resolve into clarity."""
import math

import numpy as np
import skia

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..timeline import bt

T0, T1 = bt(4, 3.6), bt(6) + 0.05

O = np.array([C.CX, C.CY])
R0 = 200.0
ANG = math.radians(-35)
PP = O + R0 * np.array([math.cos(ANG), math.sin(ANG)])
NRM = (O - PP) / np.linalg.norm(O - PP)  # from P towards O
TAN = np.array([-NRM[1], NRM[0]])
# tangent meets the x-axis
_s = (O[1] - PP[1]) / TAN[1]
TT = PP + TAN * _s
R1 = float(np.linalg.norm(PP - TT))
PP2 = np.array([PP[0], 2 * O[1] - PP[1]])

AX0 = bt(4, 3.7)
C0_0, C0_1 = bt(5, 0.05), bt(5, 0.75)
RAD = bt(5, 0.6)
TANG = bt(5, 0.8)
C1_0 = bt(5, 1.1)
PARA = bt(5, 1.2)
DENSE0 = bt(5, 1.55)
RES0 = bt(5, 2.4)
NUM0 = bt(5, 2.55)
FILL = bt(5, 2.95)
LABEL = bt(5, 3.15)

SIZE = 440.0
KEY = "dlight"
NUM = "98"
_numw = typo.width(NUM, KEY, SIZE)
PCT_SIZE = SIZE * 0.34
GAP = 22.0
_pctw = typo.width("%", KEY, PCT_SIZE)
NX = C.CX - (_numw + GAP + _pctw) / 2
BASE = C.CY + typo.CAP * SIZE / 2
PX = NX + _numw + GAP
PCT_BASE = BASE - typo.CAP * SIZE + typo.CAP * PCT_SIZE
NUM_PATH = typo.path(NUM, KEY, SIZE, NX, BASE)
NUM_CONT = typo.contours(NUM_PATH, 3.0)

# pencil of circles tangent to the tangent line at P
PENCIL = [40, 75, 120, 290, 430, 640, -55, -110, -190, -330, -520]


def _circle(c, ctr, r, u, a0, paint):
    if u <= 0:
        return
    gfx.arc(c, ctr[0], ctr[1], abs(r), a0, 2 * math.pi * min(u, 1.0), paint)


def draw(c, t):
    res = E.smooth(E.seg(t, RES0, RES0 + 0.35))
    keep = 1.0 - res
    # origin + axes grow out of the decimal point
    ua = E.reveal(E.seg(t, AX0, AX0 + 0.5))
    if keep > 0:
        ax = gfx.stroke(P.GRAY, 0.9 * keep, 1.2)
        L = 1100 * ua
        gfx.line(c, O[0] - L, O[1], O[0] + L, O[1], ax)
        gfx.line(c, O[0], O[1] - L * 0.6, O[0], O[1] + L * 0.6, ax)
        if t < bt(5, 0.2):
            pass
        gfx.dot(c, O[0], O[1], 5.0 - 2.5 * E.smooth(E.seg(t, bt(5, 0.0), bt(5, 0.4))), P.INK, keep)
        # ticks on axes
        tk = E.smooth(E.seg(t, bt(5, 0.3), bt(5, 0.6))) * keep
        for k in range(-7, 8):
            if k == 0:
                continue
            x = O[0] + k * 100
            gfx.line(c, x, O[1] - 4, x, O[1] + 4, gfx.stroke(P.GRAY, 0.8 * tk, 1.0))
        for k in (-4, -3, -2, -1, 1, 2, 3, 4):
            y = O[1] + k * 100
            gfx.line(c, O[0] - 4, y, O[0] + 4, y, gfx.stroke(P.GRAY, 0.8 * tk, 1.0))
    if keep <= 0 and t < NUM0:
        return
    ink = gfx.stroke(P.INK, 0.95 * keep, 1.4)
    soft = gfx.stroke(P.SOFT, 0.8 * keep, 1.1)
    # compass: circle about the origin with a visible arm
    u0 = E.traverse(E.seg(t, C0_0, C0_1))
    if u0 > 0 and keep > 0:
        _circle(c, O, R0, u0, -math.pi / 2, ink)
        if u0 < 1:
            a = -math.pi / 2 + 2 * math.pi * u0
            e = O + R0 * np.array([math.cos(a), math.sin(a)])
            gfx.line(c, O[0], O[1], e[0], e[1], gfx.stroke(P.SOFT, 0.7 * keep, 1.0))
    # radius OP, angle, labels
    ur = E.traverse(E.seg(t, RAD, RAD + 0.15))
    if ur > 0 and keep > 0:
        e = O + (PP - O) * ur
        gfx.line(c, O[0], O[1], e[0], e[1], ink)
        gfx.arc(c, O[0], O[1], 40, 0, ANG * ur, soft)
        la = E.smooth(E.seg(t, RAD + 0.1, RAD + 0.25)) * keep
        typo.draw(c, "\u03b8", O[0] + 50, O[1] - 10, "ital", 20, P.SOFT, la)
        m = (O + PP) / 2
        typo.draw(c, "r", m[0] - 18, m[1] - 8, "ital", 20, P.SOFT, la)
        gfx.dot(c, PP[0], PP[1], 3.0, P.INK, la)
    # tangent through P, right-angle marker
    ut = E.reveal(E.seg(t, TANG, TANG + 0.3))
    if ut > 0 and keep > 0:
        a = PP - TAN * 900 * ut
        b = PP + TAN * 900 * ut
        gfx.line(c, a[0], a[1], b[0], b[1], soft)
        q = 11.0
        path = skia.Path()
        p1 = PP + NRM * q
        p2 = p1 + TAN * q
        p3 = PP + TAN * q
        path.moveTo(*p1)
        path.lineTo(*p2)
        path.lineTo(*p3)
        c.drawPath(path, gfx.stroke(P.SOFT, 0.8 * keep * min(1, ut * 3), 1.0))
    # circle about T through P, intersection P'
    u1 = E.traverse(E.seg(t, C1_0, C1_0 + 0.35))
    if u1 > 0 and keep > 0:
        gfx.dot(c, TT[0], TT[1], 3.0, P.INK, keep)
        a0 = math.atan2(PP[1] - TT[1], PP[0] - TT[0])
        _circle(c, TT, R1, u1, a0, soft)
        if u1 > 0.7:
            gfx.dot(c, PP2[0], PP2[1], 3.0, P.INK, keep * E.seg(u1, 0.7, 0.8))
    # parabola
    up = E.traverse(E.seg(t, PARA, PARA + 0.35))
    if up > 0 and keep > 0:
        xs = np.linspace(-330, 330, 120)
        pts = np.stack([O[0] + xs, O[1] - xs * xs / 260.0 + 0 * xs], -1)
        gfx.draw_poly(c, gfx.partial(pts, 0.5 - up / 2, 0.5 + up / 2), gfx.stroke(P.SOFT, 0.7 * keep, 1.1))
        la = E.smooth(E.seg(t, PARA + 0.2, PARA + 0.4)) * keep
        typo.draw(c, "x\u00b2 + y\u00b2 = r\u00b2", O[0] - R0 - 150, O[1] + R0 + 40, "ital", 20, P.SOFT, la)
        typo.draw(c, "y = x\u00b2/k", O[0] + 250, O[1] - 280, "ital", 20, P.SOFT, la)
    # dense: the pencil of tangent circles + protractor ticks + intersections
    if t >= DENSE0 and keep > 0:
        for k, r in enumerate(PENCIL):
            u = E.traverse(E.seg(t, DENSE0 + 0.035 * k, DENSE0 + 0.2 + 0.035 * k))
            ctr = PP + NRM * r
            a0 = math.atan2(PP[1] - ctr[1], PP[0] - ctr[0])
            _circle(c, ctr, r, u, a0, gfx.stroke(P.SOFT if abs(r) < 300 else P.GRAY, 0.75 * keep, 1.0))
            if u >= 1:
                # intersections with the x axis
                dy = O[1] - ctr[1]
                if abs(dy) < abs(r):
                    dx = math.sqrt(r * r - dy * dy)
                    for sx in (-1, 1):
                        gfx.dot(c, ctr[0] + sx * dx, O[1], 2.2, P.INK, 0.9 * keep)
        pt = E.smooth(E.seg(t, DENSE0 + 0.1, DENSE0 + 0.5)) * keep
        for k in range(72):
            a = 2 * math.pi * k / 72
            n = 12 if k % 6 == 0 else 6
            e0 = O + (R0 + 6) * np.array([math.cos(a), math.sin(a)])
            e1 = O + (R0 + 6 + n) * np.array([math.cos(a), math.sin(a)])
            if k / 72 <= pt:
                gfx.line(c, e0[0], e0[1], e1[0], e1[1], gfx.stroke(P.GRAY, 0.8 * keep, 1.0))
    # resolution: the numerals are drawn along their outlines, then filled
    un = E.traverse(E.seg(t, NUM0, FILL))
    if un > 0:
        pa = gfx.stroke(P.INK, 1.0, 1.4)
        for cont in NUM_CONT:
            closed = np.vstack([cont, cont[:1]])
            gfx.draw_poly(c, gfx.partial(closed, 0, un), pa)
        f = E.reveal(E.seg(t, FILL, FILL + 0.15))
        if f > 0:
            c.drawPath(NUM_PATH, gfx.fill(P.INK, f))
        pa2 = E.smooth(E.seg(t, FILL + 0.05, FILL + 0.3))
        typo.draw(c, "%", PX, PCT_BASE, KEY, PCT_SIZE, P.SOFT, pa2)
        la = E.smooth(E.seg(t, LABEL, LABEL + 0.25))
        typo.draw(c, "FrontierMath Tier 4", NX + 8, BASE + 64, "med", 22, P.SOFT, la, tracking=0.02)


def events():
    ev = [dict(t=AX0, kind="axes"), dict(t=C0_0, kind="compass", dur=C0_1 - C0_0, freq=0),
          dict(t=RAD, kind="point", i=0), dict(t=TANG, kind="tangent"), dict(t=C1_0, kind="compass", dur=0.35, freq=1),
          dict(t=PARA, kind="curve2")]
    for k in range(len(PENCIL)):
        ev.append(dict(t=DENSE0 + 0.035 * k, kind="pencil", i=k))
    ev += [dict(t=RES0, kind="resolve"), dict(t=FILL, kind="bench", value="98"), dict(t=LABEL, kind="label")]
    return ev
