"""08 — TIME COMPRESSES / OSWorld 2.0. The baseline becomes 75 minutes of work; the same
work, compressed into 40 minutes, pulls itself shorter."""
import math

import numpy as np
import skia

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..timeline import bt
from . import s07_100 as S07

T0, T1 = bt(7), bt(8) + 0.4

XL = 300.0
PXM = 17.6  # px per minute → 75 min = 1320 px
SOL_Y, AST_Y = 640.0, 470.0

MOVE0 = bt(7, 0.25)
TICKS0, TICKS1 = bt(7, 0.55), bt(7, 1.15)
LAB0 = bt(7, 0.9)
SPLIT = bt(7, 1.35)
COMP0, COMP1 = bt(7, 1.75), bt(7, 2.85)
DIFF = bt(7, 2.85)
OUT0 = bt(8, 0.0)


def sol_line(t):
    u = E.settle(t - MOVE0, 12.0)
    x0 = E.lerp(S07.MX - 1, XL, u)
    x1 = E.lerp(S07.MX + S07.NXC * S07.CELL + 1, XL + 75 * PXM, u)
    y = E.lerp(S07.BASE_Y, SOL_Y, u)
    return x0, x1, y


def astra_scale(t, k=0.0):
    d = 0.0035 * k
    return E.lerp(1.0, 40.0 / 75.0, E.reveal(E.seg(t, COMP0 + d, COMP1 + d)))


def astra_y(t):
    return E.lerp(SOL_Y, AST_Y, E.settle(t - SPLIT, 14.0))


def minutes(t):
    return int(round(75 - 35 * E.reveal(E.seg(t, COMP0, COMP1))))


def draw(c, t):
    out = E.smooth(E.seg(t, OUT0, OUT0 + 0.3))
    x0, x1, y = sol_line(t)
    la = 1 - out
    # Sol — the long line (becomes the ghost)
    ghost = E.smooth(E.seg(t, SPLIT, SPLIT + 0.4))
    solc = P.mix(P.INK, P.GRAY, ghost)
    if t < OUT0:
        gfx.line(c, x0, y, x1, y, gfx.stroke(solc, 1.0 * (1 - out), 1.6))
    # minute ticks
    ticks = gfx.stroke(P.GRAY, 0.9 * la, 1.0)
    tp = skia.Path()
    for m in range(76):
        u = E.seg(t, TICKS0 + (TICKS1 - TICKS0) * m / 75, TICKS0 + (TICKS1 - TICKS0) * m / 75 + 0.05)
        if u <= 0:
            break
        h = (9 if m % 15 == 0 else 6 if m % 5 == 0 else 3) * u
        xm = x0 + m * PXM
        tp.moveTo(xm, y + 4)
        tp.lineTo(xm, y + 4 + h)
    c.drawPath(tp, ticks)
    lab = E.smooth(E.seg(t, LAB0, LAB0 + 0.25)) * la
    typo.draw(c, "GPT-5.6 Sol", XL, SOL_Y - 22, "reg", 22, P.SOFT, lab)
    typo.draw(c, "65.7%", XL + typo.width("GPT-5.6 Sol", "reg", 22) + 18, SOL_Y - 22, "reg", 22, P.GRAY, lab, tnum=True)
    typo.draw(c, "75 min", XL + 75 * PXM, SOL_Y - 22, "reg", 22, P.SOFT, lab, align="right", tnum=True)
    typo.draw(c, "OSWorld 2.0", XL, 330, "med", 22, P.SOFT, lab, tracking=0.02)
    # Astra — same work, less time
    if t >= SPLIT:
        ya = astra_y(t)
        sc = astra_scale(t)
        xe = XL + 75 * PXM * sc
        if t < OUT0:
            gfx.line(c, XL, ya, xe, ya, gfx.stroke(P.INK, 1.0 * (1 - out), 1.6))
        tp = skia.Path()
        for m in range(76):
            xm = XL + m * PXM * astra_scale(t, 75 - m)
            h = 9 if m % 15 == 0 else 6 if m % 5 == 0 else 3
            tp.moveTo(xm, ya + 4)
            tp.lineTo(xm, ya + 4 + h)
        c.drawPath(tp, gfx.stroke(P.SOFT, 0.95 * la, 1.0))
        al = E.smooth(E.seg(t, SPLIT + 0.1, SPLIT + 0.35)) * la
        typo.draw(c, "GPT-6 Astra", XL, ya - 22, "reg", 22, P.INK, al)
        typo.draw(c, "72.6%", XL + typo.width("GPT-6 Astra", "reg", 22) + 18, ya - 22, "reg", 22, P.SOFT, al, tnum=True)
        typo.draw(c, f"{minutes(t)} min", xe, ya - 22, "reg", 22, P.INK, al, align="right", tnum=True)
        # the saved time
        dd = E.smooth(E.seg(t, DIFF, DIFF + 0.25)) * la
        if dd > 0:
            xa = XL + 40 * PXM
            yy = np.arange(ya + 18, SOL_Y - 6, 7.0)
            for yv in yy:
                gfx.line(c, xa, yv, xa, yv + 3.0, gfx.stroke(P.SOFT, 0.8 * dd, 1.0))
            xb = XL + 75 * PXM
            by = SOL_Y + 34
            br = gfx.stroke(P.SOFT, 0.9 * dd, 1.0)
            gfx.line(c, xa, by, xb, by, br)
            gfx.line(c, xa, by - 5, xa, by + 5, br)
            gfx.line(c, xb, by - 5, xb, by + 5, br)
            typo.draw(c, "47% less time per task", (xa + xb) / 2, by + 34, "reg", 20, P.SOFT, dd, align="center", tnum=True)


def events():
    ev = [dict(t=MOVE0, kind="line_move")]
    for m in range(0, 76, 5):
        ev.append(dict(t=TICKS0 + (TICKS1 - TICKS0) * m / 75, kind="tick", major=m % 15 == 0,
                       pan=(XL + m * PXM - C.CX) / C.W))
    ev += [dict(t=SPLIT, kind="split"), dict(t=COMP0, kind="compress", dur=COMP1 - COMP0), dict(t=DIFF, kind="diff")]
    # counter clicks 75 → 40
    last = 75
    for f in range(int(COMP0 * C.FPS), int(COMP1 * C.FPS) + 1):
        m = minutes(f / C.FPS)
        if m != last:
            ev.append(dict(t=f / C.FPS, kind="count", m=m))
            last = m
    return ev
