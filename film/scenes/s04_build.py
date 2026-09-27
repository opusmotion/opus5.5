"""04 — BUILD. The selected paragraph becomes code → interface → diagram → chart,
then complexity drains away to a single curve (anticipation for the first number)."""
import math

import numpy as np
import skia

from .. import config as C
from .. import ease as E
from .. import gfx
from .. import palette as P
from ..timeline import bt
from . import s03_see as S03

T0, T1 = bt(3), bt(4) + 0.02

CODE0 = bt(3, 0.0)
GEN0, GEN1 = bt(3, 0.25), bt(3, 0.95)
UI0 = bt(3, 1.0)
DIA0 = bt(3, 1.9)
CHART0 = bt(3, 2.6)
REDUCE0 = bt(3, 3.2)

X0, Y0, PITCH, TH, IND = 600.0, 300.0, 34.0, 6.0, 36.0
INDENT = [0, 1, 1, 2, 2, 3, 3, 2, 1, 1, 2, 2, 1, 0]


def _code():
    rng = np.random.default_rng(C.SEED + 4)
    toks = []
    for L, lv in enumerate(INDENT):
        x = X0 + lv * IND
        n = int(rng.integers(1, 4)) if L not in (0, 13) else 2
        for k in range(n):
            w = float(rng.uniform(34, 150)) if k < n - 1 else float(rng.uniform(40, 220))
            y = Y0 + L * PITCH
            toks.append(dict(L=L, k=k, r=np.array([x, y - TH / 2, x + w, y + TH / 2])))
            x += w + 12
    return toks


TOK = _code()
SRC = S03.block_screen()

# interface layout
FR = (560.0, 250.0, 1360.0, 830.0)
SIDE_X = 760.0
HEAD_Y = 310.0
CARDS = [(790.0, 340.0, 1060.0, 560.0), (1090.0, 340.0, 1330.0, 560.0),
         (790.0, 590.0, 1060.0, 800.0), (1090.0, 590.0, 1330.0, 800.0)]
NODES = [(640.0, 540.0), (860.0, 420.0), (860.0, 660.0), (1080.0, 540.0), (1300.0, 540.0)]
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
AX0, AY0, AX1 = 600.0, 780.0, 1320.0
AYT = 300.0
DATA_X = np.linspace(660, 1280, 9)
DATA_Y = 760 - 420 * (1 - np.exp(-(DATA_X - 640) / 330)) - 30 * np.sin((DATA_X - 640) / 90)
NODE_TO_DATA = [0, 2, 4, 6, 8]


def _ui_targets():
    """Every code token gets an interface destination (nav item or card text)."""
    tg = []
    slots = []
    for ci, (x0, y0, x1, y1) in enumerate(CARDS):
        for j in range(3):
            w = (x1 - x0 - 40) * (1.0 if j < 2 else 0.55)
            yy = y0 + 34 + j * 22
            slots.append((ci, np.array([x0 + 20, yy - 3, x0 + 20 + w, yy + 3])))
    nav = [np.array([590, 360 + 40 * j - 3, 590 + (110 if j % 2 == 0 else 80), 360 + 40 * j + 3.0]) for j in range(6)]
    si = ni = 0
    for i, tk in enumerate(TOK):
        if tk["k"] == 0 and ni < len(nav):
            tg.append(("nav", ni, nav[ni]))
            ni += 1
        elif si < len(slots):
            tg.append(("card", slots[si][0], slots[si][1]))
            si += 1
        else:
            c = CARDS[i % 4]
            tg.append(("card", i % 4, np.array([c[0] + 20, c[3] - 30, c[0] + 60, c[3] - 24.0])))
    return tg


UIT = _ui_targets()


def _lerp(a, b, u):
    return a + (b - a) * u


def token_rect(i, t):
    tk = TOK[i]
    L = tk["L"]
    r = tk["r"].copy()
    # code generation: lines already in the paragraph morph, the rest are typed
    if L < len(SRC) and tk["k"] == 0:
        s = np.array(SRC[L], float)
        u = E.settle(t - CODE0 - 0.02 * L, 16.0)
        r = _lerp(s, r, u)
        g = 1.0
    else:
        tt = GEN0 + (GEN1 - GEN0) * (L / len(INDENT)) + 0.03 * tk["k"]
        g = E.seg(t, tt, tt + 0.06)
        r = np.array([r[0], r[1], r[0] + (r[2] - r[0]) * g, r[3]])
    if t >= UI0:
        kind, idx, dst = UIT[i]
        d = UI0 + 0.012 * i
        u = E.settle(t - d, 17.0)
        r = _lerp(r, dst, u)
    if t >= DIA0:
        kind, idx, dst = UIT[i]
        n = NODES[0] if kind == "nav" else NODES[idx + 1]
        u = E.reveal(E.seg(t, DIA0 + 0.01 * (i % 7), DIA0 + 0.28 + 0.01 * (i % 7)))
        tgt = np.array([n[0] - 1, n[1] - 1, n[0] + 1, n[1] + 1])
        r = _lerp(r, tgt, u)
    return r, g


def draw(c, t):
    ui = E.seg(t, UI0 - 0.05, UI0 + 0.3)
    dia = E.seg(t, DIA0, DIA0 + 0.35)
    red = E.smooth(E.seg(t, REDUCE0, REDUCE0 + 0.3))
    # indentation guides (code) → the first becomes the sidebar divider → the y axis
    ga = E.smooth(E.seg(t, GEN0 + 0.1, GEN0 + 0.4))
    if ga > 0 and t < DIA0 + 0.4:
        for lv in (1, 2, 3):
            runs, cur = [], None
            for L, v in enumerate(INDENT):
                if v >= lv:
                    cur = [L, L] if cur is None else [cur[0], L]
                elif cur is not None:
                    runs.append(cur)
                    cur = None
            if cur:
                runs.append(cur)
            for a, b in runs:
                x = X0 + (lv - 1) * IND + 6
                ya, yb = Y0 + a * PITCH - 14, Y0 + b * PITCH + 14
                if lv == 1 and a == 1:
                    u = E.settle(t - UI0, 15.0)
                    x = _lerp(x, SIDE_X, u)
                    ya, yb = _lerp(ya, FR[1], u), _lerp(yb, FR[3], u)
                    al = ga * (1 - E.smooth(E.seg(t, DIA0 + 0.05, DIA0 + 0.3)))
                else:
                    al = ga * (1 - E.smooth(ui))
                gfx.line(c, x, ya, x, ya + (yb - ya) * E.traverse(E.seg(t, GEN0, GEN0 + 0.35)),
                         gfx.stroke(P.GRAY, 0.9 * al, 1.2))
    # interface frame + header
    if t >= UI0 - 0.05 and t < DIA0 + 0.5:
        u = E.traverse(E.seg(t, UI0 - 0.05, UI0 + 0.3))
        out = E.smooth(E.seg(t, DIA0 - 0.02, DIA0 + 0.18))
        fr = gfx.rect_pts(*FR)
        gfx.draw_poly(c, gfx.partial(fr, 0, u), gfx.stroke(P.SOFT, 0.9 * (1 - out), 1.3))
        hy = gfx.partial(np.array([[FR[0], HEAD_Y], [FR[2], HEAD_Y]]), 0, u)
        gfx.draw_poly(c, hy, gfx.stroke(P.GRAY, 0.8 * (1 - out), 1.1))
        # header details: three small squares (window affordances) → pure geometry
        for j in range(3):
            a = E.smooth(E.seg(t, UI0 + 0.2 + 0.04 * j, UI0 + 0.3 + 0.04 * j)) * (1 - out)
            gfx.ring(c, FR[0] + 24 + 16 * j, (FR[1] + HEAD_Y) / 2, 3.2, P.GRAY, a, 1.1)
    # cards → nodes → data points
    if t >= UI0:
        for ci, cr in enumerate(CARDS):
            u = E.settle(t - UI0 - 0.1 - 0.05 * ci, 16.0)
            ctr = np.array([(cr[0] + cr[2]) / 2, (cr[1] + cr[3]) / 2])
            r = _lerp(np.array([ctr[0], ctr[1], ctr[0], ctr[1]]), np.array(cr), u)
            if t >= DIA0:
                n = NODES[ci + 1]
                v = E.reveal(E.seg(t, DIA0 + 0.04 * ci, DIA0 + 0.32 + 0.04 * ci))
                r = _lerp(r, np.array([n[0] - 7, n[1] - 7, n[0] + 7, n[1] + 7]), v)
            if t >= CHART0:
                di = NODE_TO_DATA[ci + 1]
                v = E.settle(t - CHART0 - 0.03 * ci, 18.0)
                p = np.array([DATA_X[di], DATA_Y[di]])
                cur = np.array([(r[0] + r[2]) / 2, (r[1] + r[3]) / 2])
                cc = _lerp(cur, p, v)
                hs = _lerp((r[2] - r[0]) / 2, 4.0, v)
                r = np.array([cc[0] - hs, cc[1] - hs, cc[0] + hs, cc[1] + hs])
            a = 1 - red
            c.drawRect(skia.Rect.MakeLTRB(*[float(v) for v in r]), gfx.stroke(P.SOFT, 0.95 * a, 1.3))
            if t >= DIA0 + 0.2:
                cc = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
                gfx.dot(c, cc[0], cc[1], 2.0, P.INK, E.smooth(E.seg(t, DIA0 + 0.2, DIA0 + 0.35)) * a)
    # tokens
    for i in range(len(TOK)):
        if t >= DIA0 + 0.45:
            break
        r, g = token_rect(i, t)
        if g <= 0:
            continue
        kind = UIT[i][0]
        col = P.mix(P.SOFT, P.INK if kind == "nav" and UIT[i][1] == 0 else P.SOFT, E.seg(t, UI0, UI0 + 0.3))
        a = 0.95 * (1 - E.smooth(E.seg(t, DIA0 + 0.15, DIA0 + 0.4)))
        c.drawRect(skia.Rect.MakeLTRB(*[float(v) for v in r]), gfx.fill(col if TOK[i]["L"] else P.INK, a))
    # caret while generating
    if GEN0 <= t < GEN1 + 0.1:
        L = min(len(INDENT) - 1, int((t - GEN0) / (GEN1 - GEN0) * len(INDENT)))
        xs = [TOK[j] for j in range(len(TOK)) if TOK[j]["L"] == L]
        r, _ = token_rect(TOK.index(xs[-1]), t)
        gfx.line(c, r[2] + 4, r[1] - 8, r[2] + 4, r[3] + 8, gfx.stroke(P.INK, 1.0, 2.0))
    # node 0 — where the navigation collapses
    if DIA0 + 0.1 <= t:
        k0 = E.settle(t - DIA0 - 0.1, 22.0)
        n0 = NODES[0]
        if t < CHART0:
            hs = 7 * k0
            c.drawRect(skia.Rect.MakeLTRB(n0[0] - hs, n0[1] - hs, n0[0] + hs, n0[1] + hs), gfx.stroke(P.SOFT, 0.95, 1.3))
            gfx.dot(c, n0[0], n0[1], 2.0, P.INK, k0)
        else:
            v = E.settle(t - CHART0, 18.0)
            p = np.array(n0) + (np.array([DATA_X[0], DATA_Y[0]]) - np.array(n0)) * v
            hs = 7 + (4 - 7) * v
            a = 1 - E.smooth(E.seg(t, REDUCE0, REDUCE0 + 0.3))
            c.drawRect(skia.Rect.MakeLTRB(p[0] - hs, p[1] - hs, p[0] + hs, p[1] + hs), gfx.stroke(P.SOFT, 0.95 * a, 1.3))
    # diagram edges
    if t >= DIA0 + 0.15:
        ea = 1 - E.smooth(E.seg(t, CHART0, CHART0 + 0.2))
        for k, (a, b) in enumerate(EDGES):
            u = E.traverse(E.seg(t, DIA0 + 0.15 + 0.05 * k, DIA0 + 0.4 + 0.05 * k))
            (x0, y0), (x1, y1) = NODES[a], NODES[b]
            s = np.linspace(0, 1, 40)[:, None]
            p0, p1, p2, p3 = np.array([x0 + 8, y0]), np.array([(x0 + x1) / 2, y0]), np.array([(x0 + x1) / 2, y1]), np.array([x1 - 8, y1])
            q = (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s * s * p2 + s ** 3 * p3
            gfx.draw_poly(c, gfx.partial(q, 0, u), gfx.stroke(P.SOFT, 0.9 * ea, 1.3))
    # chart: axes from the divider and header rule, extra data points, curve
    if t >= CHART0 - 0.1:
        u = E.settle(t - CHART0 + 0.1, 14.0)
        out = E.smooth(E.seg(t, REDUCE0, REDUCE0 + 0.25))
        ax = gfx.stroke(P.GRAY, 0.95 * (1 - out), 1.3)
        gfx.line(c, _lerp(SIDE_X, AX0, u), _lerp(FR[1], AYT, u), _lerp(SIDE_X, AX0, u), _lerp(FR[3], AY0, u), ax)
        gfx.line(c, _lerp(FR[0], AX0, u), _lerp(HEAD_Y, AY0, u), _lerp(FR[2], AX1, u), _lerp(HEAD_Y, AY0, u), ax)
        for k in range(1, 7):
            tk = E.smooth(E.seg(t, CHART0 + 0.1 + 0.02 * k, CHART0 + 0.2 + 0.02 * k)) * (1 - out)
            xx = AX0 + k * (AX1 - AX0) / 7
            gfx.line(c, xx, AY0, xx, AY0 + 7, gfx.stroke(P.GRAY, tk, 1.1))
        for j, (x, y) in enumerate(zip(DATA_X, DATA_Y)):
            if j in NODE_TO_DATA:
                continue
            a = E.smooth(E.seg(t, CHART0 + 0.12 + 0.03 * j, CHART0 + 0.2 + 0.03 * j)) * (1 - out)
            c.drawRect(skia.Rect.MakeLTRB(float(x - 4), float(y - 4), float(x + 4), float(y + 4)), gfx.stroke(P.SOFT, a, 1.3))
        # the curve
        cu = E.traverse(E.seg(t, CHART0 + 0.15, CHART0 + 0.5))
        curve = curve_pts(t)
        gfx.draw_poly(c, gfx.partial(curve, 0, cu), gfx.stroke(P.INK, 1.0, 1.6))


def curve_pts(t):
    """The chart curve; during the reduction it relaxes into a wide horizontal seed line."""
    base = gfx.catmull(np.stack([DATA_X, DATA_Y], -1), 12)
    base = gfx.resample(base, 240)
    u = E.reveal(E.seg(t, REDUCE0 + 0.2, bt(4)))
    xs = np.linspace(-40, C.W + 40, 240)
    seed = np.stack([xs, C.CY + 18 * np.sin(xs / 260.0) * np.exp(-((xs - C.CX) / 900) ** 2)], -1)
    return base + (seed - base) * u


def events():
    ev = [dict(t=CODE0, kind="code_start")]
    for L in range(len(SRC), len(INDENT)):
        ev.append(dict(t=GEN0 + (GEN1 - GEN0) * (L / len(INDENT)), kind="gen_line", i=L))
    ev += [dict(t=UI0, kind="ui_snap"), dict(t=UI0 + 0.3, kind="ui_frame")]
    ev += [dict(t=DIA0 + 0.04 * i, kind="node", i=i) for i in range(4)]
    ev += [dict(t=DIA0 + 0.15 + 0.05 * k, kind="edge_draw", i=k) for k in range(len(EDGES))]
    ev += [dict(t=CHART0, kind="chart"), dict(t=CHART0 + 0.15, kind="curve")]
    ev += [dict(t=REDUCE0, kind="reduce")]
    return ev
