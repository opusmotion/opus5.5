"""03 — SEE. The chosen line becomes the edge of a workspace; the camera travels along
receding information; a focus indicator locates, selects and connects."""
import math

import numpy as np
import skia
from scipy.interpolate import PchipInterpolator

from .. import config as C
from .. import ease as E
from .. import gfx
from .. import palette as P
from ..camera import Cam3
from ..timeline import bt

T0, T1 = bt(2), bt(3) + 0.02

FW, TOP, BOT = 1460.0, -450.0, 450.0
COLS = [(90.0, 470.0), (540.0, 920.0), (990.0, 1370.0)]
LINE0, PITCH, BAR_H = -372.0, 22.0, 5.0
NLINES = 34

EDGE0, EDGE1 = bt(2, 0.0), bt(2, 0.7)
FILL0 = bt(2, 0.45)


def _layout():
    rng = np.random.default_rng(C.SEED + 3)
    bars = []  # (col, line, x0, x1)
    fig = (2, 3, 11)  # col 3 figure spans lines 3..11
    for ci, (a, b) in enumerate(COLS):
        li = 0
        while li < NLINES:
            n = int(rng.integers(4, 9))
            for k in range(n):
                if li >= NLINES:
                    break
                if ci == fig[0] and fig[1] <= li <= fig[2]:
                    li += 1
                    continue
                last = k == n - 1
                ww = (b - a) * (rng.uniform(0.3, 0.75) if last else rng.uniform(0.9, 1.0))
                bars.append((ci, li, a, a + ww))
                li += 1
            li += 1
    return bars, fig


BARS, FIG = _layout()


def bar_y(li):
    return LINE0 + li * PITCH


def _find_block():
    # the longest paragraph in column 2 (ties: nearest line 17) → becomes the code in scene 04
    rows = sorted([b for b in BARS if b[0] == 1], key=lambda b: b[1])
    paras, cur = [], []
    for r in rows:
        if cur and r[1] != cur[-1][1] + 1:
            paras.append(cur)
            cur = []
        cur.append(r)
    if cur:
        paras.append(cur)
    return max(paras, key=lambda p: (len(p), -abs(p[0][1] - 17)))


BLOCK = _find_block()
_bl = [BARS.index(b) for b in BLOCK]
BX0, BX1 = COLS[1]
BY0 = bar_y(BLOCK[0][1]) - 12
BY1 = bar_y(BLOCK[-1][1]) + 12
BCX, BCY = (BX0 + BX1) / 2, (BY0 + BY1) / 2


def _pick(col, line):
    for i, b in enumerate(BARS):
        if b[0] == col and b[1] >= line:
            return i
    return 0


_t1 = _pick(0, 6)
_t3 = _pick(1, 5)
TARGETS = [  # (time of arrival, rect x0,y0,x1,y1, bar indices selected)
    (bt(2, 1.35), (BARS[_t1][2] - 8, bar_y(BARS[_t1][1]) - 9, BARS[_t1][3] + 8, bar_y(BARS[_t1][1]) + 9), [_t1]),
    (bt(2, 1.85), (COLS[2][0] - 10, bar_y(FIG[1]) - 12, COLS[2][1] + 10, bar_y(FIG[2]) + 8), []),
    (bt(2, 2.35), (BARS[_t3][2] - 8, bar_y(BARS[_t3][1]) - 9, BARS[_t3][3] + 8, bar_y(BARS[_t3][1]) + 9), [_t3]),
    (bt(2, 2.9), (BX0 - 14, BY0, BX1 + 14, BY1), _bl),
]

# camera keyframes: (time, position, target)
_K = [
    (bt(2, 0.0), (0.0, 0.0, -1650.0), (0.0, 0.0, 0.0)),
    (bt(2, 0.75), (FW / 2, 0.0, -1740.0), (FW / 2, 0.0, 0.0)),
    (bt(2, 1.55), (-240.0, -30.0, -640.0), (520.0, 0.0, 0.0)),
    (bt(2, 2.45), (360.0, -10.0, -560.0), (1000.0, 40.0, 0.0)),
    (bt(3, 0.0), (BCX, BCY, -760.0), (BCX, BCY, 0.0)),
]
_kt = np.array([k[0] for k in _K])
_kp = PchipInterpolator(_kt, np.array([k[1] for k in _K]), axis=0)
_kg = PchipInterpolator(_kt, np.array([k[2] for k in _K]), axis=0)


def camera(t):
    tt = min(max(t, _kt[0]), _kt[-1])
    return Cam3.look_at(_kp(tt), _kg(tt))


def _plane(cam):
    return cam.plane_matrix((0, 0, 0), (1, 0, 0), (0, 1, 0))


def focus_rect(t):
    """Critically damped focus indicator following successive targets."""
    r = np.array(TARGETS[0][1], float)
    start = np.array([TARGETS[0][1][0], TARGETS[0][1][1], TARGETS[0][1][0], TARGETS[0][1][3]], float)
    cur = start
    for i, (ta, rect, _) in enumerate(TARGETS):
        move0 = ta - 0.28
        if t < move0:
            break
        k = E.settle(t - move0, 20.0)
        cur = cur + (np.array(rect, float) - cur) * k
    return cur


def selected(t):
    s = {}
    for ta, _, idx in TARGETS:
        for i in idx:
            s[i] = max(s.get(i, 0.0), E.smooth(E.seg(t, ta - 0.02, ta + 0.08)))
    return s


def block_screen(t=None):
    """Screen rects of the block bars at the scene 03 → 04 handoff."""
    cam = camera(bt(3) if t is None else t)
    out = []
    for b in BLOCK:
        y = bar_y(b[1])
        pts, _ = cam.project(np.array([[b[2], y - BAR_H / 2, 0], [b[3], y + BAR_H / 2, 0]], float))
        out.append((pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
    return out


def draw(c, t):
    cam = camera(t)
    H = _plane(cam)
    out = E.smooth(E.seg(t, bt(2, 3.25), bt(3)))  # everything but the block leaves
    # boundary: grows from the line's two ends
    ue = E.traverse(E.seg(t, EDGE0, EDGE1))
    c.save()
    c.concat(H)
    fa = 1.0 - out
    edge = gfx.stroke(P.SOFT, 0.85 * fa, 1.6)
    x_end = FW * ue
    c.drawLine(0, TOP, float(x_end), TOP, edge)
    c.drawLine(0, BOT, float(x_end), BOT, edge)
    c.drawLine(0, TOP, 0, BOT, gfx.stroke(P.mix(P.INK, P.SOFT, E.seg(t, T0, EDGE1)), fa, 1.6))
    ur = E.traverse(E.seg(t, EDGE1 - 0.12, EDGE1 + 0.1))
    if ur > 0:
        mid = (TOP + BOT) / 2
        c.drawLine(FW, TOP, FW, float(TOP + (mid - TOP) * ur), edge)
        c.drawLine(FW, BOT, FW, float(BOT - (BOT - mid) * ur), edge)
    # information: bars stream in, fog with depth
    sel = selected(t)
    for i, (ci, li, a, b) in enumerate(BARS):
        ts = FILL0 + ci * 0.09 + li * 0.008
        g = E.reveal(E.seg(t, ts, ts + 0.16))
        if g <= 0:
            continue
        y = bar_y(li)
        wc = cam.to_cam(np.array([(a + b) / 2, y, 0.0]))
        fog = float(np.clip(1.15 - (wc[2] - 500) / 2400, 0.25, 1.0))
        s = sel.get(i, 0.0)
        inblock = i in _bl
        keep = 1.0 if inblock else fa
        col = P.mix(P.GRAY, P.INK, s)
        c.drawRect(skia.Rect.MakeLTRB(a, y - BAR_H / 2, a + (b - a) * g, y + BAR_H / 2),
                   gfx.fill(col, (0.8 + 0.2 * s) * fog * keep))
    # figure in column 3
    fg = E.reveal(E.seg(t, FILL0 + 0.2, FILL0 + 0.45)) * fa
    if fg > 0:
        x0, x1 = COLS[2]
        y0, y1 = bar_y(FIG[1]) - 6, bar_y(FIG[2]) + 2
        c.drawRect(skia.Rect.MakeLTRB(x0, y0, x1, y1), gfx.stroke(P.GRAY, 0.9 * fg, 1.4))
        xs = np.linspace(x0 + 16, x1 - 16, 60)
        ys = y1 - 20 - (y1 - y0 - 40) * (0.15 + 0.7 * (1 - np.exp(-(xs - x0) / 140)) + 0.05 * np.sin(xs / 17))
        pts = np.stack([xs, ys], -1)
        pts = gfx.partial(pts, 0, E.traverse(E.seg(t, FILL0 + 0.3, FILL0 + 0.6)))
        gfx.draw_poly(c, pts, gfx.stroke(P.SOFT, 0.9 * fg, 1.6))
    # focus indicator — four corners on the plane
    fa2 = E.smooth(E.seg(t, TARGETS[0][0] - 0.3, TARGETS[0][0] - 0.15)) * (1 - E.smooth(E.seg(t, bt(2, 3.3), bt(3))))
    if fa2 > 0:
        x0, y0, x1, y1 = focus_rect(t)
        L = min(14.0, (x1 - x0) / 3, (y1 - y0) / 2)
        p = gfx.stroke(P.INK, fa2, 2.0, cap="square")
        for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)):
            path = skia.Path()
            path.moveTo(cx + sx * L, cy)
            path.lineTo(cx, cy)
            path.lineTo(cx, cy + sy * L)
            c.drawPath(path, p)
    c.restore()
    # connections — arcs lifted out of the plane between successive selections
    ca = 1.0 - E.smooth(E.seg(t, bt(2, 3.2), bt(2, 3.7)))
    for i in range(1, len(TARGETS)):
        ta = TARGETS[i][0]
        u = E.traverse(E.seg(t, ta + 0.02, ta + 0.3))
        if u <= 0 or ca <= 0:
            continue
        r0, r1 = TARGETS[i - 1][1], TARGETS[i][1]
        p0 = np.array([(r0[0] + r0[2]) / 2, (r0[1] + r0[3]) / 2, 0.0])
        p1 = np.array([(r1[0] + r1[2]) / 2, (r1[1] + r1[3]) / 2, 0.0])
        mid = (p0 + p1) / 2 + np.array([0, -40.0, -220.0])
        s = np.linspace(0, u, 48)[:, None]
        q = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * mid + s * s * p1
        pts, _ = cam.project(q)
        gfx.draw_poly(c, pts, gfx.stroke(P.SOFT, 0.85 * ca, 1.2))
        e0, _ = cam.project(p0)
        gfx.dot(c, e0[0], e0[1], 2.4, P.INK, ca)
        if u >= 1:
            e1, _ = cam.project(p1)
            gfx.dot(c, e1[0], e1[1], 2.4, P.INK, ca)


def events():
    ev = [dict(t=EDGE0, kind="edge"), dict(t=EDGE1, kind="edge_close")]
    for i, (ta, _, _) in enumerate(TARGETS):
        ev.append(dict(t=ta - 0.28, kind="focus_move", i=i))
        ev.append(dict(t=ta, kind="focus_lock", i=i))
        if i:
            ev.append(dict(t=ta + 0.02, kind="connect", i=i))
    for ci in range(3):
        ev.append(dict(t=FILL0 + ci * 0.09, kind="fill_col", i=ci))
    return ev
