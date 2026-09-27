"""07 — 100 / ExploitBench. Curves quantise into constraints; routes explore a bounded
space, terminate, loop; one controlled solution passes, pulls taut, and becomes a baseline."""
import math
from collections import deque

import numpy as np
import skia

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..timeline import bt
from . import s06_98 as S06

T0, T1 = bt(6) - 0.05, bt(7) + 0.3

NXC, NYC, CELL = 44, 14, 36.0
MX = (C.W - NXC * CELL) / 2
MY = (C.H - NYC * CELL) / 2
ROW = 11
BASE_Y = MY + (ROW + 0.5) * CELL

QUANT0 = bt(6, 0.0)
WALLS0, WALLS1 = bt(6, 0.1), bt(6, 0.75)
BFS0 = bt(6, 0.6)
FOUND = bt(6, 2.2)
SOL1 = bt(6, 2.65)
TAUT0, TAUT1 = bt(6, 2.75), bt(6, 3.3)
NUM0 = bt(6, 3.25)
LABEL = bt(6, 3.55)


def _maze():
    rng = np.random.default_rng(C.SEED + 7)
    # walls: right[x,y] between (x,y)-(x+1,y); down[x,y] between (x,y)-(x,y+1)
    right = np.ones((NXC, NYC), bool)
    down = np.ones((NXC, NYC), bool)
    seen = np.zeros((NXC, NYC), bool)
    stack = [(0, ROW)]
    seen[0, ROW] = True
    while stack:
        x, y = stack[-1]
        nb = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= x + dx < NXC and 0 <= y + dy < NYC and not seen[x + dx, y + dy]]
        if not nb:
            stack.pop()
            continue
        nx, ny = nb[int(rng.integers(len(nb)))]
        if nx > x:
            right[x, y] = False
        elif nx < x:
            right[nx, y] = False
        elif ny > y:
            down[x, y] = False
        else:
            down[x, ny] = False
        seen[nx, ny] = True
        stack.append((nx, ny))
    # open a few extra passages → loops exist
    for _ in range(int(NXC * NYC * 0.24)):
        x, y = int(rng.integers(0, NXC - 1)), int(rng.integers(0, NYC - 1))
        if rng.random() < 0.5:
            right[x, y] = False
        else:
            down[x, y] = False
    return right, down


RIGHT, DOWN = _maze()


def _nbrs(x, y):
    out = []
    if x < NXC - 1 and not RIGHT[x, y]:
        out.append((x + 1, y))
    if x > 0 and not RIGHT[x - 1, y]:
        out.append((x - 1, y))
    if y < NYC - 1 and not DOWN[x, y]:
        out.append((x, y + 1))
    if y > 0 and not DOWN[x, y - 1]:
        out.append((x, y - 1))
    return out


def _bfs():
    dist = -np.ones((NXC, NYC), int)
    par = {}
    dist[0, ROW] = 0
    q = deque([(0, ROW)])
    while q:
        x, y = q.popleft()
        for n in _nbrs(x, y):
            if dist[n] < 0:
                dist[n] = dist[x, y] + 1
                par[n] = (x, y)
                q.append(n)
    return dist, par


DIST, PAR = _bfs()
EXIT = (NXC - 1, ROW)
DEXIT = int(DIST[EXIT])
RATE = DEXIT / (FOUND - BFS0)
CHILDREN = {}
for ch, pa in PAR.items():
    CHILDREN.setdefault(pa, []).append(ch)
LEAVES = [cell for cell in PAR if cell not in CHILDREN and DIST[cell] <= DEXIT]
LOOPS = []  # non-tree openings between explored cells: close a cycle when both are reached
for x in range(NXC):
    for y in range(NYC):
        for n in _nbrs(x, y):
            if n > (x, y) and PAR.get(n) != (x, y) and PAR.get((x, y)) != n:
                if DIST[x, y] <= DEXIT and DIST[n] <= DEXIT:
                    LOOPS.append(((x, y), n))


def _path():
    p = [EXIT]
    while p[-1] != (0, ROW):
        p.append(PAR[p[-1]])
    return p[::-1]


SOL = _path()


def cc(cell):
    return np.array([MX + (cell[0] + 0.5) * CELL, MY + (cell[1] + 0.5) * CELL])


SOL_PTS = np.array([[MX - 1, BASE_Y]] + [cc(p) for p in SOL] + [[MX + NXC * CELL + 1, BASE_Y]])


def _walls():
    segs = []
    for x in range(NXC):
        for y in range(NYC):
            x0, y0 = MX + x * CELL, MY + y * CELL
            if x < NXC - 1 and RIGHT[x, y]:
                segs.append((x0 + CELL, y0, x0 + CELL, y0 + CELL))
            if y < NYC - 1 and DOWN[x, y]:
                segs.append((x0, y0 + CELL, x0 + CELL, y0 + CELL))
    W, Hh = NXC * CELL, NYC * CELL
    segs += [(MX, MY, MX + W, MY), (MX, MY + Hh, MX + W, MY + Hh),
             (MX, MY, MX, MY + ROW * CELL), (MX, MY + (ROW + 1) * CELL, MX, MY + Hh),
             (MX + W, MY, MX + W, MY + ROW * CELL), (MX + W, MY + (ROW + 1) * CELL, MX + W, MY + Hh)]
    return np.array(segs)


WALLS = _walls()


def _wall_dist():
    """A wall becomes visible when the search first touches a cell next to it."""
    out = []
    for (x0, y0, x1, y1) in WALLS:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        cells = []
        if abs(x0 - x1) < 1e-6:  # vertical wall between two columns
            gx = int(round((mx - MX) / CELL))
            gy = int((my - MY) // CELL)
            cells = [(gx - 1, gy), (gx, gy)]
        else:
            gx = int((mx - MX) // CELL)
            gy = int(round((my - MY) / CELL))
            cells = [(gx, gy - 1), (gx, gy)]
        ds = [DIST[c] for c in cells if 0 <= c[0] < NXC and 0 <= c[1] < NYC and DIST[c] >= 0]
        out.append(min(ds) if ds else 1e9)
    return np.array(out, float)


WALL_D = _wall_dist()

SIZE = 440.0
KEY = "dlight"
_numw = typo.width("100", KEY, SIZE)
PCT_SIZE = SIZE * 0.34
_pctw = typo.width("%", KEY, PCT_SIZE)
NX = C.CX - (_numw + 22 + _pctw) / 2
NUM_PATH = typo.path("100", KEY, SIZE, NX, BASE_Y)
PX = NX + _numw + 22
PCT_BASE = BASE_Y - typo.CAP * SIZE + typo.CAP * PCT_SIZE


def quantised_98():
    out = []
    for cont in S06.NUM_CONT:
        q = np.round((cont - [MX, MY]) / (CELL / 2)) * (CELL / 2) + [MX, MY]
        out.append(q)
    return out


Q98 = quantised_98()


def sol_points(t):
    u = E.spring(t - TAUT0, 16.0, 0.72) if t >= TAUT0 else 0.0
    s = gfx.cumlen(SOL_PTS)
    frac = s / s[-1]
    straight = np.stack([MX - 1 + frac * (NXC * CELL + 2), np.full(len(frac), BASE_Y)], -1)
    return SOL_PTS + (straight - SOL_PTS) * u


def draw(c, t):
    # the previous numerals harden onto the lattice and dissolve into walls
    if t < WALLS1:
        snap = E.seg(t, QUANT0, QUANT0 + 3.0 / C.FPS)
        qa = 1.0 - E.smooth(E.seg(t, QUANT0 + 0.1, WALLS1))
        if t < QUANT0:
            c.drawPath(S06.NUM_PATH, gfx.fill(P.INK, 1.0))
        else:
            for cont, q in zip(S06.NUM_CONT, Q98):
                pts = cont + (q - cont) * snap
                gfx.draw_poly(c, pts, gfx.stroke(P.INK, qa, 1.6), closed=True)
    if t < QUANT0:
        return
    gone = E.smooth(E.seg(t, TAUT0, TAUT0 + 0.25))
    if gone < 1:
        # the rules exist before they are known: a lattice of points
        la = E.smooth(E.seg(t, WALLS0, WALLS1)) * (1 - gone)
        pd = gfx.fill(P.GRAY, 0.55 * la)
        for gx in range(NXC + 1):
            for gy in range(NYC + 1):
                c.drawRect(skia.Rect.MakeXYWH(MX + gx * CELL - 0.9, MY + gy * CELL - 0.9, 1.8, 1.8), pd)
        # constraints appear where the search touches them
        d = (t - BFS0) * RATE if t >= BFS0 else -1.0
        grow = np.clip((d - WALL_D) / 1.6, 0, 1)
        dimw = E.smooth(E.seg(t, FOUND, FOUND + 0.35))
        path = skia.Path()
        for (x0, y0, x1, y1), a in zip(WALLS, grow):
            if a <= 0:
                continue
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            path.moveTo(mx + (x0 - mx) * a, my + (y0 - my) * a)
            path.lineTo(mx + (x1 - mx) * a, my + (y1 - my) * a)
        c.drawPath(path, gfx.stroke(P.mix(P.SOFT, P.GRAY, dimw), 0.8 * (1 - gone), 1.2, cap="square"))
    # exploration (BFS frontier)
    if t >= BFS0 and gone < 1:
        d = min((t - BFS0) * RATE, DEXIT + 0.0)
        dim = E.smooth(E.seg(t, FOUND, FOUND + 0.3))
        pe = gfx.stroke(P.mix(P.INK, P.GRAY, dim), 0.75 * (1 - gone), 1.2)
        path = skia.Path()
        heads = []
        for ch, pa in PAR.items():
            dp = DIST[pa]
            if dp >= d:
                continue
            u = min(1.0, d - dp)
            a, b = cc(pa), cc(ch)
            e = a + (b - a) * u
            path.moveTo(*a)
            path.lineTo(*e)
            if u < 1:
                heads.append(e)
        c.drawPath(path, pe)
        for e in heads:
            gfx.dot(c, e[0], e[1], 1.5, P.INK, 0.9 * (1 - dim))
        # dead ends terminate with a tick
        tick = gfx.stroke(P.SOFT, 0.9 * (1 - gone) * (1 - 0.6 * dim), 1.1)
        tp = skia.Path()
        for leaf in LEAVES:
            if DIST[leaf] <= d - 0.95:
                p, q = cc(PAR[leaf]), cc(leaf)
                v = (q - p) / CELL
                n = np.array([-v[1], v[0]]) * 5
                tp.moveTo(*(q - n))
                tp.lineTo(*(q + n))
        c.drawPath(tp, tick)
        # loops close when both sides are reached
        lp = skia.Path()
        for a_, b_ in LOOPS:
            m = max(DIST[a_], DIST[b_])
            if m <= d - 0.5:
                lp.moveTo(*cc(a_))
                lp.lineTo(*cc(b_))
        c.drawPath(lp, gfx.stroke(P.SOFT, 0.55 * (1 - gone) * (1 - dim), 1.0))
    # the solution
    if t >= FOUND:
        us = E.reveal(E.seg(t, FOUND, SOL1))
        pts = sol_points(t)
        part = gfx.partial(pts, 0, us)
        out = E.smooth(E.seg(t, bt(7, 0.2), bt(7, 0.5)))
        gfx.draw_poly(c, part, gfx.stroke(P.INK, 1.0, 1.7, join="round"))
        if us < 1:
            gfx.dot(c, part[-1][0], part[-1][1], 3.0, P.INK, 1.0)
    # 100% rises out of the baseline
    if t >= NUM0:
        u = E.reveal(E.seg(t, NUM0, NUM0 + 0.28))
        sink = E.accel(E.seg(t, bt(7, 0.05), bt(7, 0.3)))
        off = (1 - u) * SIZE * 0.8 + sink * SIZE * 0.8
        c.save()
        c.clipRect(skia.Rect.MakeLTRB(0, 0, C.W, BASE_Y - 0.8))
        c.translate(0, off)
        c.drawPath(NUM_PATH, gfx.fill(P.INK, 1.0))
        typo.draw(c, "%", PX, PCT_BASE, KEY, PCT_SIZE, P.SOFT, 1.0)
        c.restore()
        la = E.smooth(E.seg(t, LABEL, LABEL + 0.2)) * (1 - E.smooth(E.seg(t, bt(7, 0.0), bt(7, 0.2))))
        typo.draw(c, "ExploitBench", NX + 8, BASE_Y + 64, "med", 22, P.SOFT, la, tracking=0.02)


def events():
    ev = [dict(t=QUANT0, kind="quantise"), dict(t=WALLS0, kind="walls")]
    # exploration ticks per BFS depth step (thinned)
    for dd in range(1, DEXIT + 1, 2):
        n = int((DIST == dd).sum())
        ev.append(dict(t=BFS0 + dd / RATE, kind="bfs", n=n, d=dd / DEXIT))
    for leaf in LEAVES[::3]:
        ev.append(dict(t=BFS0 + DIST[leaf] / RATE, kind="deadend", pan=(cc(leaf)[0] - C.CX) / C.W))
    ev += [dict(t=FOUND, kind="found"), dict(t=TAUT0, kind="taut"), dict(t=NUM0, kind="bench", value="100"),
           dict(t=LABEL, kind="label")]
    return ev
