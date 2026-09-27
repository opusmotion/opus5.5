"""10 — FROM MODEL TO WORK. One set of strokes transforms causally through
prompt → browse → code → interface → science → document → done, while a plan completes."""
import math

import numpy as np
import skia

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..timeline import bt

T0, T1 = bt(9), bt(10) + 0.3

K, M = 24, 72
OX, OY = 1060.0, 540.0
FX0, FX1, FY0, FY1 = OX - 300, OX + 300, OY - 200, OY + 200
STEP = C.BEAT / 2
STAGES = ["Prompt", "Browse", "Code", "Interface", "Science", "Document", "Done"]
ST = [bt(9) + i * STEP for i in range(7)]
LX = 250.0
LY = [OY + (i - 3) * 44 for i in range(7)]
TR = 0.17


def _line(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y1]], float)


def _brace(x, y0, y1, d):
    h = y1 - y0
    ym = (y0 + y1) / 2
    s = np.linspace(0, 1, 60)
    y = y0 + h * s
    bulge = d * (0.5 - 0.5 * np.cos(np.clip(np.abs(s - 0.5) * 2, 0, 1) * math.pi))
    tip = d * 1.4 * np.exp(-((s - 0.5) / 0.03) ** 2)
    xx = x + bulge * 0.6 - tip + d * 0.8
    return np.stack([xx, y], -1)


def _wave(x0, x1, yc, amp, n=90, ph=0.0):
    xs = np.linspace(x0, x1, n)
    u = (xs - x0) / (x1 - x0)
    return np.stack([xs, yc - amp * np.exp(-2.2 * u) * np.sin(u * 5.5 * math.pi + ph)], -1)


def _stage(i):
    """Each stage: list of K (polyline, width, alpha, colour)."""
    rng = np.random.default_rng(C.SEED + 100 + i)
    S = []
    if i == 0:  # PROMPT — cursor, typed words, field rule
        S.append((_line(OX - 100, OY - 22, OX - 100, OY + 22), 2.0, 1.0, P.INK))
        S.append((_line(FX0, OY + 44, FX1, OY + 44), 1.2, 0.7, P.GRAY))
        x = FX0
        for w in (120, 70, 150):
            S.append((_line(x, OY, x + w, OY), 6.0, 0.9, P.SOFT))
            x += w + 14
    elif i == 1:  # BROWSE — viewport and scrolling content
        S.append((_line(FX0, FY0, FX0, FY1), 1.3, 0.9, P.SOFT))
        S.append((_line(FX0, FY1, FX1, FY1), 1.3, 0.9, P.SOFT))
        S.append((_line(FX1, FY1, FX1, FY0), 1.3, 0.9, P.SOFT))
        S.append((_line(FX1, FY0, FX0, FY0), 1.3, 0.9, P.SOFT))
        S.append((gfx.rect_pts(FX0 + 30, FY0 + 34, FX0 + 250, FY0 + 170), 1.2, 0.8, P.GRAY))
        for k in range(9):
            y = FY0 + 60 + k * 34
            x0 = FX0 + (280 if k < 4 else 30)
            S.append((_line(x0, y, x0 + rng.uniform(140, FX1 - 30 - x0), y), 6.0, 0.85, P.SOFT))
    elif i == 2:  # CODE — braces and indentation
        S.append((_brace(FX0 + 20, FY0, FY1, -26), 1.6, 1.0, P.INK))
        S.append((_line(FX0 + 90, FY0 + 20, FX0 + 90, FY1 - 20), 1.0, 0.6, P.GRAY))
        S.append((_brace(FX1 - 20, FY0, FY1, 26), 1.6, 1.0, P.INK))
        S.append((_line(FX0 + 126, FY0 + 90, FX0 + 126, FY1 - 90), 1.0, 0.6, P.GRAY))
        ind = [0, 1, 1, 2, 2, 2, 1, 2, 1, 0]
        for k in range(10):
            y = FY0 + 40 + k * 36
            x0 = FX0 + 60 + ind[k] * 36
            S.append((_line(x0, y, x0 + rng.uniform(80, 300), y), 6.0, 0.9, P.SOFT))
    elif i == 3:  # INTERFACE — frame, sidebar, cards
        S.append((_line(FX0 + 150, FY0, FX0 + 150, FY1), 1.2, 0.9, P.SOFT))
        S.append((gfx.rect_pts(FX0, FY0, FX1, FY1), 1.3, 0.9, P.SOFT))
        S.append((_line(FX0, FY0 + 50, FX1, FY0 + 50), 1.1, 0.8, P.GRAY))
        S.append((gfx.rect_pts(FX1 - 120, FY1 - 50, FX1 - 20, FY1 - 18), 1.2, 1.0, P.INK))
        for (x0, y0, x1, y1) in ((FX0 + 175, FY0 + 75, FX0 + 375, FY0 + 205), (FX0 + 395, FY0 + 75, FX1 - 25, FY0 + 205),
                                 (FX0 + 175, FY0 + 225, FX0 + 375, FY1 - 70), (FX0 + 395, FY0 + 225, FX1 - 25, FY1 - 70)):
            S.append((gfx.rect_pts(x0, y0, x1, y1), 1.2, 0.85, P.SOFT))
        for k in range(5):
            y = FY0 + 85 + k * 34
            S.append((_line(FX0 + 22, y, FX0 + 22 + (90 if k % 2 == 0 else 64), y), 5.0, 0.8, P.SOFT))
        for k in range(4):
            S.append((_line(FX0 + 195, FY0 + 105 + 26 * (k % 2) + 150 * (k // 2), FX0 + 330, FY0 + 105 + 26 * (k % 2) + 150 * (k // 2)), 4.0, 0.7, P.GRAY))
    elif i == 4:  # SCIENCE — axes and a measured signal
        S.append((_line(FX0 + 40, FY0, FX0 + 40, FY1 - 30), 1.3, 0.9, P.SOFT))
        S.append((_line(FX0 + 40, FY1 - 30, FX1, FY1 - 30), 1.3, 0.9, P.SOFT))
        S.append((_wave(FX0 + 60, FX1 - 10, OY - 10, 150), 1.7, 1.0, P.INK))
        S.append((_wave(FX0 + 60, FX1 - 10, OY - 10, 150, ph=0.35), 1.0, 0.45, P.GRAY))
        xs = np.linspace(FX0 + 60, FX1 - 10, 90)
        u = (xs - xs[0]) / (xs[-1] - xs[0])
        env = np.stack([xs, OY - 10 - 150 * np.exp(-2.2 * u)], -1)
        S.append((env, 1.0, 0.5, P.GRAY))
        for k in range(9):
            x = FX0 + 40 + (k + 1) * 60
            S.append((_line(x, FY1 - 30, x, FY1 - 22), 1.0, 0.8, P.GRAY))
        for k in range(6):
            u = (k + 0.5) / 6
            x = FX0 + 60 + u * ((FX1 - 10) - (FX0 + 60))
            y = OY - 10 - 150 * math.exp(-2.2 * u) * math.sin(u * 5.5 * math.pi)
            S.append((gfx.rect_pts(x - 3.5, y - 3.5, x + 3.5, y + 3.5), 1.2, 1.0, P.INK))
    elif i == 5:  # DOCUMENT — the plot becomes a figure on a page
        px0, px1, py0, py1 = OX - 180, OX + 180, OY - 230, OY + 230
        S.append((_line(px0, py0, px0, py1), 1.3, 0.9, P.SOFT))
        S.append((_line(px0, py1, px1, py1), 1.3, 0.9, P.SOFT))
        S.append((_wave(px0 + 40, px1 - 40, py0 + 95, 45), 1.4, 1.0, P.INK))
        S.append((_line(px1, py1, px1, py0), 1.3, 0.9, P.SOFT))
        S.append((_line(px1, py0, px0, py0), 1.3, 0.9, P.SOFT))
        S.append((gfx.rect_pts(px0 + 30, py0 + 40, px1 - 30, py0 + 150), 1.0, 0.6, P.GRAY))
        S.append((_line(px0 + 40, py0 + 24, px0 + 190, py0 + 24), 6.0, 1.0, P.INK))
        for k in range(11):
            y = py0 + 190 + k * 22
            last = k in (4, 10)
            S.append((_line(px0 + 40, y, px0 + 40 + (160 if last else 280), y), 4.0, 0.8, P.SOFT))
    else:  # DONE — everything returns into the checkbox
        bx, by = LX + 5.5, LY[6]
        for k in range(K):
            S.append((_line(bx - 1, by, bx + 1, by), 1.0, 0.0, P.INK))
    while len(S) < K:
        last = S[-1][0]
        p = last[-1]
        S.append((np.array([p, p]), 1.0, 0.0, P.SOFT))
    return [(gfx.resample(pl, M), w, a, col) for (pl, w, a, col) in S[:K]]


STAGE = [_stage(i) for i in range(7)]


def strokes(t):
    out = []
    for k in range(K):
        cur = STAGE[0][k]
        pts, w, a, col = cur[0].copy(), cur[1], cur[2], cur[3]
        for s in range(1, 7):
            d = 0.004 * k if s < 6 else 0.006 * k
            u = E.reveal(E.seg(t, ST[s] - 0.06 + d, ST[s] - 0.06 + TR + d))
            if u <= 0:
                break
            nxt = STAGE[s][k]
            pts = pts + (nxt[0] - pts) * u
            w = E.lerp(w, nxt[1], u)
            a = E.lerp(a, nxt[2], u)
            col = P.mix(col, nxt[3], u)
        out.append((pts, w, a, col))
    return out


def draw(c, t):
    fin = E.smooth(E.seg(t, bt(10, 0.0), bt(10, 0.45)))
    # prompt: words appear as typed on arrival; scrolling during browse
    scroll = 70 * E.traverse(E.seg(t, ST[1] + 0.08, ST[2] - 0.02)) * (1 - E.seg(t, ST[2] - 0.06, ST[2] + 0.1))
    for k, (pts, w, a, col) in enumerate(strokes(t)):
        if a <= 0.01:
            continue
        if 2 <= k <= 4 and t < ST[1]:
            typed = E.seg(t, bt(9) + 0.02 + 0.05 * (k - 2), bt(9) + 0.07 + 0.05 * (k - 2))
            pts = gfx.partial(pts, 0, typed)
            if len(pts) < 2:
                continue
        if ST[1] <= t < ST[2] + 0.1 and k >= 4:
            pts = pts - np.array([0, scroll])
        gfx.draw_poly(c, pts, gfx.stroke(col, a * (1 - fin), w))
    # the plan — a checklist completing in step with the work
    la = E.smooth(E.seg(t, bt(9) - 0.05, bt(9) + 0.2)) * (1 - fin)
    for i, name in enumerate(STAGES):
        done = t >= (ST[i + 1] if i < 6 else ST[6] + 0.22)
        active = ST[i] <= t and not done
        col = P.INK if active else (P.SOFT if done else P.GRAY)
        y = LY[i]
        box = skia.Rect.MakeLTRB(LX, y - 5.5, LX + 11, y + 5.5)
        c.drawRect(box, gfx.stroke(col, la, 1.2))
        if done:
            fu = E.settle(t - (ST[i + 1] if i < 6 else ST[6] + 0.22), 30.0)
            inset = 5.5 * (1 - fu)
            c.drawRect(skia.Rect.MakeLTRB(LX + inset, y - 5.5 + inset, LX + 11 - inset, y + 5.5 - inset), gfx.fill(P.INK, la))
        typo.draw(c, name, LX + 26, y + 7.5, "reg", 22, col, la)


def events():
    ev = []
    for i in range(7):
        ev.append(dict(t=ST[i], kind="stage", i=i))
    ev.append(dict(t=ST[6] + 0.22, kind="done"))
    return ev
