"""01 — THE QUESTION. Darkness, a cursor, `Can you—`, the cursor becomes a line."""
import numpy as np

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..timeline import bt

T0, T1 = 0.0, 2.72

PROMPT = "Can you\u2014"
KEY, SIZE = "reg", 38.0
BASE = C.CY + 0.35 * SIZE  # cursor centre sits exactly on CY
GAP = 4.0
CUR_W = 2.0
CUR_H = 1.14 * SIZE

CURSOR_ON = 0.46
# (time, visible characters) — irregular human cadence, hesitation before the dash
KEYS = [(0.800, 1), (0.905, 2), (0.985, 3), (1.100, 4), (1.188, 5), (1.262, 6), (1.334, 7), (1.600, 8)]
BLINK_OFF = 1.86
BLINK_ON = 2.04
IMPULSE = bt(0, 3.5)  # 2.1875 — the single deep transient
FULL = 2.44

TEXT_W = typo.width(PROMPT, KEY, SIZE)
X0 = C.CX - GAP - TEXT_W


def visible(t):
    n = 0
    for tk, k in KEYS:
        if t >= tk:
            n = k
    return n


def cursor_x(t):
    n = visible(t)
    if n == 0:
        return X0
    return X0 + typo.width(PROMPT[:n], KEY, SIZE) + GAP


def line_height(t):
    if t < BLINK_ON:
        return CUR_H
    # it slowly stops behaving like text...
    slow = CUR_H * (1.0 + 0.55 * E.accel(E.seg(t, BLINK_ON, IMPULSE + 0.02)))
    if t < IMPULSE:
        return slow
    # ...then one transient launches it
    u = E.reveal(E.seg(t, IMPULSE, FULL))
    return E.lerp(slow, C.H + 80, u)


def draw(c, t):
    # prompt text
    n = visible(t)
    if n:
        fade = 1.0 - E.smooth(E.seg(t, IMPULSE - 0.06, FULL + 0.18))
        col = P.mix(P.INK, P.SOFT, E.smooth(E.seg(t, BLINK_ON, IMPULSE + 0.1)))
        drift = -14.0 * E.discover(E.seg(t, IMPULSE, FULL + 0.3))
        typo.draw(c, PROMPT[:n], X0 + drift, BASE, KEY, SIZE, col, 0.92 * fade)
    if t >= C.BAR:
        return  # scene 02 owns the line from the downbeat
    on = (CURSOR_ON <= t < BLINK_OFF) or (t >= BLINK_ON)
    if not on:
        return
    x = cursor_x(t)
    h = line_height(t)
    w = E.lerp(CUR_W, 1.6, E.seg(t, IMPULSE, FULL))
    c.drawRect(_rect(x - w / 2, C.CY - h / 2, x + w / 2, C.CY + h / 2), gfx.fill(P.INK, 1.0))


def _rect(x0, y0, x1, y1):
    import skia

    return skia.Rect.MakeLTRB(float(x0), float(y0), float(x1), float(y1))


def events():
    ev = [dict(t=tk, kind="key", i=i, char=PROMPT[k - 1], pan=(cursor_x(tk) - C.CX) / C.W)
          for i, (tk, k) in enumerate(KEYS)]
    ev.append(dict(t=IMPULSE, kind="impulse", gain=1.0))
    return ev
