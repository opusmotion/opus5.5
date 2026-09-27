"""12 — FINAL. The cursor, one keystroke, the statement, the identity, and stillness."""
import numpy as np
import skia

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..timeline import bt
from . import s01_question as S01

T0, T1 = bt(11) - 0.1, C.DURATION + 1.0

KEYT = bt(11)  # 27.5 — the final keystroke
CLOSE0 = KEYT + 0.16
BRAND0, BRAND1 = bt(11, 1.4), bt(11, 2.3)

SIZE = 58.0
KEY = "dlight"
LEFT = "From answer"
RIGHT = "to action."
BASE = C.CY + 0.35 * SIZE
SPACE = typo.width("From answer to action.", KEY, SIZE) - typo.width(LEFT, KEY, SIZE) - typo.width(RIGHT, KEY, SIZE)
GAP0 = 30.0


def draw(c, t):
    if t < KEYT:
        # the cursor waits at the centre (handed over from scene 11)
        if t >= bt(10, 3.85):
            c.drawRect(skia.Rect.MakeLTRB(C.CX - S01.CUR_W / 2, C.CY - S01.CUR_H / 2, C.CX + S01.CUR_W / 2,
                                          C.CY + S01.CUR_H / 2), gfx.fill(P.INK, 1.0))
        return
    u = E.settle(t - CLOSE0, 16.0)
    gap = E.lerp(GAP0, SPACE, u)
    wl = typo.width(LEFT, KEY, SIZE)
    wr = typo.width(RIGHT, KEY, SIZE)
    total = wl + gap + wr
    xl = C.CX - total / 2
    # while the cursor is still present it sits between answer and action
    cursor_x = xl + wl + gap / 2
    typo.draw(c, LEFT, xl, BASE, KEY, SIZE, P.SOFT, 1.0)
    typo.draw(c, RIGHT, xl + wl + gap, BASE, KEY, SIZE, P.INK, 1.0)
    ch = S01.CUR_H * (1 - E.settle(t - CLOSE0, 22.0))
    if ch > 0.5:
        c.drawRect(skia.Rect.MakeLTRB(cursor_x - 1, C.CY - ch / 2, cursor_x + 1, C.CY + ch / 2), gfx.fill(P.INK, 1.0))
    ba = E.smoother(E.seg(t, BRAND0, BRAND1))
    typo.draw(c, "OpenAI", C.CX, C.H - 150, "semi", 26, P.INK, 0.9 * ba, align="center", tracking=-0.01)


def events():
    return [dict(t=KEYT, kind="final_key"), dict(t=BRAND0, kind="brand")]
