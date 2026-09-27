"""11 — ORGANIZE. Fragments of everything return at once; STOP; one event, and every
element finds its place — a vertical structure — one line — the original cursor."""
import math

import numpy as np
import skia

from .. import config as C
from .. import ease as E
from .. import gfx, typo
from .. import palette as P
from ..camera import Cam3
from ..timeline import bt
from . import s01_question as S01

T0, T1 = bt(10) - 0.1, bt(11) + 0.02

BIRTH0, BIRTH1 = bt(10, 0.0), bt(10, 1.25)
STOP = bt(10, 1.45)
TRIG = bt(10, 1.6)  # 26.0 — the event that organises everything
FLAT1 = TRIG + 0.3
MOVE0, MOVE1 = TRIG + 0.12, TRIG + 0.6
CONTRACT0, CONTRACT1 = TRIG + 0.55, TRIG + 0.9
ZOOM0, ZOOM1 = bt(10, 3.2), bt(10, 3.85)

NF = 110
TYPES = ["traj", "bars", "code", "stream", "circle", "maze", "ticks", "ring", "frame", "num", "points", "wave"]


def _frags():
    rng = np.random.default_rng(C.SEED + 11)
    F = []
    for i in range(NF):
        ty = TYPES[i % len(TYPES)]
        z = rng.uniform(-500, 1600)
        sc_ = (C.FOCAL + z) / C.FOCAL
        x = rng.uniform(-1100, 1100) * sc_ * 0.95
        y = rng.uniform(-620, 620) * sc_ * 0.95
        F.append(dict(ty=ty, p=np.array([x, y, z]), size=rng.uniform(110, 260), seed=int(rng.integers(1e9)),
                      birth=BIRTH0 + (BIRTH1 - BIRTH0) * rng.random() ** 0.8, drift=rng.normal(0, 30, 3)))
    return F


FR = _frags()


def cam_z(t):
    tt = min(t, STOP)
    return -C.FOCAL + 420 * E.seg(tt, BIRTH0, STOP + 0.8) ** 1.1


def camera(t):
    return Cam3((0.0, 0.0, cam_z(t)))


def _proj(f, t):
    cam = camera(t)
    tt = min(t, STOP)
    p = f["p"] + f["drift"] * (tt - BIRTH0)
    q, z = cam.project(p)
    s = C.FOCAL / max(z, 50.0)
    return q + np.array([0, 0]) - np.array([C.CX, C.CY]) + np.array([C.CX, C.CY]), s, z


def _draw_frag(c, f, x, y, s, a, sy=1.0):
    """Fragments are drawn in local px around (x,y) with scale s, vertical squash sy."""
    rng = np.random.default_rng(f["seed"])
    sz = f["size"] * s
    ty = f["ty"]
    st = gfx.stroke(P.SOFT, a, 1.1)
    ink = gfx.stroke(P.INK, a, 1.3)
    c.save()
    c.translate(x, y)
    c.scale(1.0, max(sy, 0.001))
    if ty == "traj":
        xs = np.linspace(-sz / 2, sz / 2, 40)
        for k in range(3):
            ys = sz * 0.25 * np.sin(xs / sz * 3 + k * 1.3 + rng.uniform(0, 3)) * (k + 1) / 3
            gfx.draw_poly(c, np.stack([xs, ys], -1), st)
        gfx.dot(c, xs[-1], ys[-1], 2, P.INK, a)
    elif ty in ("bars", "code"):
        for k in range(6):
            ind = (rng.integers(0, 3) * 14) if ty == "code" else 0
            w = sz * rng.uniform(0.4, 1.0)
            c.drawRect(skia.Rect.MakeLTRB(-sz / 2 + ind, -sz * 0.3 + k * sz * 0.12, -sz / 2 + ind + w, -sz * 0.3 + k * sz * 0.12 + 3), gfx.fill(P.SOFT, a))
    elif ty == "stream":
        xs = np.linspace(-sz / 2, sz / 2, 40)
        for k in range(6):
            ys = -sz * 0.2 + k * sz * 0.08 + 4 * np.sin(xs / 20 + k)
            gfx.draw_poly(c, np.stack([xs, ys], -1), gfx.stroke(P.SOFT, a * 0.7, 1.0))
    elif ty == "circle":
        c.drawCircle(0, 0, sz * 0.35, st)
        c.drawLine(-sz / 2, 0, sz / 2, 0, gfx.stroke(P.GRAY, a, 1.0))
        gfx.dot(c, sz * 0.25, -sz * 0.25, 2.2, P.INK, a)
    elif ty == "maze":
        path = skia.Path()
        for k in range(10):
            x0 = (rng.integers(0, 6) - 3) * sz / 6
            y0 = (rng.integers(0, 4) - 2) * sz / 6
            if rng.random() < 0.5:
                path.moveTo(x0, y0)
                path.lineTo(x0 + sz / 6, y0)
            else:
                path.moveTo(x0, y0)
                path.lineTo(x0, y0 + sz / 6)
        c.drawPath(path, gfx.stroke(P.GRAY, a, 1.2))
    elif ty == "ticks":
        c.drawLine(-sz / 2, 0, sz / 2, 0, ink)
        for k in range(16):
            xx = -sz / 2 + k * sz / 15
            c.drawLine(xx, 3, xx, 3 + (6 if k % 5 == 0 else 3), gfx.stroke(P.GRAY, a, 1.0))
    elif ty == "ring":
        c.drawCircle(0, 0, sz * 0.3, st)
        for k in range(24):
            an = 2 * math.pi * k / 24
            c.drawLine(math.cos(an) * sz * 0.22, math.sin(an) * sz * 0.22, math.cos(an) * sz * 0.27, math.sin(an) * sz * 0.27, gfx.stroke(P.SOFT, a, 1.0))
    elif ty == "frame":
        c.drawRect(skia.Rect.MakeLTRB(-sz / 2, -sz * 0.32, sz / 2, sz * 0.32), st)
        c.drawLine(-sz / 2, -sz * 0.2, sz / 2, -sz * 0.2, gfx.stroke(P.GRAY, a, 1.0))
    elif ty == "num":
        txt = ["99.9", "98", "100", "72.6", "40"][f["seed"] % 5]
        sizep = sz * 0.5
        p = typo.path(txt, "dlight", sizep, 0, sizep * typo.CAP / 2, align="center")
        c.drawPath(p, gfx.stroke(P.SOFT, a, 1.0))
    elif ty == "points":
        for k in range(12):
            gfx.dot(c, rng.uniform(-sz / 2, sz / 2), rng.uniform(-sz / 4, sz / 4), 1.8, P.INK, a)
    elif ty == "wave":
        xs = np.linspace(-sz / 2, sz / 2, 60)
        ys = sz * 0.18 * np.sin(xs / sz * 12) * np.exp(-((xs / sz) ** 2) * 4)
        gfx.draw_poly(c, np.stack([xs, ys], -1), ink)
    c.restore()


ORDER = sorted(range(NF), key=lambda i: FR[i]["p"][1] / ((C.FOCAL + FR[i]["p"][2]) / C.FOCAL))
SLOT = {i: k for k, i in enumerate(ORDER)}
SLOT_Y = np.linspace(150, C.H - 150, NF)


def zoom(t):
    """Scale of the final vertical line relative to the cursor it becomes."""
    u = E.discover(E.seg(t, ZOOM0, ZOOM1))
    return E.expi(C.H * 1.12 / S01.CUR_H, 1.0, u)


def draw(c, t):
    tt = min(t, STOP)
    fl_all = E.seg(t, TRIG, FLAT1)
    if t < CONTRACT1 + 0.05:
        items = []
        for i, f in enumerate(FR):
            if tt < f["birth"]:
                continue
            q, s, z = _proj(f, tt)
            if z < 80:
                continue
            a = E.smooth(E.seg(tt, f["birth"], f["birth"] + 0.12))
            fog = float(np.clip(1.25 - (z - 1100) / 2300, 0.25, 1.0))
            items.append((z, i, q, s, a * fog))
        items.sort(key=lambda r: -r[0])
        for z, i, q, s, a in items:
            f = FR[i]
            k = SLOT[i]
            d = abs(k - NF / 2) / NF * 0.12
            fl = E.smooth(E.seg(t, TRIG + d, FLAT1 + d))
            mv = E.settle(t - MOVE0 - d, 13.0)
            ct = E.reveal(E.seg(t, CONTRACT0 + d * 0.5, CONTRACT1))
            w = min(f["size"] * s, 560.0)
            x = E.lerp(q[0], C.CX, mv)
            y = E.lerp(q[1], SLOT_Y[k], mv)
            w = E.lerp(w, min(f["size"], 420.0) * 0.8, mv) * (1 - ct)
            aa = E.lerp(a, 0.9, mv)
            if fl < 1:
                _draw_frag(c, f, q[0], q[1], s, a * (1 - fl), 1 - fl)
            if fl > 0:
                gfx.line(c, x - w / 2, y, x + w / 2 + 0.01, y, gfx.stroke(P.mix(P.SOFT, P.INK, mv), aa * fl, 1.2))
    # the spine: the organised stack becomes one line, which becomes the cursor
    sp = E.smooth(E.seg(t, CONTRACT0 + 0.1, CONTRACT1))
    if sp > 0:
        z = zoom(t)
        h = S01.CUR_H * z
        grow = E.reveal(E.seg(t, CONTRACT0, CONTRACT1 + 0.05))
        hh = E.lerp(SLOT_Y[-1] - SLOT_Y[0], h, grow)
        w = E.lerp(1.6, S01.CUR_W, E.seg(t, ZOOM0, ZOOM1))
        c.drawRect(skia.Rect.MakeLTRB(C.CX - w / 2, C.CY - hh / 2, C.CX + w / 2, C.CY + hh / 2), gfx.fill(P.INK, sp))
        # residual ticks — the organised elements, shrinking into the cursor
        ra = (1 - E.smooth(E.seg(t, ZOOM0 + 0.1, ZOOM1 - 0.15))) * E.smooth(E.seg(t, CONTRACT1 - 0.1, CONTRACT1 + 0.05))
        if ra > 0:
            path = skia.Path()
            for k in range(NF):
                yk = C.CY + (SLOT_Y[k] - C.CY) * (hh / (SLOT_Y[-1] - SLOT_Y[0]))
                ww = 5.0 * min(1.0, z / 6)
                path.moveTo(C.CX - ww, yk)
                path.lineTo(C.CX + ww, yk)
            c.drawPath(path, gfx.stroke(P.SOFT, 0.8 * ra, 1.0))


def events():
    ev = []
    for f in sorted(FR, key=lambda f: f["birth"])[::2]:
        ev.append(dict(t=f["birth"], kind="frag", ty=f["ty"], pan=float(np.clip(f["p"][0] / 1400, -1, 1)),
                       z=float(f["p"][2])))
    ev += [dict(t=STOP, kind="stop"), dict(t=TRIG, kind="trigger")]
    for k in range(0, NF, 4):
        i = ORDER[k]
        d = abs(k - NF / 2) / NF * 0.12
        ev.append(dict(t=MOVE0 + d + 0.12, kind="place", y=float(SLOT_Y[k]), pan=0.0))
    ev += [dict(t=CONTRACT1, kind="spine"), dict(t=ZOOM0, kind="zoomout", dur=ZOOM1 - ZOOM0)]
    return ev
