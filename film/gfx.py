"""Skia drawing helpers and polyline utilities."""
import math

import numpy as np
import skia

_CAPS = {"butt": skia.Paint.kButt_Cap, "round": skia.Paint.kRound_Cap, "square": skia.Paint.kSquare_Cap}
_JOINS = {"miter": skia.Paint.kMiter_Join, "round": skia.Paint.kRound_Join, "bevel": skia.Paint.kBevel_Join}


def c4(c, a=1.0):
    return skia.Color4f(float(c[0]), float(c[1]), float(c[2]), float(max(0.0, min(1.0, a))))


def stroke(c, a=1.0, w=1.0, cap="butt", join="miter"):
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=float(w), Color4f=c4(c, a))
    p.setStrokeCap(_CAPS[cap])
    p.setStrokeJoin(_JOINS[join])
    return p


def fill(c, a=1.0):
    return skia.Paint(AntiAlias=True, Style=skia.Paint.kFill_Style, Color4f=c4(c, a))


def poly(pts, closed=False):
    path = skia.Path()
    pts = np.asarray(pts, float)
    if len(pts) >= 2:
        path.addPoly([skia.Point(float(x), float(y)) for x, y in pts], closed)
    return path


def draw_poly(canvas, pts, paint, closed=False):
    if len(pts) >= 2:
        canvas.drawPath(poly(pts, closed), paint)


def line(canvas, x0, y0, x1, y1, paint):
    canvas.drawLine(float(x0), float(y0), float(x1), float(y1), paint)


def dot(canvas, x, y, r, c, a=1.0):
    if a > 0.002 and r > 0:
        canvas.drawCircle(float(x), float(y), float(r), fill(c, a))


def ring(canvas, x, y, r, c, a=1.0, w=1.0):
    if a > 0.002 and r > 0:
        canvas.drawCircle(float(x), float(y), float(r), stroke(c, a, w))


def arc(canvas, x, y, r, a0, sweep, paint):
    """Angles in radians, 0 = +x, positive = clockwise (screen)."""
    if abs(sweep) < 1e-5 or r <= 0:
        return
    rect = skia.Rect.MakeLTRB(x - r, y - r, x + r, y + r)
    canvas.drawArc(rect, math.degrees(a0), math.degrees(sweep), False, paint)


def cumlen(pts):
    d = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(-1))
    return np.concatenate([[0.0], np.cumsum(d)])


def resample(pts, n, closed=False):
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    s = cumlen(pts)
    L = s[-1]
    if L <= 0:
        return np.repeat(pts[:1], n, axis=0)
    u = np.linspace(0, L, n, endpoint=not closed)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], -1)


def partial(pts, a, b):
    """Sub-polyline between arclength fractions a..b."""
    pts = np.asarray(pts, float)
    a, b = max(0.0, a), min(1.0, b)
    if b <= a or len(pts) < 2:
        return pts[:0]
    s = cumlen(pts)
    L = s[-1]
    if L <= 0:
        return pts[:0]
    sa, sb = a * L, b * L
    inner = pts[(s > sa) & (s < sb)]
    pa = np.array([np.interp(sa, s, pts[:, 0]), np.interp(sa, s, pts[:, 1])])
    pb = np.array([np.interp(sb, s, pts[:, 0]), np.interp(sb, s, pts[:, 1])])
    return np.vstack([pa, inner, pb])


def point_at(pts, u):
    pts = np.asarray(pts, float)
    s = cumlen(pts)
    L = s[-1]
    su = min(max(u, 0.0), 1.0) * L
    p = np.array([np.interp(su, s, pts[:, 0]), np.interp(su, s, pts[:, 1])])
    i = int(np.clip(np.searchsorted(s, su) - 1, 0, len(pts) - 2))
    d = pts[i + 1] - pts[i]
    n = np.linalg.norm(d)
    return p, (d / n if n > 0 else np.array([1.0, 0.0]))


def catmull(pts, n_per=16, closed=False):
    """Centripetal-ish Catmull-Rom (uniform) through control points."""
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P[-1:], P, P[:2]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    t = np.linspace(0, 1, n_per, endpoint=False)[:, None]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        a = 2 * p1
        b = p2 - p0
        c = 2 * p0 - 5 * p1 + 4 * p2 - p3
        d = -p0 + 3 * p1 - 3 * p2 + p3
        out.append(0.5 * (a + b * t + c * t * t + d * t * t * t))
    out.append(P[-2][None])
    return np.vstack(out)


def rect_pts(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float)
