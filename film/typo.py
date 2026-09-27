"""Typography: HarfBuzz shaping (kerning, tabular figures) → Skia glyph runs / outlines."""
import os

import numpy as np
import skia
import uharfbuzz as hb

from . import gfx

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
FILES = {
    "dthin": "InterDisplay-Thin.ttf",
    "dxlight": "InterDisplay-ExtraLight.ttf",
    "dlight": "InterDisplay-Light.ttf",
    "dreg": "InterDisplay-Regular.ttf",
    "dmed": "InterDisplay-Medium.ttf",
    "light": "Inter-Light.ttf",
    "reg": "Inter-Regular.ttf",
    "med": "Inter-Medium.ttf",
    "semi": "Inter-SemiBold.ttf",
    "ital": "Inter-Italic.ttf",
}
_faces = {}
_fonts = {}
_shape_cache = {}


class _Face:
    def __init__(self, key):
        path = os.path.join(_DIR, FILES[key])
        self.tf = skia.Typeface.MakeFromFile(path)
        blob = hb.Blob.from_file_path(path)
        self.hbface = hb.Face(blob)
        self.hbfont = hb.Font(self.hbface)
        self.upem = self.hbface.upem


def face(key):
    if key not in _faces:
        _faces[key] = _Face(key)
    return _faces[key]


def skfont(key, size):
    k = (key, round(size, 3))
    f = _fonts.get(k)
    if f is None:
        f = skia.Font(face(key).tf, size)
        f.setSubpixel(True)
        f.setEdging(skia.Font.Edging.kAntiAlias)
        f.setHinting(skia.FontHinting.kNone)
        f.setLinearMetrics(True)
        _fonts[k] = f
    return f


def shape(text, key, size, tracking=0.0, tnum=False):
    """Returns (glyph ids, x offsets, total advance, chars) for a single line."""
    ck = (text, key, round(size, 3), round(tracking, 4), tnum)
    r = _shape_cache.get(ck)
    if r is not None:
        return r
    fc = face(key)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    feats = {"kern": True, "liga": True, "calt": True}
    if tnum:
        feats["tnum"] = True
    hb.shape(fc.hbfont, buf, feats)
    s = size / fc.upem
    gids, xs, clusters = [], [], []
    x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gids.append(info.codepoint)
        xs.append(x + pos.x_offset * s)
        clusters.append(info.cluster)
        x += pos.x_advance * s + tracking * size
    adv = x - (tracking * size if gids else 0.0)
    r = (gids, np.array(xs), adv, clusters)
    _shape_cache[ck] = r
    return r


def width(text, key, size, tracking=0.0, tnum=False):
    return shape(text, key, size, tracking, tnum)[2]


def _origin(text, key, size, x, align, tracking, tnum):
    w = width(text, key, size, tracking, tnum)
    if align == "center":
        return x - w / 2
    if align == "right":
        return x - w
    return x


def blob(text, key, size, x, y, tracking=0.0, align="left", tnum=False):
    gids, xs, _, _ = shape(text, key, size, tracking, tnum)
    if not gids:
        return None
    x0 = _origin(text, key, size, x, align, tracking, tnum)
    pos = [skia.Point(float(x0 + xi), float(y)) for xi in xs]
    b = skia.TextBlobBuilder()
    b.allocRunPos(skfont(key, size), gids, pos)
    return b.make()


def draw(canvas, text, x, y, key, size, color, alpha=1.0, tracking=0.0, align="left", tnum=False):
    if alpha <= 0.001 or not text:
        return
    b = blob(text, key, size, x, y, tracking, align, tnum)
    if b is not None:
        canvas.drawTextBlob(b, 0, 0, gfx.fill(color, alpha))


def glyphs(text, key, size, x, y, tracking=0.0, align="left", tnum=False):
    """Per-glyph outlines: list of (char, skia.Path, x_left, advance)."""
    gids, xs, adv, clusters = shape(text, key, size, tracking, tnum)
    x0 = _origin(text, key, size, x, align, tracking, tnum)
    f = skfont(key, size)
    widths = f.getWidths(gids)
    out = []
    for i, g in enumerate(gids):
        p = f.getPath(g)
        if p is None:
            p = skia.Path()
        p.offset(float(x0 + xs[i]), float(y))
        ch = text[clusters[i]] if clusters[i] < len(text) else ""
        out.append((ch, p, x0 + xs[i], widths[i]))
    return out


def path(text, key, size, x, y, tracking=0.0, align="left", tnum=False):
    p = skia.Path()
    for _, gp, _, _ in glyphs(text, key, size, x, y, tracking, align, tnum):
        p.addPath(gp)
    return p


def contours(p, step=2.0):
    """Flatten a path into closed polylines (Nx2 arrays), roughly `step` px apart."""
    pm = skia.PathMeasure(p, False)
    out = []
    while True:
        L = pm.getLength()
        if L > 0:
            n = max(8, int(L / step))
            pts = []
            for i in range(n):
                pos, _ = pm.getPosTan(L * i / n)
                pts.append((pos.x(), pos.y()))
            out.append(np.array(pts))
        if not pm.nextContour():
            break
    return out


# typical metrics (Inter, per em)
CAP = 0.727
XH = 0.546
ASC = 0.969
DESC = 0.242
