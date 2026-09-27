"""Render stills / contact sheets.

  python -m tools.stills --every 0.5 --scale 0.25 --cols 10 --out out/sheet.png
  python -m tools.stills --range 2.0 3.0 --step 4 --scale 0.5 --out out/strip.png
  python -m tools.stills --times 2.2,4.5 --scale 1 --outdir out/frames
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from film import config as C  # noqa: E402

_R = None


def _init(scale, mb):
    global _R
    from film.frame import Renderer

    _R = Renderer(scale, motion_blur=mb)


def _render(fi):
    return fi, _R.frame(fi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float)
    ap.add_argument("--range", nargs=2, type=float)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--times", type=str)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--out", type=str)
    ap.add_argument("--outdir", type=str)
    ap.add_argument("--mb", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.times:
        frames = [int(round(float(x) * C.FPS)) for x in a.times.split(",")]
    elif a.range:
        frames = list(range(int(a.range[0] * C.FPS), int(a.range[1] * C.FPS) + 1, a.step))
    else:
        ev = a.every or 0.5
        frames = [int(round(i * ev * C.FPS)) for i in range(int(C.DURATION / ev))]
    frames = [min(max(f, 0), C.N_FRAMES - 1) for f in frames]
    with Pool(a.workers, initializer=_init, initargs=(a.scale, a.mb)) as pool:
        res = dict(pool.map(_render, frames))
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        for fi in frames:
            Image.fromarray(res[fi]).save(os.path.join(a.outdir, f"f{fi:04d}.png"))
    if a.out:
        h, w = res[frames[0]].shape[:2]
        cols = min(a.cols, len(frames))
        rows = (len(frames) + cols - 1) // cols
        pad, lab = 4, 16
        sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad + lab) + pad), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("assets/fonts/Inter-Regular.ttf", 12)
        except OSError:
            font = None
        for k, fi in enumerate(frames):
            r, cc = divmod(k, cols)
            x = pad + cc * (w + pad)
            y = pad + r * (h + pad + lab)
            sheet.paste(Image.fromarray(res[fi]), (x, y + lab))
            d.text((x + 2, y + 1), f"{fi / C.FPS:6.2f}s  f{fi}", fill=(170, 170, 170), font=font)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        sheet.save(a.out)
        print("wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
