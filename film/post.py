"""Optical finishing: tonal black, halation/bloom, vignette, deterministic film grain."""
import numpy as np
from scipy import ndimage

from . import config as C
from . import palette as P


class Post:
    def __init__(self, scale=1.0):
        self.w = int(round(C.W * scale))
        self.h = int(round(C.H * scale))
        self.scale = scale
        w, h = self.w, self.h
        rng = np.random.default_rng(C.SEED + 101)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx - w / 2) / (w / 2)
        ny = (yy - h / 2) / (h / 2)
        r2 = nx * nx * 0.62 + ny * ny
        base = np.array(P.BG, np.float32)
        lift = 0.012 * np.exp(-r2 * 1.35)
        lo = rng.standard_normal((h // 36 + 3, w // 36 + 3)).astype(np.float32)
        lo = ndimage.gaussian_filter(lo, 1.2)
        lo = ndimage.zoom(lo, (h / (lo.shape[0] - 2), w / (lo.shape[1] - 2)), order=3)[:h, :w]
        lo = lo / (lo.std() + 1e-6) * 0.0022
        self.bg = (base[None, None, :] + (lift + lo)[..., None]).astype(np.float32)
        r = np.sqrt(r2)
        self.vignette = (1.0 - 0.20 * np.clip((r - 0.35) / 1.05, 0, 1) ** 1.6).astype(np.float32)[..., None]
        self.grain = []
        for _ in range(3):
            g = rng.standard_normal((h, w)).astype(np.float32)
            g = ndimage.gaussian_filter(g, 0.6 * max(scale, 0.5))
            g /= g.std() + 1e-6
            self.grain.append(g.astype(np.float16))

    def _down4(self, img):
        h, w = self.h, self.w
        hh, ww = h // 4, w // 4
        return img[: hh * 4, : ww * 4].reshape(hh, 4, ww, 4, 3).mean(axis=(1, 3))

    def _up4(self, small):
        up = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)
        up = ndimage.uniform_filter(up, size=(5, 5, 1), mode="nearest")
        out = np.zeros((self.h, self.w, 3), np.float32)
        out[: up.shape[0], : up.shape[1]] = up
        return out

    def apply(self, content, fi, bloom=1.0, grain=1.0):
        small = self._down4(content)
        s = self.scale
        b1 = ndimage.gaussian_filter(small, (2.2 * s, 2.2 * s, 0))
        b2 = ndimage.gaussian_filter(small, (11.0 * s, 11.0 * s, 0))
        glow = self._up4(0.30 * b1 + 0.22 * b2) * bloom
        img = 1.0 - (1.0 - self.bg) * (1.0 - content)
        img = img + glow
        img *= self.vignette
        lum = np.clip(img.mean(-1), 0, 1)
        amp = (0.0075 + 0.020 * np.sqrt(lum) * (1.0 - 0.7 * lum)) * grain
        rng = np.random.default_rng(C.SEED * 7 + fi)
        g = self.grain[fi % 3]
        dy, dx = int(rng.integers(0, self.h)), int(rng.integers(0, self.w))
        g = np.roll(g, (dy, dx), axis=(0, 1)).astype(np.float32)
        img += (g * amp)[..., None]
        return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
