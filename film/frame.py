"""Frame renderer: t → pixels. Pure function of frame index and render settings."""
import numpy as np
import skia

from . import config as C
from . import post, timeline
from .scenes import ALL


class Renderer:
    def __init__(self, scale=1.0, motion_blur=True):
        self.scale = scale
        self.mb = motion_blur
        self.post = post.Post(scale)
        self.w, self.h = self.post.w, self.post.h
        self.buf = np.zeros((self.h, self.w, 4), np.uint8)
        self.surface = skia.Surface(self.buf)

    def content(self, t):
        c = self.surface.getCanvas()
        c.clear(skia.ColorBLACK)
        c.save()
        c.scale(self.scale, self.scale)
        for s in ALL:
            if s.T0 <= t < s.T1:
                c.save()
                s.draw(c, t)
                c.restore()
        c.restore()
        return self.buf[:, :, :3]

    def frame(self, fi):
        t = fi / C.FPS
        n = timeline.mb_samples(t) if self.mb else 1
        acc = np.zeros((self.h, self.w, 3), np.float32)
        for j in range(n):
            tj = t + ((j + 0.5) / n - 0.5) * C.SHUTTER / C.FPS if n > 1 else t
            tj = min(max(tj, 0.0), C.DURATION - 1e-6)
            acc += self.content(tj)
        acc *= 1.0 / (255.0 * n)
        return self.post.apply(acc, fi, bloom=timeline.bloom(t), grain=timeline.grain(t))
