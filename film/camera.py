"""Cameras. Cam2: orthographic 2D. Cam3: pinhole perspective with plane homographies,
so Skia can draw any 2D content on a plane in 3D with exact perspective."""
import math

import numpy as np
import skia

from . import config as C


class Cam2:
    def __init__(self, cx=C.CX, cy=C.CY, zoom=1.0, rot=0.0):
        self.cx, self.cy, self.zoom, self.rot = cx, cy, zoom, rot

    def matrix(self):
        m = skia.Matrix()
        m.preTranslate(C.CX, C.CY)
        m.preRotate(math.degrees(self.rot))
        m.preScale(self.zoom, self.zoom)
        m.preTranslate(-self.cx, -self.cy)
        return m

    def apply(self, pts):
        p = np.asarray(pts, float)
        c, s = math.cos(self.rot), math.sin(self.rot)
        x = (p[..., 0] - self.cx) * self.zoom
        y = (p[..., 1] - self.cy) * self.zoom
        return np.stack([C.CX + c * x - s * y, C.CY + s * x + c * y], -1)


def _basis(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    f = np.array([sy * cp, -sp, cy * cp])
    r = np.array([cy, 0.0, -sy])
    d = np.cross(f, r)
    cr, sr = math.cos(roll), math.sin(roll)
    r2 = cr * r + sr * d
    d2 = -sr * r + cr * d
    return np.stack([r2, d2, f])


class Cam3:
    """World: x right, y down, z forward (px units). A camera at (CX, CY, -FOCAL)
    looking +z maps the z=0 plane 1:1 onto the screen."""

    def __init__(self, pos, yaw=0.0, pitch=0.0, roll=0.0, f=C.FOCAL):
        self.pos = np.asarray(pos, float)
        self.R = _basis(yaw, pitch, roll)
        self.f = f

    @classmethod
    def look_at(cls, pos, target, f=C.FOCAL, roll=0.0):
        d = np.asarray(target, float) - np.asarray(pos, float)
        d /= np.linalg.norm(d)
        yaw = math.atan2(d[0], d[2])
        pitch = math.asin(-d[1])
        return cls(pos, yaw, pitch, roll, f)

    def to_cam(self, P):
        return (np.asarray(P, float) - self.pos) @ self.R.T

    def project(self, P):
        Pc = self.to_cam(P)
        z = Pc[..., 2]
        zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
        return np.stack([C.CX + self.f * Pc[..., 0] / zz, C.CY + self.f * Pc[..., 1] / zz], -1), z

    def plane_matrix(self, O, U, V):
        K = np.array([[self.f, 0, C.CX], [0, self.f, C.CY], [0, 0, 1.0]])
        M = np.stack([self.R @ np.asarray(U, float), self.R @ np.asarray(V, float),
                      self.R @ (np.asarray(O, float) - self.pos)], axis=1)
        Hm = K @ M
        Hm = Hm / Hm[2, 2]
        return skia.Matrix.MakeAll(*[float(v) for v in Hm.flatten()])


def frontal(zoom=1.0, cx=C.CX, cy=C.CY):
    """Perspective camera equivalent to a 2D zoom on the z=0 plane."""
    return Cam3((cx, cy, -C.FOCAL / zoom))
