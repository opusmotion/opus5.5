"""Motion vocabulary. Each curve has a semantic role (see README)."""
import math


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def seg(t, t0, t1):
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return clamp((t - t0) / (t1 - t0))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp(x)
    return x * x * x * (x * (6 * x - 15) + 10)


class Bezier:
    """CSS-style cubic-bezier timing function."""

    def __init__(self, x1, y1, x2, y2):
        self.p = (x1, y1, x2, y2)

    def _x(self, s):
        x1, _, x2, _ = self.p
        return 3 * (1 - s) ** 2 * s * x1 + 3 * (1 - s) * s * s * x2 + s ** 3

    def _dx(self, s):
        x1, _, x2, _ = self.p
        return 3 * (1 - s) ** 2 * x1 + 6 * (1 - s) * s * (x2 - x1) + 3 * s * s * (1 - x2)

    def _y(self, s):
        _, y1, _, y2 = self.p
        return 3 * (1 - s) ** 2 * s * y1 + 3 * (1 - s) * s * s * y2 + s ** 3

    def __call__(self, x):
        x = clamp(x)
        if x <= 0.0 or x >= 1.0:
            return x
        s = x
        for _ in range(8):
            fx = self._x(s) - x
            if abs(fx) < 1e-7:
                break
            d = self._dx(s)
            if abs(d) < 1e-6:
                break
            s = clamp(s - fx / d)
        if abs(self._x(s) - x) > 1e-5:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if self._x(mid) < x:
                    lo = mid
                else:
                    hi = mid
            s = 0.5 * (lo + hi)
        return self._y(s)


# COMPUTATION — precise traversal (near-linear, softened ends)
traverse = Bezier(0.3, 0.0, 0.7, 1.0)
# IMPORTANT REVEAL — fast acceleration, long controlled deceleration
reveal = Bezier(0.55, 0.0, 0.08, 1.0)
# DISCOVERY — slow reveal / camera scale change
discover = Bezier(0.45, 0.0, 0.2, 1.0)
# exits / anticipation
accel = Bezier(0.6, 0.0, 0.9, 0.6)
exit_ = Bezier(0.7, 0.0, 1.0, 1.0)


def settle(dt, omega=18.0):
    """INTERFACE — critically damped response to a step at dt=0 (seconds)."""
    if dt <= 0:
        return 0.0
    return 1.0 - (1.0 + omega * dt) * math.exp(-omega * dt)


def spring(dt, omega=20.0, zeta=0.55):
    """Rare overshoot — underdamped step response."""
    if dt <= 0:
        return 0.0
    wd = omega * math.sqrt(1 - zeta * zeta)
    return 1.0 - math.exp(-zeta * omega * dt) * (math.cos(wd * dt) + zeta * omega / wd * math.sin(wd * dt))


def expi(a, b, x):
    """Exponential interpolation (constant perceived zoom rate)."""
    return a * (b / a) ** x


def pulse(t, t0, attack, release):
    if t < t0:
        return 0.0
    if t < t0 + attack:
        return (t - t0) / attack
    return math.exp(-(t - t0 - attack) / release)
