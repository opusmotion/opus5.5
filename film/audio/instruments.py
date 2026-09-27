"""Procedural instruments. Each call takes its own seeded RNG → no two hits identical."""
import numpy as np

from .dsp import SR, ar, bp, expdec, hp, hz, lp, tt


def _norm(x, peak=1.0):
    m = np.max(np.abs(x)) + 1e-9
    return x / m * peak


def keystroke(rng, weight=1.0):
    """Close mechanical key: contact click, keycap modes, body, bottom-out, release."""
    n = int(0.24 * SR)
    t = tt(n)
    x = np.zeros(n)
    v = rng.uniform(0.93, 1.07)
    # contact click
    w = rng.standard_normal(n)
    click = bp(w, 1900 * v, 8200, 2) * expdec(n, 0.0007 * rng.uniform(0.8, 1.3))
    x += 0.9 * _norm(click)
    # keycap / stem modes
    for f, tau, a in ((1550, 0.006, 0.35), (2750, 0.004, 0.3), (4300, 0.0025, 0.22)):
        ff = f * v * rng.uniform(0.95, 1.05) / (weight ** 0.3)
        x += a * np.sin(2 * np.pi * ff * t + rng.uniform(0, 6.28)) * expdec(n, tau * rng.uniform(0.8, 1.25))
    # bottom-out: plate + case body
    d = int(rng.uniform(0.0035, 0.0065) * SR)
    bo = np.zeros(n)
    m = n - d
    tb = tt(m)
    body = (0.55 * np.sin(2 * np.pi * rng.uniform(175, 235) / weight ** 0.5 * tb) * np.exp(-tb / (0.020 * weight))
            + 0.35 * np.sin(2 * np.pi * rng.uniform(390, 520) / weight ** 0.3 * tb) * np.exp(-tb / 0.010)
            + 0.45 * np.sin(2 * np.pi * rng.uniform(88, 118) * tb) * np.exp(-tb / (0.013 * weight)))
    thock = bp(rng.standard_normal(m), 600, 3200, 2) * np.exp(-tb / 0.0018)
    bo[d:] = body * weight + 0.6 * _norm(thock)
    x += 0.85 * bo
    # release (upstroke), softer
    r = int(rng.uniform(0.070, 0.115) * SR)
    mr = n - r
    tr = tt(mr)
    rel = hp(rng.standard_normal(mr), 2600, 2) * np.exp(-tr / 0.0006)
    rel += 0.3 * np.sin(2 * np.pi * rng.uniform(2300, 3100) * tr) * np.exp(-tr / 0.003)
    x[r:] += 0.28 * _norm(rel) * rng.uniform(0.7, 1.1)
    x = lp(x, 11000)
    # tiny desk reflection
    k = int(rng.uniform(0.0011, 0.0019) * SR)
    x[k:] += 0.22 * x[:-k]
    return _norm(x) * rng.uniform(0.86, 1.0)


def click(rng, lo=1800, hi=8000, tau=0.0008, dur=0.03):
    n = int(dur * SR)
    return _norm(bp(rng.standard_normal(n), lo, hi) * expdec(n, tau))


def relay(rng, bright=1.0):
    """Two dry metallic contacts a few ms apart."""
    n = int(0.05 * SR)
    x = np.zeros(n)
    for i, (off, a) in enumerate(((0.0, 1.0), (rng.uniform(0.004, 0.008), 0.6))):
        o = int(off * SR)
        m = n - o
        tm = tt(m)
        c = bp(rng.standard_normal(m), 2200 * bright, 9000) * np.exp(-tm / 0.0005)
        c += 0.35 * np.sin(2 * np.pi * rng.uniform(4800, 6200) * bright * tm) * np.exp(-tm / 0.0022)
        c += 0.25 * np.sin(2 * np.pi * rng.uniform(900, 1300) * tm) * np.exp(-tm / 0.004)
        x[o:] += a * _norm(c)
    return _norm(x)


def pip(freq, tau=0.006, dur=0.06, click_amt=0.15, rng=None):
    n = int(dur * SR)
    t = tt(n)
    x = np.sin(2 * np.pi * freq * t) * ar(n, 0.0004, tau)
    if click_amt and rng is not None:
        x += click_amt * bp(rng.standard_normal(n), 3000, 9000) * expdec(n, 0.0004)
    return x


def pluck(freq, rng, tau=0.45, bright=0.5, dur=None):
    """Modal pluck: inharmonic-ish partials with faster decay up the series."""
    dur = dur or tau * 4
    n = int(dur * SR)
    t = tt(n)
    x = np.zeros(n)
    for k, a in enumerate((1.0, 0.45 * bright + 0.1, 0.25 * bright, 0.12 * bright, 0.06 * bright)):
        f = freq * (k + 1) * (1 + 0.0009 * k * k)
        if f > SR * 0.45:
            break
        x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * np.exp(-t / (tau / (1 + 0.9 * k)))
    x *= ar(n, 0.0015, 10.0)
    x += 0.08 * bright * bp(rng.standard_normal(n), 2000, 8000) * expdec(n, 0.0008)
    return x


def bell(freq, rng, tau=0.9, ratio=2.76, index=1.6, dur=None):
    dur = dur or tau * 4
    n = int(dur * SR)
    t = tt(n)
    ienv = index * np.exp(-t / (tau * 0.35))
    x = np.sin(2 * np.pi * freq * t + ienv * np.sin(2 * np.pi * freq * ratio * t)) * ar(n, 0.002, tau)
    return x


def sub(f0=60.0, f1=40.0, tau=0.45, dur=1.6, drive=1.3):
    n = int(dur * SR)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t / 0.09)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * ar(n, 0.004, tau)
    x = np.tanh(drive * x) / np.tanh(drive)
    return lp(x, 220)


def tone(freqs, dur, attack=0.02, release=0.8, amps=None, rng=None, detune=0.0015, warmth=0.12):
    n = int(dur * SR)
    t = tt(n)
    x = np.zeros(n)
    amps = amps or [1.0] * len(freqs)
    for f, a in zip(freqs, amps):
        f = hz(f)
        for d in (-detune, detune):
            ph = rng.uniform(0, 6.28) if rng is not None else 0.0
            x += 0.5 * a * (np.sin(2 * np.pi * f * (1 + d) * t + ph) + warmth * np.sin(4 * np.pi * f * (1 + d) * t + ph))
    return x * ar(n, attack, release)


def glide(f0, f1, dur, rng=None, amp=1.0, tau=None):
    n = int(dur * SR)
    t = tt(n)
    u = t / dur
    f = f0 * (f1 / f0) ** u
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.sin(np.pi * np.clip(u, 0, 1)) ** 1.5 if tau is None else ar(n, 0.005, tau)
    return amp * np.sin(ph) * env


def swell(rng, dur, lo=400, hi=4000, rise=True, amp=1.0):
    n = int(dur * SR)
    t = tt(n)
    u = t / dur
    env = (u ** 2.2 if rise else (1 - u) ** 2.2) * (1 - np.exp(-(1 - u) * 40) if rise else 1)
    return amp * bp(rng.standard_normal(n), lo, hi) * env


def knock(rng, weight=1.0):
    """Wooden low knock (modal)."""
    n = int(0.25 * SR)
    t = tt(n)
    x = np.zeros(n)
    for f, tau, a in ((170, 0.05, 1.0), (410, 0.025, 0.5), (760, 0.012, 0.3), (1320, 0.006, 0.2)):
        x += a * np.sin(2 * np.pi * f * rng.uniform(0.97, 1.03) / weight ** 0.3 * t) * np.exp(-t / tau)
    x += 0.4 * bp(rng.standard_normal(n), 400, 3000) * expdec(n, 0.0015)
    return _norm(x)
