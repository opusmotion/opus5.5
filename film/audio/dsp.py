"""Small DSP toolkit: filters, envelopes, panning, synthetic reverb."""
import numpy as np
from scipy import signal

from .. import config as C

SR = C.SR

NOTES = {
    "D2": 73.416, "E2": 82.407, "A2": 110.0, "B2": 123.471, "D3": 146.832, "E3": 164.814, "F#3": 184.997,
    "A3": 220.0, "B3": 246.942, "C#4": 277.183, "D4": 293.665, "E4": 329.628, "F#4": 369.994, "G4": 391.995,
    "A4": 440.0, "B4": 493.883, "C#5": 554.365, "D5": 587.330, "E5": 659.255, "F#5": 739.989, "A5": 880.0,
    "B5": 987.767, "C#6": 1108.73, "D6": 1174.659, "E6": 1318.51, "F#6": 1479.98, "A6": 1760.0,
}


def hz(n):
    return NOTES[n] if isinstance(n, str) else float(n)


def tt(n):
    return np.arange(int(n)) / SR


_sos = {}


def _get(kind, f, order):
    key = (kind, tuple(np.atleast_1d(f)), order)
    if key not in _sos:
        _sos[key] = signal.butter(order, f, btype=kind, fs=SR, output="sos")
    return _sos[key]


def lp(x, f, order=2):
    return signal.sosfilt(_get("low", min(f, SR * 0.45), order), x)


def hp(x, f, order=2):
    return signal.sosfilt(_get("high", f, order), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(_get("band", [lo, min(hi, SR * 0.45)], order), x)


def expdec(n, tau):
    return np.exp(-tt(n) / max(tau, 1e-5))


def ar(n, attack, release_tau):
    t = tt(n)
    a = np.clip(t / max(attack, 1e-5), 0, 1)
    a = a * a * (3 - 2 * a)
    return a * np.exp(-np.maximum(t - attack, 0) / release_tau)


def adsr(n, attack, hold, release):
    t = tt(n)
    e = np.clip(t / max(attack, 1e-5), 0, 1)
    e = e * e * (3 - 2 * e)
    r = np.clip(1 - (t - attack - hold) / max(release, 1e-5), 0, 1)
    return e * r * r


def pan2(x, pan):
    pan = float(np.clip(pan, -1, 1))
    a = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)])


def reverb_ir(rt60=1.6, predelay=0.014, seed=1, bright=5500.0, dark=1400.0):
    rng = np.random.default_rng(seed)
    n = int(rt60 * SR * 1.2)
    t = tt(n)
    env = 10 ** (-3 * t / rt60)
    out = []
    for ch in range(2):
        w = rng.standard_normal(n)
        early = lp(w, bright) * np.exp(-t / (rt60 * 0.12))
        late = lp(w, dark) * env
        ir = 0.55 * early + late
        ir = np.concatenate([np.zeros(int(predelay * SR) + ch * 23), ir])
        out.append(ir)
    m = max(len(o) for o in out)
    ir = np.stack([np.pad(o, (0, m - len(o))) for o in out])
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return ir


def convolve_st(x, ir):
    return np.stack([signal.fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])
