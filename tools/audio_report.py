"""Per-scene level report: python -m tools.audio_report out/audio.wav"""
import os
import re
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from film.timeline import SCENES  # noqa: E402


def momentary(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-v", "verbose", "-i", path, "-af", "ebur128=framelog=verbose:peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    ts, ms, ss = [], [], []
    for line in r.stderr.splitlines():
        m = re.search(r"t:\s*([\d.]+)\s+TARGET.*?M:\s*(-?[\d.inf]+)\s+S:\s*(-?[\d.inf]+)", line)
        if m:
            ts.append(float(m.group(1)))
            ms.append(float(m.group(2)) if "inf" not in m.group(2) else -120.0)
            ss.append(float(m.group(3)) if "inf" not in m.group(3) else -120.0)
    summ = r.stderr[r.stderr.find("Summary"):]
    I = re.search(r"I:\s*(-?[\d.]+) LUFS", summ)
    LRA = re.search(r"LRA:\s*([\d.]+) LU", summ)
    TP = re.search(r"Peak:\s*(-?[\d.]+) dBFS", summ)
    return np.array(ts), np.array(ms), np.array(ss), I and I.group(1), LRA and LRA.group(1), TP and TP.group(1)


def main():
    path = sys.argv[1]
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / (2 ** 31 if x.dtype == np.int32 else 32768)
    ts, M, S, I, LRA, TP = momentary(path)
    print(f"integrated {I} LUFS   LRA {LRA} LU   true peak {TP} dBFS   len {len(x) / sr:.3f}s")
    for name, a, b in SCENES:
        seg = x[int(a * sr):int(b * sr)]
        pk = 20 * np.log10(np.abs(seg).max() + 1e-12)
        rms = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-12)
        m = M[(ts > a + 0.4) & (ts <= b + 0.4)]
        print(f"{name:12s} peak {pk:6.1f} dBFS  rms {rms:6.1f} dBFS  M max {m.max():6.1f}  M median {np.median(m):6.1f} LUFS")
    w = int(0.25 * sr)
    env = [20 * np.log10(np.sqrt((x[i:i + w] ** 2).mean()) + 1e-12) for i in range(0, len(x) - w, w)]
    print("rms/0.25s:", " ".join(f"{v:.0f}" for v in env))


if __name__ == "__main__":
    main()
