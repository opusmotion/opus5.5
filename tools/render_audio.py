"""python -m tools.render_audio out/audio.wav"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from film.audio.score import render

    path = sys.argv[1] if len(sys.argv) > 1 else "out/audio.wav"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = render(path)
    import numpy as np

    print(f"wrote {path}: {out.shape[1] / 48000:.3f}s peak {20 * np.log10(np.max(np.abs(out)) + 1e-12):.2f} dBFS")


if __name__ == "__main__":
    main()
