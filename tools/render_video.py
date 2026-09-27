"""Parallel deterministic render → FFmpeg.

  python -m tools.render_video --scale 0.5 --out out/preview.mp4 --audio out/audio.wav --preview
  python -m tools.render_video --scale 1 --out dist/final.mp4 --audio out/audio.wav

Frames are rendered in chunks by worker processes (each a pure function of frame index),
encoded to lossless (final) or light (preview) intermediates, concatenated in order, muxed."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from film import config as C  # noqa: E402

_R = None
_A = None


def _init(scale, mb, tmp, preview):
    global _R, _A
    from film.frame import Renderer

    _R = Renderer(scale, motion_blur=mb)
    _A = (tmp, preview)


def _chunk(job):
    k, a, b = job
    tmp, preview = _A
    w, h = _R.w, _R.h
    path = os.path.join(tmp, f"c{k:04d}.mkv")
    if preview:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        codec = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(C.FPS), "-i", "-", *codec, path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(a, b):
        p.stdin.write(_R.frame(fi).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on chunk {k}")
    return k, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=C.N_FRAMES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--no-mb", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--tmp", default="out/tmp_chunks")
    a = ap.parse_args()

    shutil.rmtree(a.tmp, ignore_errors=True)
    os.makedirs(a.tmp)
    jobs = []
    for k, s in enumerate(range(a.start, a.end, a.chunk)):
        jobs.append((k, s, min(s + a.chunk, a.end)))
    t0 = time.time()
    done = {}
    with Pool(a.workers, initializer=_init, initargs=(a.scale, not a.no_mb, a.tmp, a.preview)) as pool:
        for k, path in pool.imap_unordered(_chunk, jobs):
            done[k] = path
            print(f"  chunk {k + 1}/{len(jobs)}  {time.time() - t0:6.1f}s", flush=True)
    lst = os.path.join(a.tmp, "list.txt")
    with open(lst, "w") as f:
        for k in sorted(done):
            f.write(f"file '{os.path.abspath(done[k])}'\n")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst]
    if a.audio:
        cmd += ["-i", a.audio]
    if a.preview:
        cmd += ["-c:v", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-tune", "film", "-pix_fmt", "yuv420p",
                "-profile:v", "high", "-level:v", "4.2", "-g", "120", "-bf", "2",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv"]
    if a.audio:
        cmd += ["-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-r", str(C.FPS), "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    print(f"wrote {a.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
