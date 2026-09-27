"""Global render constants. Everything is evaluated from film time t (seconds)."""

W, H = 1920, 1080
CX, CY = W / 2.0, H / 2.0
FPS = 60
DURATION = 30.0
N_FRAMES = int(round(FPS * DURATION))  # 1800
BPM = 96.0
BEAT = 60.0 / BPM  # 0.625 s
BAR = 4 * BEAT  # 2.5 s
SEED = 20260927
SR = 48000
SHUTTER = 0.5  # fraction of the frame interval (180° shutter)
FOCAL = 1600.0  # default camera focal length in px
