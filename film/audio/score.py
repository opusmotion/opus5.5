"""The score is the event timeline. Each event kind maps to a small synthesis gesture;
tonality emerges in stages: noise+impulse → rhythm → pitch relationships → harmony → silence."""
import numpy as np

from .. import config as C
from ..timeline import all_events, bt
from . import instruments as I
from .dsp import SR, ar, bp, convolve_st, hp, hz, lp, pan2, reverb_ir, tt

DUR = C.DURATION
N = int(round(SR * DUR))
PAD = int(SR * 4)


class Bus:
    def __init__(self):
        self.x = np.zeros((2, N + PAD))

    def add(self, t, sig, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        s = pan2(sig, pan) if sig.ndim == 1 else sig
        s = s * gain
        if i < 0:
            s = s[:, -i:]
            i = 0
        m = min(s.shape[1], self.x.shape[1] - i)
        if m > 0:
            self.x[:, i:i + m] += s[:, :m]


class Score:
    def __init__(self):
        from ..scenes import s11_organize as S11

        # material that starts before the STOP is gated by it; material after the trigger is not
        self.stop, self.trig = S11.STOP, S11.TRIG
        self.b = {p: {n: Bus() for n in ("sfx", "music", "low", "send")} for p in ("pre", "post")}
        self.events = all_events()

    def _ph(self, t):
        return "pre" if t < self.stop else "post"

    def rng(self, ev, k=0):
        return np.random.default_rng([C.SEED, int(round(ev["t"] * 10000)), hash(ev["kind"]) % 100000, k])

    def fx(self, t, sig, gain, pan=0.0, verb=0.0):
        b = self.b[self._ph(t)]
        b["sfx"].add(t, sig, gain, pan)
        if verb:
            b["send"].add(t, sig, gain * verb, pan)

    def mu(self, t, sig, gain, pan=0.0, verb=0.25):
        b = self.b[self._ph(t)]
        b["music"].add(t, sig, gain, pan)
        if verb:
            b["send"].add(t, sig, gain * verb, pan)

    def lo(self, t, sig, gain):
        self.b[self._ph(t)]["low"].add(t, sig, gain)

    # ---- benchmark impact: restrained, tonal, never a trailer boom
    def impact(self, t, chord, rng, strength=1.0):
        self.lo(t, I.sub(58, 41, 0.42, 1.8), 0.42 * strength)
        self.fx(t, I.click(rng, 2500, 11000, 0.0006), 0.16 * strength, 0, 0.3)
        for k, n in enumerate(chord):
            f = hz(n)
            self.mu(t + 0.004 * k, I.bell(f, rng, tau=1.1 if f < 300 else 0.8, ratio=2.0, index=0.8), 0.09 * strength,
                    pan=(k - len(chord) / 2) * 0.18, verb=0.55)
        self.mu(t, I.tone(chord[:2], 2.4, 0.01, 0.9, rng=rng), 0.05 * strength, verb=0.6)

    def build(self):
        hand = {k[3:]: getattr(self, k) for k in dir(self) if k.startswith("on_")}
        for ev in self.events:
            h = hand.get(ev["kind"])
            if h:
                h(ev, self.rng(ev))
        self.beds()
        return self.mix()

    # ---------------------------------------------------------------- 01
    def on_key(self, ev, rng):
        w = 1.12 if ev["char"] == "\u2014" else (1.25 if ev["char"] == " " else 1.0)
        self.fx(ev["t"], I.keystroke(rng, w), 0.55, ev["pan"] * 0.5, 0.05)

    def on_impulse(self, ev, rng):
        t = ev["t"]
        self.lo(t, I.sub(66, 36, 0.62, 2.4, drive=1.6), 0.62)
        self.fx(t, I.knock(rng, 1.6), 0.16, 0, 0.2)
        self.fx(t, hp(rng.standard_normal(int(0.03 * SR)), 5000) * np.exp(-tt(0.03 * SR) / 0.004), 0.03)

    # ---------------------------------------------------------------- 02
    def on_spawn(self, ev, rng):
        self.fx(ev["t"], I.pip(rng.uniform(2800, 5600), 0.0025, 0.02, 0.2, rng), 0.035, ev["pan"], 0.15)

    def on_terminate(self, ev, rng):
        self.fx(ev["t"], I.click(rng, 900, 2600, 0.0012), 0.05, ev["pan"] * 0.8, 0.1)

    def on_verify(self, ev, rng):
        self.fx(ev["t"], I.pip(1760.0, 0.010, 0.06, 0.3, rng), 0.07, 0.0, 0.25)

    def on_select(self, ev, rng):
        self.fx(ev["t"], I.relay(rng), 0.12, 0, 0.15)
        self.lo(ev["t"], I.sub(90, 55, 0.08, 0.4), 0.22)

    # ---------------------------------------------------------------- 03
    def on_edge(self, ev, rng):
        self.fx(ev["t"], I.swell(rng, 0.45, 900, 3500, rise=False), 0.018, 0, 0.3)

    def on_edge_close(self, ev, rng):
        self.fx(ev["t"], I.click(rng, 1500, 6000, 0.0006), 0.05)

    def on_fill_col(self, ev, rng):
        for k in range(12):
            self.fx(ev["t"] + k * 0.013 + rng.uniform(0, 0.004), I.click(rng, 4000, 10000, 0.0003, 0.01), 0.022,
                    -0.5 + ev["i"] * 0.5, 0.1)

    def on_focus_move(self, ev, rng):
        self.fx(ev["t"], I.glide(1300, 1900, 0.09, amp=1.0), 0.012, 0.1, 0.2)

    def on_focus_lock(self, ev, rng):
        self.fx(ev["t"], I.relay(rng), 0.10, 0.15 * (ev["i"] - 1.5), 0.12)
        note = ["D5", "F#5", "A5", "D6"][ev["i"]]
        self.mu(ev["t"], I.pluck(hz(note), rng, 0.35, 0.3), 0.05, 0.15 * (ev["i"] - 1.5), 0.4)

    def on_connect(self, ev, rng):
        self.mu(ev["t"], I.glide(hz("F#4"), hz("A4"), 0.26, amp=1.0), 0.018, 0, 0.5)

    # ---------------------------------------------------------------- 04
    def on_code_start(self, ev, rng):
        self.fx(ev["t"], I.relay(rng, 0.8), 0.07)

    def on_gen_line(self, ev, rng):
        for k in range(int(rng.integers(2, 4))):
            self.fx(ev["t"] + k * 0.022, I.keystroke(rng, 0.7), 0.085, 0.1, 0.03)

    def on_ui_snap(self, ev, rng):
        self.fx(ev["t"], I.relay(rng), 0.13, 0, 0.1)
        self.lo(ev["t"], I.sub(80, 55, 0.06, 0.3), 0.18)

    def on_ui_frame(self, ev, rng):
        self.fx(ev["t"], I.pip(2349.0, 0.006, 0.05, 0.1, rng), 0.03, 0, 0.2)

    def on_node(self, ev, rng):
        note = ["D4", "F#4", "A4", "D5"][ev["i"]]
        self.mu(ev["t"], I.pluck(hz(note), rng, 0.4, 0.4), 0.07, -0.3 + 0.2 * ev["i"], 0.35)

    def on_edge_draw(self, ev, rng):
        self.mu(ev["t"], I.glide(hz("A4"), hz("E5"), 0.18), 0.008, 0, 0.3)

    def on_chart(self, ev, rng):
        self.fx(ev["t"], I.relay(rng, 0.9), 0.08)

    def on_curve(self, ev, rng):
        self.mu(ev["t"], I.glide(hz("D4"), hz("A4"), 0.4), 0.02, 0, 0.4)

    def on_reduce(self, ev, rng):
        t = ev["t"]
        # anticipation: a low fifth leans in, then breath before 10.0
        s = I.tone(["D2", "A2"], bt(4) - t - 0.05, attack=0.45, release=10.0, rng=rng)
        n = len(s)
        s *= np.clip((n - np.arange(n)) / (0.03 * SR), 0, 1)
        self.mu(t, s, 0.07, verb=0.1)

    # ---------------------------------------------------------------- 05
    def on_spread(self, ev, rng):
        self.fx(ev["t"], I.swell(rng, 0.7, 500, 5000, rise=False), 0.05, 0, 0.3)

    def on_traj(self, ev, rng):
        self.fx(ev["t"], I.pip(rng.uniform(3000, 6500), 0.002, 0.02, 0.3, rng), 0.03, ev["pan"], 0.2)

    def on_knock(self, ev, rng):
        d = bt(4, 2.0) - ev["t"]
        s = I.tone(["E3", "B3"], d, attack=d * 0.9, release=10, rng=rng)
        s *= np.clip((len(s) - np.arange(len(s))) / (0.01 * SR), 0, 1)
        self.mu(ev["t"], s, 0.05, verb=0.3)

    def on_bench(self, ev, rng):
        chord = {"99.9": ["E3", "B3", "E4", "B4"], "98": ["A2", "E3", "A3", "E4"],
                 "100": ["D2", "A2", "D3", "A3", "D4", "F#4"]}[ev["value"]]
        self.impact(ev["t"], chord, rng, 1.15 if ev["value"] == "100" else 1.0)

    def on_label(self, ev, rng):
        self.fx(ev["t"], I.pip(3136.0, 0.004, 0.03, 0.1, rng), 0.015, 0.2, 0.2)

    def on_zoom(self, ev, rng):
        d = bt(5) - ev["t"]
        self.fx(ev["t"], I.swell(rng, d, 300, 2500, rise=True), 0.05, 0, 0.2)
        self.fx(bt(5), I.click(rng, 1200, 5000, 0.0005), 0.06)

    # ---------------------------------------------------------------- 06
    def on_axes(self, ev, rng):
        self.fx(ev["t"], I.pip(1175.0, 0.01, 0.06, 0.2, rng), 0.03, -0.2, 0.3)
        self.fx(ev["t"] + 0.07, I.pip(1175.0, 0.01, 0.06, 0.2, rng), 0.025, 0.2, 0.3)

    def on_compass(self, ev, rng):
        f0, f1 = (hz("D4"), hz("D5")) if ev["freq"] == 0 else (hz("A4"), hz("E5"))
        self.mu(ev["t"], I.glide(f0, f1, ev["dur"]), 0.022, -0.2 if ev["freq"] == 0 else 0.3, 0.4)
        scratch = bp(rng.standard_normal(int(ev["dur"] * SR)), 3000, 9000) * 0.3
        self.fx(ev["t"], scratch * np.sin(np.linspace(0, np.pi, len(scratch))), 0.01)

    def on_point(self, ev, rng):
        self.fx(ev["t"], I.pip(2637.0, 0.006, 0.04, 0.2, rng), 0.03, 0.2, 0.2)

    def on_tangent(self, ev, rng):
        self.mu(ev["t"], I.glide(hz("E5"), hz("A4"), 0.3), 0.012, 0.3, 0.4)

    def on_curve2(self, ev, rng):
        self.mu(ev["t"], I.glide(hz("A3"), hz("E4"), 0.35), 0.014, -0.2, 0.4)

    def on_pencil(self, ev, rng):
        scale = ["D5", "E5", "F#5", "A5", "B5", "D6", "E6", "F#6", "A6", "B5", "D6"]
        self.mu(ev["t"], I.pluck(hz(scale[ev["i"]]), rng, 0.3, 0.35), 0.04, -0.6 + 0.12 * ev["i"], 0.45)

    def on_resolve(self, ev, rng):
        self.fx(ev["t"], I.swell(rng, 0.5, 600, 3000, rise=True), 0.025, 0, 0.3)

    # ---------------------------------------------------------------- 07
    def on_quantise(self, ev, rng):
        for k in range(3):
            self.fx(ev["t"] + k / 60, I.click(rng, 3000, 12000, 0.0004), 0.07, -0.3 + 0.3 * k)

    def on_walls(self, ev, rng):
        for k in range(26):
            self.fx(ev["t"] + k * 0.022 + rng.uniform(0, 0.01), I.click(rng, 2500, 9000, 0.0003, 0.01), 0.018,
                    rng.uniform(-0.8, 0.8), 0.1)

    def on_bfs(self, ev, rng):
        f = 900 * (3.2 ** ev["d"])
        self.fx(ev["t"], I.pip(f, 0.0025, 0.02, 0.3, rng), 0.012 * min(3.0, np.sqrt(ev["n"])) / 1.5, rng.uniform(-0.6, 0.6), 0.15)

    def on_deadend(self, ev, rng):
        self.fx(ev["t"], I.click(rng, 700, 2000, 0.001), 0.03, ev["pan"], 0.1)

    def on_found(self, ev, rng):
        self.fx(ev["t"], I.relay(rng, 1.1), 0.12, 0, 0.2)
        notes = ["D4", "F#4", "A4", "D5", "F#5", "A5", "D6"]
        for k, n in enumerate(notes):
            self.mu(ev["t"] + k * 0.055, I.pluck(hz(n), rng, 0.25, 0.4), 0.045, -0.6 + 0.2 * k, 0.35)

    def on_taut(self, ev, rng):
        s = I.pluck(hz("D3"), rng, 0.5, 0.9)
        self.mu(ev["t"], s, 0.12, 0, 0.3)
        self.fx(ev["t"], I.click(rng, 1500, 9000, 0.0005), 0.12)

    # ---------------------------------------------------------------- 08
    def on_line_move(self, ev, rng):
        self.fx(ev["t"], I.swell(rng, 0.3, 400, 2000, rise=False), 0.02)

    def on_tick(self, ev, rng):
        self.fx(ev["t"], I.click(rng, 2500, 9000, 0.0005, 0.015), 0.05 if ev["major"] else 0.028, ev["pan"], 0.1)

    def on_split(self, ev, rng):
        self.fx(ev["t"], I.relay(rng), 0.1, 0, 0.15)

    def on_compress(self, ev, rng):
        self.mu(ev["t"], I.glide(hz("A4"), hz("D4"), ev["dur"]), 0.03, 0.2, 0.35)

    def on_count(self, ev, rng):
        self.fx(ev["t"], I.click(rng, 3500, 11000, 0.0003, 0.01), 0.03, 0.4, 0.05)

    def on_diff(self, ev, rng):
        self.mu(ev["t"], I.pluck(hz("A5"), rng, 0.35, 0.3), 0.04, 0.2, 0.4)

    # ---------------------------------------------------------------- 09
    def on_curl(self, ev, rng):
        self.mu(ev["t"], I.tone(["D3"], 1.2, 0.5, 0.6, rng=rng), 0.04, -0.4, 0.4)
        self.mu(ev["t"] + 0.08, I.tone(["A3"], 1.2, 0.5, 0.6, rng=rng), 0.035, 0.0, 0.4)

    def on_luna_in(self, ev, rng):
        for k, n in enumerate(["F#5", "A5", "D6"]):
            self.mu(ev["t"] + k * 0.05, I.bell(hz(n), rng, 0.5, 3.0, 1.2), 0.035, 0.6, 0.5)

    def on_family(self, ev, rng):
        # Sol: slow, deep, deliberate
        d = bt(8, 3.45) - ev["t"]
        s = I.tone(["D2", "D3"], d, attack=0.6, release=3.0, rng=rng, warmth=0.2)
        s *= np.clip((len(s) - np.arange(len(s))) / (0.2 * SR), 0, 1)
        self.mu(ev["t"], lp(s, 700), 0.09, -0.45, 0.3)
        # Luna: one soft bell per orbit summit
        per = 1 / 1.6
        sync = bt(8, 3.0)
        k = -3
        while sync + k * per < bt(8, 3.45):
            tt_ = sync + k * per
            if tt_ > ev["t"]:
                self.mu(tt_, I.bell(hz("F#5"), rng, 0.4, 3.0, 1.0), 0.02, 0.55, 0.5)
            k += 1

    def on_astra_step(self, ev, rng):
        n = ["A4", "E5", "A4", "D5"][ev["k"] % 4]
        self.mu(ev["t"], I.pluck(hz(n), rng, 0.12, 0.25), 0.03, 0.0, 0.15)

    def on_sync(self, ev, rng):
        t = ev["t"]
        chord = ["D2", "A2", "D3", "F#3", "A3", "E4", "F#4", "A4", "D5"]
        self.mu(t, I.tone(chord, 3.2, 0.012, 1.2, rng=rng, warmth=0.08), 0.06, 0, 0.7)
        for k, n in enumerate(["D5", "F#5", "A5", "D6"]):
            self.mu(t + 0.003 * k, I.bell(hz(n), rng, 1.0, 2.0, 0.9), 0.05, -0.45 + 0.3 * k, 0.6)
        self.lo(t, I.sub(52, 44, 0.6, 2.0), 0.4)
        self.fx(t, I.click(rng, 2000, 10000, 0.0006), 0.1, 0, 0.3)

    def on_merge(self, ev, rng):
        self.fx(ev["t"], I.swell(rng, 0.25, 600, 4000, rise=True), 0.025)

    def on_flatten(self, ev, rng):
        self.fx(ev["t"] + 0.12, I.click(rng, 1500, 8000, 0.0005), 0.06)

    # ---------------------------------------------------------------- 10
    def on_stage(self, ev, rng):
        t, i = ev["t"], ev["i"]
        notes = ["D4", "F#4", "A4", "D5", "E5", "F#5", "A5"]
        self.mu(t, I.pluck(hz(notes[i]), rng, 0.3, 0.5), 0.07, -0.3 + 0.1 * i, 0.3)
        self.mu(t, I.bell(hz(notes[i]) * 2, rng, 0.25, 2.0, 0.6), 0.02, 0.2, 0.4)
        [lambda: self.fx(t, I.keystroke(rng, 0.9), 0.14),
         lambda: self.fx(t, I.relay(rng), 0.1),
         lambda: [self.fx(t + k * 0.03, I.keystroke(rng, 0.7), 0.07, 0.1) for k in range(3)],
         lambda: self.fx(t, I.relay(rng, 1.2), 0.12),
         lambda: self.fx(t, I.click(rng, 1200, 6000, 0.0008), 0.1),
         lambda: self.fx(t, I.swell(rng, 0.12, 2000, 8000, rise=False), 0.04),
         lambda: self.fx(t, I.relay(rng, 0.9), 0.12)][i]()
        self.lo(t, I.sub(72, 50, 0.07, 0.35), 0.28)
        for k in (1, 3):
            self.fx(t + k * C.BEAT / 8, I.click(rng, 6000, 14000, 0.0003, 0.01), 0.018, 0.3)

    def on_done(self, ev, rng):
        self.mu(ev["t"], I.bell(hz("A5"), rng, 0.4, 2.0, 0.7), 0.05, 0, 0.5)
        self.mu(ev["t"] + 0.07, I.bell(hz("D6"), rng, 0.6, 2.0, 0.7), 0.05, 0, 0.5)
        self.fx(ev["t"], I.relay(rng, 1.0), 0.1)

    # ---------------------------------------------------------------- 11
    def on_frag(self, ev, rng):
        t, ty, pan = ev["t"], ev["ty"], ev["pan"]
        g = 0.9 * float(np.clip(1.3 - ev["z"] / 2200, 0.35, 1.0))
        scale = ["D5", "E5", "F#5", "A5", "B5", "D6"]
        if ty in ("traj", "points"):
            self.fx(t, I.pip(rng.uniform(2500, 5000), 0.003, 0.02, 0.3, rng), 0.03 * g, pan, 0.2)
        elif ty in ("bars", "code"):
            self.fx(t, I.keystroke(rng, 0.7), 0.06 * g, pan, 0.05)
        elif ty in ("circle", "ring", "num"):
            self.mu(t, I.pluck(hz(scale[int(rng.integers(6))]), rng, 0.3, 0.3), 0.03 * g, pan, 0.35)
        elif ty in ("maze", "frame", "ticks"):
            self.fx(t, I.relay(rng), 0.05 * g, pan, 0.1)
        else:
            self.mu(t, I.glide(hz("A4"), hz("E5"), 0.12), 0.012 * g, pan, 0.3)

    def on_trigger(self, ev, rng):
        t = ev["t"]
        self.lo(t, I.sub(58, 40, 0.4, 1.6), 0.5)
        self.fx(t, I.knock(rng, 1.2), 0.2, 0, 0.3)
        self.fx(t, I.relay(rng, 0.8), 0.08)

    def on_place(self, ev, rng):
        # higher elements land first → a descending, settling line
        seq = ["D6", "A5", "F#5", "E5", "D5", "A4", "F#4", "E4", "D4"]
        u = (ev["y"] - 150) / (C.H - 300)
        n = seq[int(np.clip(u * (len(seq) - 1), 0, len(seq) - 1))]
        self.mu(ev["t"], I.pluck(hz(n), rng, 0.22, 0.25), 0.028, 0.0, 0.4)

    def on_spine(self, ev, rng):
        t = ev["t"]
        d = bt(11) - t
        s = I.tone(["D3", "A3", "D4"], d, attack=0.25, release=10, rng=rng, warmth=0.05)
        s *= np.clip((len(s) - np.arange(len(s))) / (0.12 * SR), 0, 1)
        self.mu(t, s, 0.05, 0, 0.35)

    def on_zoomout(self, ev, rng):
        self.fx(ev["t"], I.swell(rng, ev["dur"], 800, 5000, rise=True), 0.012, 0, 0.2)

    # ---------------------------------------------------------------- 12
    def on_final_key(self, ev, rng):
        self.fx(ev["t"], I.keystroke(rng, 1.3), 0.6, 0.0, 0.1)

    def on_brand(self, ev, rng):
        d = C.DURATION - ev["t"] - 0.35
        s = I.tone(["D4", "A4", "D5"], d, attack=0.9, release=0.55, rng=rng, amps=[1.0, 0.5, 0.35], warmth=0.02)
        self.mu(ev["t"], s, 0.03, 0, 0.5)

    # ---------------------------------------------------------------- beds
    def beds(self):
        rng = np.random.default_rng(C.SEED + 500)
        # rhythm emerges: a quiet clock on eighth notes from scene 03 into scene 04
        t = bt(2, 1.0)
        k = 0
        while t < bt(3, 3.2):
            self.fx(t, I.click(rng, 1800, 5000, 0.0007, 0.02), 0.022 * (0.7 + 0.3 * (k % 2 == 0)), -0.25 if k % 2 else 0.25)
            t += C.BEAT / 2
            k += 1
        # harmonic floor from scene 05, a chord from 09, thickening towards the stop
        a, b = bt(4, 2.0), bt(10, 1.45)
        n = int((b - a) * SR)
        tt_ = np.arange(n) / SR + a
        env = np.interp(tt_, [a, a + 1.0, bt(8, 0.5), bt(9), bt(10), b - 0.2, b], [0, 0.6, 0.6, 0.9, 1.0, 1.25, 1.25])
        pad = np.zeros(n)
        for note, g, start in (("D2", 0.8, a), ("A2", 0.55, a), ("D3", 0.4, bt(8)), ("F#3", 0.3, bt(8, 3)),
                               ("A3", 0.3, bt(9)), ("E4", 0.18, bt(10)), ("B3", 0.14, bt(10, 0.6))):
            f = hz(note)
            on = np.clip((tt_ - start) / 0.6, 0, 1)
            pad += g * on * (np.sin(2 * np.pi * f * tt_) + 0.1 * np.sin(4 * np.pi * f * tt_ + 0.7))
        pad = lp(pad * env, 900)
        pad *= np.clip((n - np.arange(n)) / (0.006 * SR), 0, 1)
        self.mu(a, pad, 0.028, 0, 0.3)

    # ---------------------------------------------------------------- mix
    def mix(self):
        ir = reverb_ir(1.7, seed=C.SEED)
        stems = {}
        for p in ("pre", "post"):
            b = self.b[p]
            stems[p] = b["sfx"].x + b["music"].x + b["low"].x + convolve_st(b["send"].x, ir) * 0.55
        # the STOP: everything that was sounding disappears (6 ms ramp); only room tone remains
        n_all = stems["pre"].shape[1]
        g = np.ones(n_all)
        i0 = int(self.stop * SR)
        ramp = int(0.006 * SR)
        g[i0 - ramp:i0] = np.linspace(1, 0, ramp)
        g[i0:] = 0.0
        out = stems["pre"] * g + stems["post"]
        self.stems = stems
        room = self.room(out.shape[1])
        out = out[:, :N] + room[:, :N]
        out = hp(out, 22)
        # peak normalise to -1 dBFS (no limiter: dynamics are part of the composition)
        peak = np.max(np.abs(out))
        out = out / peak * 0.891
        # end: room fades to nothing over the last 0.35 s
        fade = np.clip((N - np.arange(N)) / (0.35 * SR), 0, 1)
        out *= fade
        return out

    def room(self, n):
        rng = np.random.default_rng(C.SEED + 900)
        w = rng.standard_normal((2, n))
        r = lp(hp(w, 50), 900, 2) * 0.0025
        r += lp(hp(w[::-1], 2000), 6000, 1) * 0.00035
        return r


def render(path=None):
    out = Score().build()
    if path:
        from scipy.io import wavfile

        pcm = np.clip(out.T, -1, 1)
        wavfile.write(path, SR, (pcm * 2147483647.0).astype(np.int32))
    return out
