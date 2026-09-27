"""Single source of truth for musical time and global render modulation.

Every scene module declares its active window (T0, T1), its key times (derived from
`bt`), a pure `draw(canvas, t)` and an `events()` list consumed by the audio engine.
`all_events()` merges them so picture and sound share one timeline."""
from .config import BAR, BEAT


def bt(bar, beat=0.0):
    """Absolute time of `beat` (may be fractional) within `bar` (0-based)."""
    return bar * BAR + beat * BEAT


SCENES = [
    ("01 question", bt(0), bt(1)),
    ("02 reason", bt(1), bt(2)),
    ("03 see", bt(2), bt(3)),
    ("04 build", bt(3), bt(4)),
    ("05 99.9", bt(4), bt(5)),
    ("06 98", bt(5), bt(6)),
    ("07 100", bt(6), bt(7)),
    ("08 time", bt(7), bt(8)),
    ("09 family", bt(8), bt(9)),
    ("10 work", bt(9), bt(10)),
    ("11 organize", bt(10), bt(11)),
    ("12 final", bt(11), bt(12)),
]

# temporal supersampling (motion blur) — samples per frame for the final render
_MB = [
    (0.00, 2.15, 1),
    (2.15, 2.60, 4),  # cursor → line
    (2.60, 10.0, 2),
    (10.0, 12.0, 2),
    (12.0, 12.6, 4),  # push into the decimal point
    (12.6, 22.1, 2),
    (22.1, 25.0, 4),  # fastest section
    (25.0, 27.5, 3),
    (27.5, 30.0, 1),
]


def mb_samples(t):
    for a, b, n in _MB:
        if a <= t < b:
            return n
    return 1


def bloom(t):
    """Global halation amount."""
    from .ease import pulse

    b = 1.0
    b += 0.9 * pulse(t, bt(0, 3.5), 0.02, 0.18)  # cursor transient
    b += 0.5 * pulse(t, bt(8, 3.0), 0.05, 0.35)  # family synchronisation
    return b


def grain(t):
    return 1.0


def all_events():
    from .scenes import ALL

    ev = []
    for s in ALL:
        if hasattr(s, "events"):
            ev.extend(s.events())
    ev.sort(key=lambda e: e["t"])
    return ev
