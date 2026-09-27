"""Colour system: near-black, warm white, graphite, soft neutral gray.
Spectral colour exists only for the family synchronisation (scene 09)."""

BG = (0.038, 0.038, 0.041)
INK = (0.945, 0.930, 0.900)  # warm white
SOFT = (0.600, 0.592, 0.575)  # soft neutral gray
GRAY = (0.420, 0.415, 0.405)
GRAPHITE = (0.250, 0.247, 0.242)
DEEP = (0.150, 0.149, 0.147)
BLACK = (0.0, 0.0, 0.0)

# scene 09 only — warm (sun) → white (star) → cool (moon)
SOL = (0.960, 0.520, 0.330)
ASTRA = (0.985, 0.905, 0.720)
LUNA = (0.520, 0.730, 0.980)


def mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))
