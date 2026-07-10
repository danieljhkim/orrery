"""__TITLE__

What this models, what question it answers, and where the discussion lives
(almanac note path) go here — the docstring is the sim's abstract.

Usage: uv run lab/sims/__SLUG__/main.py
"""

import numpy as np

from orrery import rng

r = rng(42)
N = 1_000_000

# --- replace below with the actual model -----------------------------------
pts = r.uniform(-1, 1, size=(N, 2))
inside = (pts**2).sum(axis=1) < 1
pi_est = 4 * inside.mean()

print(f"N = {N:,}")
print(f"pi estimate = {pi_est:.5f}  (error {abs(pi_est - np.pi):.2e})")
