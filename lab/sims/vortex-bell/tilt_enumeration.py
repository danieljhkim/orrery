"""Tilt-enriched vortex model vs the CHSH bound, plus the exhaustive argument.

Extends vortex_bell.py (same discussion: vortex-electron-vs-bell) with
Daniel's up/down-tilt degree of freedom: on top of rotation sense and axis,
the vortex tilts to adapt to the incoming current angle (up-tilt conducive at
steep angles, down-tilt at shallow ones).

Then the argument that ends the search: enumerate every behavior ANY local
mechanism can produce for four fixed settings. Each run yields four local
answers A(a1),A(a2),B(b1),B(b2) = +-1 -> 16 combinations; every one gives
S = +-2 exactly (S = A1(B1+B2) + A2(B1-B2); one bracket is 0, the other +-2).
Averages of +-2 can never exceed 2 — mechanism-independent, quantum-free.

Usage: python3 tilt_enumeration.py
"""

import numpy as np

rng = np.random.default_rng(7)
N = 300_000

axis = rng.normal(size=(N, 3)); axis /= np.linalg.norm(axis, axis=1, keepdims=True)
sense = rng.choice([-1.0, 1.0], N)
tilt_phase = rng.uniform(0, 2 * np.pi, N)
tilt_gain = rng.uniform(0, 1.5, N)


def det(a):
    return np.array([np.sin(a), 0.0, np.cos(a)])


def rule(ax, sn, ph, g, a_ang):
    friction = sn * (ax @ det(a_ang))
    tilt_help = g * np.cos(a_ang - ph)
    tilt_help *= np.where(np.cos(a_ang) < 0, +1, -1)
    return np.sign(friction + tilt_help + 1e-12)


def E(aa, ba):
    A = rule(axis, sense, tilt_phase, tilt_gain, aa)
    B = rule(-axis, -sense, tilt_phase + np.pi, tilt_gain, ba)
    return float(np.mean(A * B))


if __name__ == "__main__":
    grid = np.deg2rad(np.arange(0, 180, 7.5))
    Ec = {(i, j): E(x, y) for i, x in enumerate(grid) for j, y in enumerate(grid)}
    n = len(grid)
    best = max(abs(Ec[i1, j1] + Ec[i1, j2] + Ec[i2, j1] - Ec[i2, j2])
               for i1 in range(n) for i2 in range(n)
               for j1 in range(n) for j2 in range(n))
    print(f"tilt-enriched vortex model, best settings: |S| = {best:.3f}")

    print("\nall 16 deterministic local behaviors:")
    vals = [A1 * B1 + A1 * B2 + A2 * B1 - A2 * B2
            for A1 in (1, -1) for A2 in (1, -1)
            for B1 in (1, -1) for B2 in (1, -1)]
    print(sorted(set(vals)))
    print(f"maximum |S| over every possible local behavior: {max(map(abs, vals))}")
