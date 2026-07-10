"""Corkscrew-biased vortex model vs the CHSH bound, plus the singles-flatness kill.

Daniel's proposal (discussion: vortex-electron-vs-bell): entangled electrons
retain a corkscrew/orbital motion from their encounter (shared phase), or all
electrons corkscrew clockwise by nature; the bias is engineered to peak at the
45/135-degree settings where QM most exceeds the classical ceiling.

Results: (1) both variants cap below 2 — the per-run budget is zero-sum, since
every run's four potential answers sum to +-2, so pumping E at one angle pair
drains another; (2) any angle-dependent bias shows up in SINGLE-station
statistics (P(up) varies with angle), whereas experiment shows flat 0.500
marginals at every setting — required by no-signaling. Two kills, one quantum-
free, one experimental.

Usage: python3 corkscrew_bell.py
"""

import numpy as np

rng = np.random.default_rng(11)
N = 400_000
K = 0.6  # corkscrew bias strength

axis = rng.normal(size=(N, 3)); axis /= np.linalg.norm(axis, axis=1, keepdims=True)
phase_shared = rng.uniform(0, 2 * np.pi, N)


def det(a):
    return np.array([np.sin(a), 0.0, np.cos(a)])


def rule(ax, ph, a_ang, k):
    return np.sign(ax @ det(a_ang) + k * np.cos(2 * a_ang - ph) + 1e-12)


def E(aa, ba, ph_B, k):
    return float(np.mean(rule(axis, phase_shared, aa, k) * rule(-axis, ph_B, ba, k)))


def best_S(ph_B, k):
    grid = np.deg2rad(np.arange(0, 180, 5))
    Ec = {(i, j): E(x, y, ph_B, k) for i, x in enumerate(grid) for j, y in enumerate(grid)}
    n = len(grid)
    return max(abs(Ec[i1, j1] + Ec[i1, j2] + Ec[i2, j1] - Ec[i2, j2])
               for i1 in range(n) for i2 in range(n)
               for j1 in range(n) for j2 in range(n))


if __name__ == "__main__":
    print("variant 1 (shared corkscrew phase from the encounter):")
    print(f"  best |S| = {best_S(phase_shared, K):.3f}")
    print("variant 2 (universal clockwise corkscrew, same phase for all):")
    print(f"  best |S| = {best_S(np.zeros(N), K):.3f}")

    print("\nsingle-station P(+1) by angle (experiment: 0.500 everywhere):")
    for deg in [0, 22.5, 45, 90, 135]:
        A = rule(axis, np.zeros(N), np.deg2rad(deg), K)
        print(f"  {deg:5.1f} deg: P(+1) = {np.mean(A > 0):.3f}")
