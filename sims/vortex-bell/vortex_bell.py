"""CHSH Monte Carlo for the 'friction vortex' local hidden-variable model.

Model (Daniel, 2026-07-05, discussion: vortex-electron-vs-bell):
  - Each pair is born with a shared random unit axis lam; particle A carries
    +lam, particle B carries -lam (perfect anticorrelation at the source).
  - At its detector, each particle meets a 'current' at a locally chosen angle
    and is captured into the up/down channel by tilt/friction dynamics.
    Response rule: outcome = sign(spin_axis . current_direction) — bistable,
    contextual, deterministic (plus a stochastic variant).
  - Locality: each outcome depends ONLY on (own axis, own detector angle).

Result: optimized over all detector settings, |S| saturates at exactly 2
(the Bell/CHSH bound for any such rule). QM predicts 2*sqrt(2) ~ 2.828;
loophole-free experiments measure ~2.4-2.7. The gap above 2 is correlation
no local response rule can produce, regardless of mechanism.

Usage: python3 vortex_bell.py
"""

import numpy as np

rng = np.random.default_rng(42)
N = 400_000

lam = rng.normal(size=(N, 3))
lam /= np.linalg.norm(lam, axis=1, keepdims=True)


def detector(angle: float) -> np.ndarray:
    """Current direction in the x-z plane at the given angle (radians)."""
    return np.array([np.sin(angle), 0.0, np.cos(angle)])


def outcome_friction(spin_axis: np.ndarray, a_vec: np.ndarray) -> np.ndarray:
    """Deterministic tilt/friction capture: up if rotation sense is conducive
    to the local current, down if it fights it. Bistable — no middle."""
    return np.sign(spin_axis @ a_vec + 1e-12)


def outcome_friction_stochastic(spin_axis, a_vec, rng) -> np.ndarray:
    """Softer variant: near-orthogonal encounters are chaotic; tilt adaptation
    succeeds with probability rising with alignment."""
    c = spin_axis @ a_vec
    p_up = np.clip(0.5 * (1 + np.sign(c) * np.abs(c) ** 0.5), 0, 1)
    return np.where(rng.random(len(c)) < p_up, 1.0, -1.0)


def E(a_ang: float, b_ang: float, rule=outcome_friction, rng=None) -> float:
    """Correlation <A*B> for detector angles a_ang, b_ang."""
    a, b = detector(a_ang), detector(b_ang)
    if rng is None:
        A, B = rule(lam, a), rule(-lam, b)
    else:
        A, B = rule(lam, a, rng), rule(-lam, b, rng)
    return float(np.mean(A * B))


def chsh(a1, a2, b1, b2, **kw) -> float:
    return E(a1, b1, **kw) + E(a1, b2, **kw) + E(a2, b1, **kw) - E(a2, b2, **kw)


def best_chsh(step_deg: float = 5.0):
    """Brute-force the model's best possible |S| over a grid of settings."""
    grid = np.deg2rad(np.arange(0, 180, step_deg))
    Ec = {(i, j): E(x, y) for i, x in enumerate(grid) for j, y in enumerate(grid)}
    best, best_set = 0.0, None
    n = len(grid)
    for i1 in range(n):
        for i2 in range(n):
            for j1 in range(n):
                for j2 in range(n):
                    S = Ec[i1, j1] + Ec[i1, j2] + Ec[i2, j1] - Ec[i2, j2]
                    if abs(S) > best:
                        best, best_set = abs(S), (i1, i2, j1, j2)
    deg = lambda k: round(float(np.rad2deg(grid[k])))
    return best, tuple(deg(k) for k in best_set)


if __name__ == "__main__":
    a1, a2 = 0.0, np.pi / 2          # Alice: 0, 90 deg
    b1, b2 = np.pi / 4, 3 * np.pi / 4  # Bob: 45, 135 deg (QM-optimal)

    print("At the QM-optimal settings (0/90, 45/135):")
    print(f"  deterministic friction rule: |S| = {abs(chsh(a1, a2, b1, b2)):.4f}")
    print(f"  stochastic variant:          |S| = "
          f"{abs(chsh(a1, a2, b1, b2, rule=outcome_friction_stochastic, rng=rng)):.4f}")

    S, settings = best_chsh()
    print(f"\nModel's best settings A:{settings[0]},{settings[1]} "
          f"B:{settings[2]},{settings[3]}  ->  |S| = {S:.3f}")
    print(f"Quantum mechanics, best settings:  S = {2 * np.sqrt(2):.3f}")
    print(f"Measured (Delft/NIST/Vienna 2015): S ~ 2.4-2.7")

    th = np.pi / 8
    print(f"\nE(22.5 deg): model = {E(0.0, th):.4f}, QM/experiment = {-np.cos(th):.4f}")
