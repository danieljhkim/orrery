"""Shared helpers for orrery Python sims (Monte Carlo / numerics lane).

Keep this deliberately small: only pull a helper in here once a second sim
actually needs it.
"""

import numpy as np

__all__ = ["rng", "unit_vectors", "chsh_S"]


def rng(seed: int = 42) -> np.random.Generator:
    """The house RNG — seeded by default so runs are reproducible by default."""
    return np.random.default_rng(seed)


def unit_vectors(r: np.random.Generator, n: int, dim: int = 3) -> np.ndarray:
    """n uniformly distributed unit vectors, shape (n, dim)."""
    v = r.normal(size=(n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def chsh_S(E, a: float, ap: float, b: float, bp: float) -> float:
    """CHSH statistic S = E(a,b) - E(a,b') + E(a',b) + E(a',b').

    E is a correlation function of two detector angles (radians).
    Local hidden-variable models obey |S| <= 2; QM reaches 2*sqrt(2).
    """
    return E(a, b) - E(a, bp) + E(ap, b) + E(ap, bp)
