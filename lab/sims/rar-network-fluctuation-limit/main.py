"""Bound galaxy-to-galaxy fluctuations in a RAR-like extra acceleration.

The mock apparatus generates 153 disk galaxies and 2693 resolved points, uses
the McGaugh-Lelli-Schombert RAR as its static ground truth, multiplies only the
extra-acceleration term by a median-preserving log-normal deviate of width
``sigma_net`` dex, and then applies galaxy-correlated distance, inclination,
stellar M/L and gas-calibration errors plus pointwise velocity errors.  It
reports the recovered residual width and the sigma_net at which 95% of mocks
are broader than the observed 0.11-dex Gaussian ridge.

This is a forward-model sensitivity calculation, not a fit to the SPARC
catalog.  The default 0.75 error-budget scale is deliberately paired with
zero-, half-, and full-budget sensitivity cases because the 2016 quadrature
budget (0.12 dex) is itself slightly wider than the cited 0.11-dex ridge.

Usage: uv run lab/sims/rar-network-fluctuation-limit/main.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orrery import rng


G_DAGGER_MS2 = 1.20e-10
OBSERVED_SCATTER_DEX = 0.11
DEFAULT_SIGMA_GRID_DEX = np.linspace(0.0, 0.35, 36)


@dataclass(frozen=True)
class ErrorBudget:
    """One-sigma observational nuisance amplitudes before global scaling."""

    distance_dex: float = 0.08
    inclination_deg: float = 4.0
    velocity_gobs_dex: float = 0.03
    stellar_ml_dex: float = 0.11
    hi_flux_dex: float = 0.04

    def scaled(self, factor: float) -> "ErrorBudget":
        return ErrorBudget(**{key: factor * value for key, value in asdict(self).items()})


@dataclass(frozen=True)
class MockDraw:
    """A SPARC-shaped population and reusable standard-normal error draws."""

    galaxy_index: np.ndarray
    gbar_true_ms2: np.ndarray
    extra_true_ms2: np.ndarray
    stellar_fraction: np.ndarray
    inclination_true_rad: np.ndarray
    z_network_galaxy: np.ndarray
    z_network_point: np.ndarray
    z_distance: np.ndarray
    z_inclination: np.ndarray
    z_stellar_ml: np.ndarray
    z_hi_flux: np.ndarray
    z_velocity: np.ndarray


def rar_acceleration(gbar_ms2: np.ndarray) -> np.ndarray:
    """McGaugh et al. 2016 equation 4, evaluated stably."""
    x = np.sqrt(gbar_ms2 / G_DAGGER_MS2)
    return gbar_ms2 / (-np.expm1(-x))


def make_mock_draw(generator: np.random.Generator, galaxies: int, points: int) -> MockDraw:
    if galaxies < 2 or points < 5 * galaxies:
        raise ValueError("need at least two galaxies and five points per galaxy")

    # SPARC has a broad, unequal number of resolved measurements per galaxy.
    weights = generator.lognormal(mean=0.0, sigma=0.55, size=galaxies)
    counts = np.full(galaxies, 5, dtype=int)
    counts += generator.multinomial(points - 5 * galaxies, weights / weights.sum())
    galaxy_index = np.repeat(np.arange(galaxies), counts)

    # Each curve runs from a high-acceleration inner point to a low-acceleration
    # outer point. Galaxy centers span the surface-brightness range of SPARC.
    center_log_gbar = generator.uniform(-11.45, -8.70, size=galaxies)
    log_gbar = np.empty(points)
    stellar_fraction = np.empty(points)
    gas_richness = generator.normal(0.0, 0.35, size=galaxies)
    cursor = 0
    for galaxy, count in enumerate(counts):
        radial_rank = np.linspace(0.0, 1.0, count)
        values = (
            center_log_gbar[galaxy]
            + 0.95 * (1.0 - 2.0 * radial_rank)
            + generator.normal(0.0, 0.07, size=count)
        )
        values = np.clip(values, -12.35, -8.0)
        log_gbar[cursor : cursor + count] = values
        stellar_fraction[cursor : cursor + count] = 1.0 / (
            1.0 + np.exp(-1.45 * (values + 10.55) + gas_richness[galaxy])
        )
        cursor += count

    gbar_true = 10.0**log_gbar
    gobs_rar = rar_acceleration(gbar_true)
    inclinations = np.arccos(
        generator.uniform(np.cos(np.deg2rad(80.0)), np.cos(np.deg2rad(30.0)), size=galaxies)
    )

    return MockDraw(
        galaxy_index=galaxy_index,
        gbar_true_ms2=gbar_true,
        extra_true_ms2=gobs_rar - gbar_true,
        stellar_fraction=stellar_fraction,
        inclination_true_rad=inclinations,
        z_network_galaxy=generator.normal(size=galaxies),
        z_network_point=generator.normal(size=points),
        z_distance=generator.normal(size=galaxies),
        z_inclination=generator.normal(size=galaxies),
        z_stellar_ml=generator.normal(size=galaxies),
        z_hi_flux=generator.normal(size=galaxies),
        z_velocity=generator.normal(size=points),
    )


def observe(draw: MockDraw, sigma_net_dex: float, radial_coherence: float,
            budget: ErrorBudget) -> tuple[np.ndarray, np.ndarray]:
    """Apply network fluctuations and the observational nuisance model."""
    if not 0.0 <= radial_coherence <= 1.0:
        raise ValueError("radial_coherence must lie in [0, 1]")
    galaxy = draw.galaxy_index
    z_network = (
        np.sqrt(radial_coherence) * draw.z_network_galaxy[galaxy]
        + np.sqrt(1.0 - radial_coherence) * draw.z_network_point
    )
    gobs_true = (
        draw.gbar_true_ms2
        + draw.extra_true_ms2 * 10.0 ** (sigma_net_dex * z_network)
    )

    distance_factor = 10.0 ** (budget.distance_dex * draw.z_distance[galaxy])
    assumed_inclination = np.clip(
        draw.inclination_true_rad
        + np.deg2rad(budget.inclination_deg * draw.z_inclination),
        np.deg2rad(15.0),
        np.deg2rad(89.0),
    )
    inclination_factor = (
        np.sin(draw.inclination_true_rad[galaxy]) / np.sin(assumed_inclination[galaxy])
    ) ** 2
    gobs_measured = (
        gobs_true / distance_factor * inclination_factor
        * 10.0 ** (budget.velocity_gobs_dex * draw.z_velocity)
    )

    star_factor = 10.0 ** (budget.stellar_ml_dex * draw.z_stellar_ml[galaxy])
    gas_factor = 10.0 ** (budget.hi_flux_dex * draw.z_hi_flux[galaxy])
    gbar_measured = draw.gbar_true_ms2 * (
        draw.stellar_fraction * star_factor
        + (1.0 - draw.stellar_fraction) * gas_factor
    )
    return gbar_measured, gobs_measured


def residual_scatter_dex(gbar_ms2: np.ndarray, gobs_ms2: np.ndarray) -> float:
    residual = np.log10(gobs_ms2) - np.log10(rar_acceleration(gbar_ms2))
    return float(np.std(residual, ddof=1))


def scatter_trials(seed: int, sigma_grid: np.ndarray, replicates: int, galaxies: int,
                   points: int, error_scale: float, radial_coherence: float,
                   source: str | None = None) -> np.ndarray:
    """Return shape (replicates, len(sigma_grid)) using common random numbers."""
    base = ErrorBudget().scaled(error_scale)
    if source is not None:
        values = {key: 0.0 for key in asdict(base)}
        values[source] = getattr(base, source)
        base = ErrorBudget(**values)
    generator = rng(seed)
    out = np.empty((replicates, len(sigma_grid)))
    for trial in range(replicates):
        draw = make_mock_draw(generator, galaxies, points)
        for column, sigma_net in enumerate(sigma_grid):
            gbar, gobs = observe(draw, float(sigma_net), radial_coherence, base)
            out[trial, column] = residual_scatter_dex(gbar, gobs)
    return out


def summarize_curve(sigma_grid: np.ndarray, trials: np.ndarray) -> dict:
    percentiles = np.percentile(trials, [5.0, 50.0, 95.0], axis=0)
    probability = np.mean(trials <= OBSERVED_SCATTER_DEX, axis=0)
    lower = percentiles[0]
    crossings = np.flatnonzero(lower >= OBSERVED_SCATTER_DEX)
    if len(crossings) == 0:
        limit = None
    elif crossings[0] == 0:
        limit = 0.0
    else:
        hi = int(crossings[0])
        lo = hi - 1
        fraction = (
            (OBSERVED_SCATTER_DEX - lower[lo]) / (lower[hi] - lower[lo])
            if lower[hi] != lower[lo] else 0.0
        )
        limit = float(sigma_grid[lo] + fraction * (sigma_grid[hi] - sigma_grid[lo]))
    return {
        "sigma_net_dex": sigma_grid.tolist(),
        "scatter_p05_dex": percentiles[0].tolist(),
        "scatter_median_dex": percentiles[1].tolist(),
        "scatter_p95_dex": percentiles[2].tolist(),
        "probability_scatter_le_observed": probability.tolist(),
        "upper_limit_95pct_dex": limit,
        "baseline_scatter_median_dex": float(percentiles[1, 0]),
    }


def run_curve(seed: int, sigma_grid: np.ndarray, replicates: int, galaxies: int,
              points: int, error_scale: float, radial_coherence: float) -> dict:
    trials = scatter_trials(seed, sigma_grid, replicates, galaxies, points,
                            error_scale, radial_coherence)
    summary = summarize_curve(sigma_grid, trials)
    summary.update({
        "seed": seed,
        "replicates": replicates,
        "galaxies": galaxies,
        "points": points,
        "error_scale": error_scale,
        "radial_coherence": radial_coherence,
    })
    return summary


def make_plot(path: Path, result: dict) -> None:
    curve = result["exclusion_curve"]
    x = np.asarray(curve["sigma_net_dex"])
    median = np.asarray(curve["scatter_median_dex"])
    low = np.asarray(curve["scatter_p05_dex"])
    high = np.asarray(curve["scatter_p95_dex"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axes[0].fill_between(x, low, high, color="#8db3e2", alpha=0.35, label="5–95% mocks")
    axes[0].plot(x, median, color="#315f9b", lw=2, label="median")
    axes[0].axhline(OBSERVED_SCATTER_DEX, color="#a33b20", ls="--", label="observed 0.11 dex")
    limit = curve["upper_limit_95pct_dex"]
    if limit is not None:
        axes[0].axvline(limit, color="#a33b20", ls=":", label=f"95% limit {limit:.3f} dex")
    axes[0].set(xlabel=r"network fluctuation $\sigma_{net}$ (dex)",
                ylabel="recovered RAR scatter (dex)", title="Exclusion curve")
    axes[0].legend(fontsize=8)

    sensitivity = result["sensitivity"]
    labels = list(sensitivity)
    values = [sensitivity[label]["upper_limit_95pct_dex"] for label in labels]
    plot_values = [0.0 if value is None else value for value in values]
    axes[1].barh(np.arange(len(labels)), plot_values, color="#6b9f78")
    axes[1].set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set(xlabel=r"95% upper limit on $\sigma_{net}$ (dex)", title="Model sensitivity")

    targets = result["literature_scatter_budget_dex"]
    measured = result["mock_component_scatter_median_dex"]
    source_labels = list(targets)
    positions = np.arange(len(source_labels))
    axes[2].bar(positions - 0.2, [targets[key] for key in source_labels], width=0.4,
                color="#777777", label="2016 residual budget")
    axes[2].bar(positions + 0.2, [measured[key] for key in source_labels], width=0.4,
                color="#d29b54", label="mock, full scale")
    axes[2].set_xticks(positions, source_labels, rotation=35, ha="right", fontsize=8)
    axes[2].set(ylabel="scatter contribution (dex)", title="Error-budget check")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("SPARC-like RAR network-force fluctuation bound (ORB-10195)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=256)
    parser.add_argument("--galaxies", type=int, default=153)
    parser.add_argument("--points", type=int, default=2693)
    parser.add_argument("--error-scale", type=float, default=0.75)
    parser.add_argument("--radial-coherence", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("assets"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sigma_grid = DEFAULT_SIGMA_GRID_DEX
    curve = run_curve(args.seed, sigma_grid, args.replicates, args.galaxies, args.points,
                      args.error_scale, args.radial_coherence)

    sensitivity_specs = {
        "zero errors, coherent": (0.0, 1.0),
        "half budget, coherent": (0.5, 1.0),
        "fiducial 0.75, coherent": (0.75, 1.0),
        "full budget, coherent": (1.0, 1.0),
        "fiducial, mixed coherence": (0.75, 0.5),
        "fiducial, pointwise": (0.75, 0.0),
    }
    sensitivity = {
        label: run_curve(args.seed + 10_000, sigma_grid, 128,
                         args.galaxies, args.points, scale, coherence)
        for label, (scale, coherence) in sensitivity_specs.items()
    }

    convergence_specs = [
        (64, args.galaxies, args.points),
        (128, args.galaxies, args.points),
        (256, args.galaxies, args.points),
        (128, 76, 1346),
        (128, 306, 5386),
    ]
    convergence = {
        f"replicates={replicates},galaxies={galaxies}": run_curve(
            args.seed + (0 if index < 3 else 20_000 + index),
            sigma_grid, replicates, galaxies, points,
            args.error_scale, args.radial_coherence
        )
        for index, (replicates, galaxies, points) in enumerate(convergence_specs)
    }

    source_map = {
        "velocity": "velocity_gobs_dex",
        "inclination": "inclination_deg",
        "distance": "distance_dex",
        "stellar_ML": "stellar_ml_dex",
        "HI_flux": "hi_flux_dex",
    }
    component_scatter = {}
    for index, (label, field) in enumerate(source_map.items()):
        trials = scatter_trials(args.seed + 30_000 + index, np.array([0.0]), 256,
                                args.galaxies, args.points, 1.0, 1.0, source=field)
        component_scatter[label] = float(np.median(trials[:, 0]))

    limits = [value["upper_limit_95pct_dex"] for value in convergence.values()]
    finite_limits = [value for value in limits if value is not None]
    result = {
        "schema_version": 1,
        "question": (
            "What galaxy-to-galaxy log-normal fluctuation in the RAR extra acceleration "
            "is excluded by an observed 0.11-dex residual width?"
        ),
        "interpretation": {
            "exclusion_rule": (
                "sigma_net is excluded at 95% when the 5th percentile recovered scatter "
                "is at least 0.11 dex"
            ),
            "supports_dynamic_network_if": "a nonzero fluctuation is required by data (not tested here)",
            "weakens_dynamic_network_if": "its predicted coherent sigma_net exceeds the upper limit",
            "unresolved_if": "its fluctuations are non-log-normal, time-averaged, or correlated with baryons",
        },
        "rar": {"g_dagger_ms2": G_DAGGER_MS2, "observed_gaussian_width_dex": OBSERVED_SCATTER_DEX},
        "network_model": {
            "formula": "gobs = gbar + (g_RAR(gbar)-gbar) * 10**delta",
            "delta_distribution": "Normal(0, sigma_net_dex); zero-median log-normal multiplier",
            "default_radial_coherence": args.radial_coherence,
        },
        "error_budget_parameters": asdict(ErrorBudget()),
        "literature_scatter_budget_dex": {
            "velocity": 0.03,
            "inclination": 0.05,
            "distance": 0.08,
            "stellar_ML": 0.06,
            "HI_flux": 0.01,
        },
        "mock_component_scatter_median_dex": component_scatter,
        "exclusion_curve": curve,
        "sensitivity": sensitivity,
        "convergence": convergence,
        "convergence_limit_range_dex": (
            [float(min(finite_limits)), float(max(finite_limits))] if finite_limits else []
        ),
        "limitations": [
            "This is a synthetic SPARC-shaped population, not a hierarchical fit to the catalog.",
            "Distance, inclination, M/L, and gas errors are simplified Gaussian galaxy-level nuisances.",
            "The historical full error budget is approximate and already predicts about the observed width; the bound is therefore error-model dominated.",
            "The injected fluctuation is log-normal and independent of galaxy properties; structured or time-averaged network dynamics can evade this mapping.",
            "Scatter is measured about the fixed published RAR rather than refitting its functional form in every realization.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    plot_path = args.output_dir / "exclusion-curve.png"
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    make_plot(plot_path, result)

    print(f"mock: {args.galaxies} galaxies, {args.points} points, seed {args.seed}")
    print(f"baseline scatter: {curve['baseline_scatter_median_dex']:.4f} dex")
    if curve["upper_limit_95pct_dex"] is None:
        print("95% sigma_net upper limit: above scanned grid")
    else:
        print(f"95% sigma_net upper limit: {curve['upper_limit_95pct_dex']:.4f} dex")
    print(f"wrote {results_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
