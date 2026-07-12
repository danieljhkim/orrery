"""Measure steady radial flow in a consuming spherical counting lattice.

The apparatus is a one-cell-thick radial reduction of a three-dimensional
lattice.  Each shell stores a point density.  Points hop across a shell face
toward the lower-density neighbour at the nearest-neighbour diffusion rate;
the outer ghost shell is a unit-density reservoir.  A local sink destroys
points either only in the innermost shell or by the same absolute budget in
every equal-width shell.  Implicit time steps evolve the initially full
lattice until its density field is stationary.

The script measures, rather than assumes, the resulting face flux and radial
speed, consumption per shell, standing density deficit, and the destruction
hazard accumulated by a comoving point.  It reports power-law fits and their
regression uncertainty without making a theory-level judgment.

Run with:
    uv run lab/sims/dynamical-consumption-lattice/main.py
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
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import factorized, spsolve

from orrery import rng


SEED = 42
N_SHELLS = 1000
INNER_RADIUS = 1.0
SHELL_WIDTH = 1.0
DIFFUSIVITY = 10_000.0
RESERVOIR_DENSITY = 1.0
TARGET_INNER_DEFICIT = 0.05
TIME_STEP = 2.0
MAX_STEPS = 2000
STEADY_TOLERANCE = 1.0e-9
FIT_MIN_RADIUS = 10.0
FIT_MAX_RADIUS = 500.0
MEASUREMENT_TIME = 50.0


@dataclass(frozen=True)
class PowerFit:
    exponent: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    intercept: float
    radial_decades: float
    samples: int


@dataclass(frozen=True)
class ShapeFit:
    amplitude: float
    relative_rmse: float
    r_squared: float | None


def lattice_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = INNER_RADIUS + SHELL_WIDTH * np.arange(N_SHELLS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    return edges, centers, volumes


def diffusion_operator(
    edges: np.ndarray, centers: np.ndarray, volumes: np.ndarray
) -> tuple[csc_matrix, np.ndarray, np.ndarray]:
    """Return dn/dt = A n + boundary - sink/volume and face conductances."""
    conductance = np.zeros(N_SHELLS + 1)
    center_spacing = np.diff(centers)
    conductance[1:N_SHELLS] = (
        4.0 * np.pi * DIFFUSIVITY * edges[1:N_SHELLS] ** 2 / center_spacing
    )
    outer_spacing = edges[-1] - centers[-1]
    conductance[-1] = 4.0 * np.pi * DIFFUSIVITY * edges[-1] ** 2 / outer_spacing

    lower = conductance[1:N_SHELLS] / volumes[1:]
    upper = conductance[1:N_SHELLS] / volumes[:-1]
    diagonal = -(conductance[:-1] + conductance[1:]) / volumes
    operator = diags((lower, diagonal, upper), (-1, 0, 1), format="csc")
    boundary = np.zeros(N_SHELLS)
    boundary[-1] = conductance[-1] * RESERVOIR_DENSITY / volumes[-1]
    return operator, boundary, conductance


def sink_rates(rule: str) -> np.ndarray:
    rates = np.zeros(N_SHELLS)
    outer_radius = INNER_RADIUS + N_SHELLS * SHELL_WIDTH
    if rule == "central-only":
        rates[0] = (
            4.0
            * np.pi
            * DIFFUSIVITY
            * TARGET_INNER_DEFICIT
            / (1.0 / INNER_RADIUS - 1.0 / outer_radius)
        )
    elif rule == "equal-per-shell":
        rates[:] = (
            4.0
            * np.pi
            * DIFFUSIVITY
            * TARGET_INNER_DEFICIT
            / np.log(outer_radius / INNER_RADIUS)
        )
    else:
        raise ValueError(f"unknown rule: {rule}")
    return rates


def face_flux(density: np.ndarray, conductance: np.ndarray) -> np.ndarray:
    flux = np.zeros(N_SHELLS + 1)
    flux[1:N_SHELLS] = conductance[1:N_SHELLS] * (
        density[1:] - density[:-1]
    )
    flux[-1] = conductance[-1] * (RESERVOIR_DENSITY - density[-1])
    return flux


def power_fit(radius: np.ndarray, values: np.ndarray) -> PowerFit:
    x = np.log(radius)
    y = np.log(values)
    design = np.column_stack((np.ones_like(x), x))
    intercept, exponent = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([intercept, exponent])
    variance = float(np.sum(residual**2) / (len(x) - 2))
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(covariance[1, 1]))
    return PowerFit(
        exponent=float(exponent),
        standard_error=standard_error,
        ci95_low=float(exponent - 1.96 * standard_error),
        ci95_high=float(exponent + 1.96 * standard_error),
        intercept=float(intercept),
        radial_decades=float(np.log10(radius.max() / radius.min())),
        samples=len(radius),
    )


def candidate_comparison(
    radius: np.ndarray, values: np.ndarray
) -> dict[str, dict[str, float]]:
    log_radius = np.log(radius)
    log_values = np.log(values)
    comparisons = {}
    for name, exponent in (
        ("flux-conserving", -2.0),
        ("equal-per-shell-budget", -1.0),
        ("free-fall", -0.5),
    ):
        intercept = float(np.mean(log_values - exponent * log_radius))
        residual = log_values - (intercept + exponent * log_radius)
        comparisons[name] = {
            "candidate_exponent": exponent,
            "log_rmse": float(np.sqrt(np.mean(residual**2))),
        }
    return comparisons


def static_inverse_radius_fit(radius: np.ndarray, values: np.ndarray) -> ShapeFit:
    template = 1.0 / radius - 1.0 / (INNER_RADIUS + N_SHELLS * SHELL_WIDTH)
    amplitude = float(np.dot(template, values) / np.dot(template, template))
    predicted = amplitude * template
    residual = values - predicted
    scale = max(float(np.max(values) - np.min(values)), np.finfo(float).eps)
    relative_rmse = float(np.sqrt(np.mean(residual**2)) / scale)
    total = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = (
        None
        if total <= np.finfo(float).eps
        else 1.0 - float(np.sum(residual**2)) / total
    )
    return ShapeFit(amplitude, relative_rmse, r_squared)


def run_rule(rule: str) -> tuple[dict, dict[str, np.ndarray]]:
    edges, centers, volumes = lattice_geometry()
    operator, boundary, conductance = diffusion_operator(edges, centers, volumes)
    sink = sink_rates(rule)
    forcing = boundary - sink / volumes
    step_matrix = csc_matrix(diags(np.ones(N_SHELLS)) - TIME_STEP * operator)
    advance = factorized(step_matrix)
    density = np.full(N_SHELLS, RESERVOIR_DENSITY)
    diffusion_time = (edges[-1] - edges[0]) ** 2 / DIFFUSIVITY
    relative_change = np.inf

    for step in range(1, MAX_STEPS + 1):
        next_density = advance(density + TIME_STEP * forcing)
        deficit_scale = max(float(np.max(RESERVOIR_DENSITY - next_density)), 1.0e-15)
        relative_change = float(np.max(np.abs(next_density - density)) / deficit_scale)
        density = next_density
        if relative_change < STEADY_TOLERANCE:
            break
    else:
        raise RuntimeError(f"{rule} failed to reach steady tolerance")

    flux = face_flux(density, conductance)
    balance = operator @ density + forcing
    balance_flux_units = balance * volumes
    conservation_residual = float(
        np.max(np.abs(balance_flux_units)) / np.sum(sink)
    )

    face_radius = edges[1:]
    face_density = np.empty(N_SHELLS)
    face_density[:-1] = 0.5 * (density[:-1] + density[1:])
    face_density[-1] = 0.5 * (density[-1] + RESERVOIR_DENSITY)
    speed = flux[1:] / (4.0 * np.pi * face_radius**2 * face_density)
    fit_mask = (
        (face_radius >= FIT_MIN_RADIUS)
        & (face_radius <= FIT_MAX_RADIUS)
        & (speed > 0.0)
    )
    flow_fit = power_fit(face_radius[fit_mask], speed[fit_mask])

    random = rng(SEED + (0 if rule == "central-only" else 1))
    measured_counts = random.poisson(sink * MEASUREMENT_TIME)
    sink_mask = (centers >= FIT_MIN_RADIUS) & (centers <= FIT_MAX_RADIUS)
    positive_sink = sink_mask & (measured_counts > 0)
    sink_fit = None
    if np.count_nonzero(positive_sink) > 2:
        sink_fit = power_fit(
            centers[positive_sink], measured_counts[positive_sink] / MEASUREMENT_TIME
        )

    deficit = (RESERVOIR_DENSITY - density) / RESERVOIR_DENSITY
    shell_speed = 0.5 * (speed + np.concatenate(([speed[0]], speed[:-1])))
    fractional_sink_rate = sink / (density * volumes)
    crossing_time = SHELL_WIDTH / np.maximum(shell_speed, np.finfo(float).tiny)
    shell_hazard = fractional_sink_rate * crossing_time
    accumulated_hazard = np.cumsum(shell_hazard[::-1])[::-1]
    comoving_dilution = -np.expm1(-accumulated_hazard)

    static_mask = (centers >= FIT_MIN_RADIUS) & (centers <= FIT_MAX_RADIUS)
    standing_static_fit = static_inverse_radius_fit(
        centers[static_mask], deficit[static_mask]
    )
    comoving_static_fit = static_inverse_radius_fit(
        centers[static_mask], comoving_dilution[static_mask]
    )

    lattice_signal_speed = DIFFUSIVITY / SHELL_WIDTH
    flow_sigma = shell_speed**2 / (2.0 * lattice_signal_speed**2)
    identity_mask = static_mask & (deficit > 0.0)
    identity_log_ratio = np.log10(flow_sigma[identity_mask] / deficit[identity_mask])
    identity = {
        "c_lattice": lattice_signal_speed,
        "definition": "c = nearest-neighbour hop rate D/dr^2 times hop length dr",
        "median_sigma_flow_over_standing_sigma": float(
            np.median(10.0**identity_log_ratio)
        ),
        "rms_log10_ratio": float(np.sqrt(np.mean(identity_log_ratio**2))),
        "passes_factor_two_shape_and_normalization": bool(
            np.max(np.abs(identity_log_ratio)) <= np.log10(2.0)
        ),
    }

    result = {
        "rule": rule,
        "rule_definition": (
            "all destruction occurs in the innermost cell"
            if rule == "central-only"
            else "the same absolute mean point budget is destroyed in every equal-width shell"
        ),
        "steady_state": {
            "reached": True,
            "steps": step,
            "elapsed_time": step * TIME_STEP,
            "outer_diffusion_time": diffusion_time,
            "last_step_change_over_max_deficit": relative_change,
            "tolerance": STEADY_TOLERANCE,
            "max_shell_balance_over_total_sink": conservation_residual,
            "inner_density": float(density[0]),
            "outer_density": float(density[-1]),
        },
        "flow_fit": asdict(flow_fit),
        "candidate_comparison": candidate_comparison(
            face_radius[fit_mask], speed[fit_mask]
        ),
        "sink_fit": asdict(sink_fit) if sink_fit else None,
        "sink_localization": (
            "innermost shell only" if rule == "central-only" else "distributed"
        ),
        "total_consumption_rate": float(np.sum(sink)),
        "standing_deficit_measurements": {
            "inner": float(deficit[0]),
            "at_fit_min": float(deficit[np.searchsorted(centers, FIT_MIN_RADIUS)]),
            "at_fit_max": float(deficit[np.searchsorted(centers, FIT_MAX_RADIUS)]),
        },
        "comoving_dilution_measurements": {
            "inner": float(comoving_dilution[0]),
            "at_fit_min": float(
                comoving_dilution[np.searchsorted(centers, FIT_MIN_RADIUS)]
            ),
            "at_fit_max": float(
                comoving_dilution[np.searchsorted(centers, FIT_MAX_RADIUS)]
            ),
            "maximum": float(np.max(comoving_dilution)),
        },
        "standing_deficit_vs_static_inverse_r": asdict(standing_static_fit),
        "comoving_dilution_vs_static_inverse_r": asdict(comoving_static_fit),
        "sigma_identity": identity,
    }
    profiles = {
        "radius": centers,
        "face_radius": face_radius,
        "density": density,
        "deficit": deficit,
        "speed": speed,
        "sink": sink,
        "measured_sink": measured_counts / MEASUREMENT_TIME,
        "comoving_dilution": comoving_dilution,
        "flow_sigma": flow_sigma,
    }
    return result, profiles


def make_plot(all_profiles: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    colors = {"central-only": "#59c3ff", "equal-per-shell": "#ffb45c"}
    for rule, profile in all_profiles.items():
        color = colors[rule]
        axes[0, 0].loglog(profile["face_radius"], profile["speed"], color=color, label=rule)
        positive = profile["measured_sink"] > 0
        axes[0, 1].loglog(
            profile["radius"][positive], profile["measured_sink"][positive],
            ".", color=color, alpha=0.6, label=rule,
        )
        axes[1, 0].loglog(profile["radius"], profile["deficit"], color=color, label=f"{rule}: standing")
        axes[1, 0].loglog(profile["radius"], profile["comoving_dilution"], "--", color=color, label=f"{rule}: comoving")
        axes[1, 1].loglog(profile["radius"], profile["deficit"], color=color, label=f"{rule}: standing sigma")
        axes[1, 1].loglog(profile["radius"], profile["flow_sigma"], "--", color=color, label=f"{rule}: v²/2c²")

    reference_radius = np.geomspace(FIT_MIN_RADIUS, FIT_MAX_RADIUS, 200)
    for exponent, style, label in ((-2.0, ":", "r^-2"), (-1.0, "--", "r^-1"), (-0.5, "-.", "r^-1/2")):
        anchor = all_profiles["central-only"]["speed"][int(FIT_MIN_RADIUS)]
        axes[0, 0].loglog(reference_radius, anchor * (reference_radius / FIT_MIN_RADIUS) ** exponent, style, color="#999999", alpha=0.7, label=label)
    static_shape = 1.0 / reference_radius - 1.0 / (INNER_RADIUS + N_SHELLS * SHELL_WIDTH)
    axes[1, 0].loglog(reference_radius, 0.5 * static_shape / static_shape[0], ":", color="#dddddd", label="scaled static 1/r")

    axes[0, 0].set(title="Measured inward speed", xlabel="radius", ylabel="v(r)")
    axes[0, 1].set(title="Measured consumption per shell", xlabel="radius", ylabel="points / time")
    axes[1, 0].set(title="Snapshot consistency", xlabel="radius", ylabel="fractional dilution")
    axes[1, 1].set(title="River identity check", xlabel="radius", ylabel="sigma")
    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(fontsize=7)
    fig.suptitle("Dynamical-consumption lattice: measured steady states")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    profiles = {}
    for rule in ("central-only", "equal-per-shell"):
        results[rule], profiles[rule] = run_rule(rule)

    payload = {
        "schema_version": 1,
        "seed": SEED,
        "apparatus": {
            "shells": N_SHELLS,
            "radius_range": [INNER_RADIUS, INNER_RADIUS + N_SHELLS * SHELL_WIDTH],
            "shell_width": SHELL_WIDTH,
            "diffusivity": DIFFUSIVITY,
            "reservoir_density": RESERVOIR_DENSITY,
            "fit_radius_range": [FIT_MIN_RADIUS, FIT_MAX_RADIUS],
            "fit_radial_decades": float(np.log10(FIT_MAX_RADIUS / FIT_MIN_RADIUS)),
            "steady_tolerance": STEADY_TOLERANCE,
        },
        "rules": results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    make_plot(profiles, args.output_dir / "profiles.png")

    for rule, result in results.items():
        fit = result["flow_fit"]
        print(
            f"{rule}: v ~ r^{fit['exponent']:.5f} ± {fit['standard_error']:.5f}; "
            f"steady change={result['steady_state']['last_step_change_over_max_deficit']:.2e}"
        )


if __name__ == "__main__":
    main()
