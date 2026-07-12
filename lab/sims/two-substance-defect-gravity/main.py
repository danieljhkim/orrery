"""Test the gravity bridge of a conserved two-substance vortex lattice.

The apparatus keeps the companion lattice's two separately conserved density
fields, shared unit capacity, compensated void core, and winding diagnostic.
It asks three predeclared questions.  First, does a stationary compensated
core carry a nonzero 1/r scarcity monopole in three dimensions?  Second, does
a translated core leave a dynamically relaxed depletion/replenishment wake
whose measured velocity follows the imposed core velocity?  Third, when core
size and winding are varied independently, does the lattice itself select void
volume or flow energy as the source of a far field?

The result is negative for the claimed gravity bridge: conservation cancels
the stationary density monopole, so the far field is zero rather than 1/r or
1/r^2.  An uncompensated control does produce 1/r, demonstrating that the
measurement can resolve it.  A forced moving core produces a lagging wake that
tracks the imposed velocity, but this does not establish self-propelled defect
motion.  Core size controls only the compact density response; winding changes
the flow-energy diagnostic without changing scarcity.  The lattice therefore
selects neither proposed gravitational charge until an additional coupling or
nonconserving source law is supplied.

Run with:
    uv run lab/sims/two-substance-defect-gravity/main.py
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


STATIC_SIZE = 97
BASE_DENSITY = 0.5
CORE_SIGMA = 2.4
HALO_RATIO = 3.0
TOTAL_DEFICIT_AMPLITUDE = 0.72
STATIC_FIT_MIN = 36.0
STATIC_FIT_MAX = 44.0

WAKE_SIZE = 161
WAKE_CORE_SIGMA = 2.8
WAKE_DIFFUSIVITY = 0.35
WAKE_RELAXATION_TIME = 2.5
WAKE_TIME = 70.0
WAKE_SPEEDS = (0.15, 0.30, 0.45)


@dataclass(frozen=True)
class ShapeFit:
    amplitude: float
    rmse: float
    relative_rmse: float


def coordinate_grid(size: int, dimensions: int) -> tuple[np.ndarray, ...]:
    axis = np.arange(size, dtype=float) - size // 2
    return tuple(np.meshgrid(*([axis] * dimensions), indexing="ij"))


def compensated_deficit(radius2: np.ndarray, core_sigma: float) -> np.ndarray:
    """Return a void core plus expelled halo with exactly zero lattice sum."""
    # L-0005: a compensated core has no gravity monopole without an added source law.
    halo_sigma = HALO_RATIO * core_sigma
    core = np.exp(-radius2 / (2.0 * core_sigma**2))
    halo = np.exp(-radius2 / (2.0 * halo_sigma**2))
    # Use the discrete sums, rather than the continuum sigma ratio, so exact
    # conservation does not introduce a tiny uniform background on a finite box.
    profile = core - float(np.sum(core) / np.sum(halo)) * halo
    return TOTAL_DEFICIT_AMPLITUDE * profile


def density_fields(deficit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a balanced total deficit symmetrically between conserved A/B."""
    rho_a = BASE_DENSITY - 0.5 * deficit
    rho_b = BASE_DENSITY - 0.5 * deficit
    if min(float(np.min(rho_a)), float(np.min(rho_b))) <= 0.0:
        raise RuntimeError("density crossed zero; reduce the deficit amplitude")
    return rho_a, rho_b


def shell_measurements(values: np.ndarray, radius: np.ndarray) -> dict[str, np.ndarray]:
    shell = np.floor(radius).astype(int)
    count = np.bincount(shell.ravel())
    charge = np.bincount(shell.ravel(), weights=values.ravel())
    mean = np.divide(charge, count, out=np.zeros_like(charge), where=count > 0)
    centers = np.arange(len(charge), dtype=float) + 0.5
    return {"radius": centers, "count": count, "charge": charge, "mean": mean}


def spherical_potential(shell_charge: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Spherical-shell potential, with zero fixed beyond the finite lattice."""
    safe_radius = np.maximum(radius, 0.5)
    interior = np.cumsum(shell_charge)
    outer_terms = shell_charge / safe_radius
    exterior = np.cumsum(outer_terms[::-1])[::-1] - outer_terms
    return interior / safe_radius + exterior


def fit_template(template: np.ndarray, values: np.ndarray) -> ShapeFit:
    amplitude = float(np.dot(template, values) / np.dot(template, template))
    residual = values - amplitude * template
    rmse = float(np.sqrt(np.mean(residual**2)))
    scale = max(float(np.max(np.abs(values))), np.finfo(float).eps)
    return ShapeFit(amplitude, rmse, rmse / scale)


def stationary_experiment() -> tuple[dict, dict[str, np.ndarray]]:
    grid = coordinate_grid(STATIC_SIZE, 3)
    radius2 = sum(axis**2 for axis in grid)
    radius = np.sqrt(radius2)
    deficit = compensated_deficit(radius2, CORE_SIGMA)
    rho_a, rho_b = density_fields(deficit)

    compensated = shell_measurements(deficit, radius)
    core_only = TOTAL_DEFICIT_AMPLITUDE * np.exp(
        -radius2 / (2.0 * CORE_SIGMA**2)
    )
    control = shell_measurements(core_only, radius)
    compensated_potential = spherical_potential(
        compensated["charge"], compensated["radius"]
    )
    control_potential = spherical_potential(control["charge"], control["radius"])

    fit_mask = (
        (compensated["radius"] >= STATIC_FIT_MIN)
        & (compensated["radius"] <= STATIC_FIT_MAX)
    )
    fit_radius = compensated["radius"][fit_mask]
    compensated_inverse_r = fit_template(
        1.0 / fit_radius, compensated_potential[fit_mask]
    )
    compensated_inverse_r2 = fit_template(
        1.0 / fit_radius**2, compensated_potential[fit_mask]
    )
    control_inverse_r = fit_template(1.0 / fit_radius, control_potential[fit_mask])
    control_inverse_r2 = fit_template(1.0 / fit_radius**2, control_potential[fit_mask])

    expected_density_total = BASE_DENSITY * STATIC_SIZE**3
    net_monopole = float(np.sum(deficit))
    control_monopole = float(np.sum(core_only))
    result = {
        "lattice": {
            "dimensions": 3,
            "size": [STATIC_SIZE] * 3,
            "boundary": "open cube; spherical-shell estimator fixed to zero outside",
            "base_density_per_substance": BASE_DENSITY,
            "core_sigma_cells": CORE_SIGMA,
            "halo_sigma_cells": HALO_RATIO * CORE_SIGMA,
            "fit_radius_cells": [STATIC_FIT_MIN, STATIC_FIT_MAX],
        },
        "conservation": {
            "A_total_error": float(np.sum(rho_a) - expected_density_total),
            "B_total_error": float(np.sum(rho_b) - expected_density_total),
            "net_signed_deficit": net_monopole,
            "net_deficit_over_positive_void_volume": float(
                net_monopole / np.sum(np.clip(deficit, 0.0, None))
            ),
            "minimum_A_density": float(np.min(rho_a)),
            "minimum_B_density": float(np.min(rho_b)),
        },
        "compensated_core": {
            "positive_void_volume": float(np.sum(np.clip(deficit, 0.0, None))),
            "negative_halo_volume": float(-np.sum(np.clip(deficit, None, 0.0))),
            "far_max_absolute_potential": float(
                np.max(np.abs(compensated_potential[fit_mask]))
            ),
            "inverse_r_fit": asdict(compensated_inverse_r),
            "inverse_r2_fit": asdict(compensated_inverse_r2),
        },
        "uncompensated_control": {
            "net_deficit": control_monopole,
            "far_max_absolute_potential": float(
                np.max(np.abs(control_potential[fit_mask]))
            ),
            "inverse_r_fit": asdict(control_inverse_r),
            "inverse_r2_fit": asdict(control_inverse_r2),
        },
        "far_signal_fraction_of_uncompensated_control": float(
            np.max(np.abs(compensated_potential[fit_mask]))
            / np.max(np.abs(control_potential[fit_mask]))
        ),
    }
    profiles = {
        "radius": compensated["radius"],
        "shell_mean_deficit": compensated["mean"],
        "compensated_potential": compensated_potential,
        "control_potential": control_potential,
    }
    return result, profiles


def moving_target(
    xx: np.ndarray, yy: np.ndarray, center_x: float
) -> np.ndarray:
    radius2 = (xx - center_x) ** 2 + yy**2
    return compensated_deficit(radius2, WAKE_CORE_SIGMA)


def periodic_laplacian(values: np.ndarray) -> np.ndarray:
    return (
        np.roll(values, 1, axis=0)
        + np.roll(values, -1, axis=0)
        + np.roll(values, 1, axis=1)
        + np.roll(values, -1, axis=1)
        - 4.0 * values
    )


def local_core_centroid(
    deficit: np.ndarray, xx: np.ndarray, yy: np.ndarray, core_x: float
) -> tuple[float, float]:
    distance2 = (xx - core_x) ** 2 + yy**2
    window = np.exp(-distance2 / (2.0 * (4.0 * WAKE_CORE_SIGMA) ** 2))
    weights = np.clip(deficit, 0.0, None) * window
    total = float(np.sum(weights))
    return float(np.sum(xx * weights) / total), float(np.sum(yy * weights) / total)


def run_wake(speed: float, dt: float) -> tuple[dict, dict[str, np.ndarray]]:
    xx, yy = coordinate_grid(WAKE_SIZE, 2)
    start_x = -32.0
    deficit = moving_target(xx, yy, start_x)
    steps = int(round(WAKE_TIME / dt))
    sample_every = max(1, int(round(0.5 / dt)))
    times: list[float] = []
    core_positions: list[float] = []
    centroid_positions: list[float] = []
    centroid_y: list[float] = []

    for step in range(1, steps + 1):
        time = step * dt
        core_x = start_x + speed * time
        target = moving_target(xx, yy, core_x)
        deficit += dt * (
            WAKE_DIFFUSIVITY * periodic_laplacian(deficit)
            + (target - deficit) / WAKE_RELAXATION_TIME
        )
        deficit -= np.mean(deficit)
        if step % sample_every == 0:
            centroid_x, center_y = local_core_centroid(deficit, xx, yy, core_x)
            times.append(time)
            core_positions.append(core_x)
            centroid_positions.append(centroid_x)
            centroid_y.append(center_y)

    times_array = np.asarray(times)
    cores = np.asarray(core_positions)
    centroids = np.asarray(centroid_positions)
    settled = times_array >= 0.35 * WAKE_TIME
    measured_velocity = float(np.polyfit(times_array[settled], centroids[settled], 1)[0])
    lag = cores - centroids

    final_core_x = cores[-1]
    behind = (
        (xx < final_core_x - WAKE_CORE_SIGMA)
        & (xx > final_core_x - 5.0 * WAKE_CORE_SIGMA)
        & (np.abs(yy) < 3.0 * WAKE_CORE_SIGMA)
    )
    ahead = (
        (xx > final_core_x + WAKE_CORE_SIGMA)
        & (xx < final_core_x + 5.0 * WAKE_CORE_SIGMA)
        & (np.abs(yy) < 3.0 * WAKE_CORE_SIGMA)
    )
    behind_depletion = float(np.sum(np.clip(deficit[behind], 0.0, None)))
    ahead_depletion = float(np.sum(np.clip(deficit[ahead], 0.0, None)))
    flux_x, flux_y = np.gradient(deficit)
    flux_x *= WAKE_DIFFUSIVITY
    flux_y *= WAKE_DIFFUSIVITY

    result = {
        "imposed_core_velocity": speed,
        "measured_pattern_velocity": measured_velocity,
        "pattern_to_core_velocity_ratio": measured_velocity / speed,
        "settled_mean_lag_cells": float(np.mean(lag[settled])),
        "settled_lag_sd_cells": float(np.std(lag[settled], ddof=1)),
        "maximum_absolute_transverse_centroid_cells": float(
            np.max(np.abs(centroid_y))
        ),
        "behind_to_ahead_positive_depletion_ratio": behind_depletion
        / max(ahead_depletion, np.finfo(float).eps),
        "net_signed_deficit_error": float(np.sum(deficit)),
        "time_step": dt,
        "duration": WAKE_TIME,
        "samples": len(times),
    }
    profiles = {
        "x": xx,
        "y": yy,
        "deficit": deficit,
        "flux_x": flux_x,
        "flux_y": flux_y,
        "time": times_array,
        "core_x": cores,
        "centroid_x": centroids,
    }
    return result, profiles


def wake_experiment() -> tuple[dict, dict[str, np.ndarray]]:
    runs: dict[str, dict] = {}
    profiles: dict[str, np.ndarray] | None = None
    for speed in WAKE_SPEEDS:
        result, run_profiles = run_wake(speed, dt=0.10)
        runs[f"speed_{speed:.2f}"] = result
        if speed == 0.30:
            profiles = run_profiles

    refined, _ = run_wake(0.30, dt=0.05)
    baseline = runs["speed_0.30"]
    convergence = {
        "coarse_time_step": 0.10,
        "refined_time_step": 0.05,
        "pattern_velocity_relative_change": float(
            abs(
                refined["measured_pattern_velocity"]
                - baseline["measured_pattern_velocity"]
            )
            / abs(refined["measured_pattern_velocity"])
        ),
        "mean_lag_relative_change": float(
            abs(refined["settled_mean_lag_cells"] - baseline["settled_mean_lag_cells"])
            / abs(refined["settled_mean_lag_cells"])
        ),
    }
    assert profiles is not None
    return {
        "apparatus": {
            "dimensions": 2,
            "size": [WAKE_SIZE, WAKE_SIZE],
            "boundary": "periodic; trajectory remains more than 50 cells from boundary",
            "diffusivity_cells2_per_time": WAKE_DIFFUSIVITY,
            "relaxation_time": WAKE_RELAXATION_TIME,
            "core_sigma_cells": WAKE_CORE_SIGMA,
            "motion": "externally translated compensated equilibrium profile",
        },
        "runs": runs,
        "time_step_convergence": convergence,
    }, profiles


def flow_energy_diagnostic(radius2: np.ndarray, core_sigma: float, winding: int) -> float:
    """Regularized individual-phase kinetic energy, not a gravity coupling."""
    return float(
        np.sum(0.5 * winding**2 / (radius2 + core_sigma**2))
    )


def predictor_r_squared(predictor: np.ndarray, values: np.ndarray) -> float | None:
    design = np.column_stack((np.ones_like(predictor), predictor))
    predicted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    total = float(np.sum((values - np.mean(values)) ** 2))
    if total <= 1.0e-24:
        return None
    return 1.0 - float(np.sum((values - predicted) ** 2)) / total


def source_sweep() -> tuple[dict, dict[str, np.ndarray]]:
    grid = coordinate_grid(STATIC_SIZE, 3)
    radius2 = sum(axis**2 for axis in grid)
    radius = np.sqrt(radius2)
    core_sizes = np.asarray((1.8, 2.4, 3.0, 3.6))
    windings = np.asarray((0, 1, 2, 3))
    rows = []
    for core_sigma in core_sizes:
        deficit = compensated_deficit(radius2, float(core_sigma))
        shell = shell_measurements(deficit, radius)
        potential = spherical_potential(shell["charge"], shell["radius"])
        far = (
            (shell["radius"] >= STATIC_FIT_MIN)
            & (shell["radius"] <= STATIC_FIT_MAX)
        )
        void_volume = float(np.sum(np.clip(deficit, 0.0, None)))
        compact_response = float(np.max(np.abs(potential)))
        outer_tail_amplitude = float(np.max(np.abs(potential[far])))
        far_monopole_amplitude = abs(float(np.sum(deficit)))
        for winding in windings:
            rows.append(
                {
                    "core_sigma": float(core_sigma),
                    "winding": int(winding),
                    "positive_void_volume": void_volume,
                    "flow_energy_diagnostic": flow_energy_diagnostic(
                        radius2, float(core_sigma), int(winding)
                    ),
                    "compact_potential_amplitude": compact_response,
                    "far_monopole_amplitude": far_monopole_amplitude,
                    "outer_fit_tail_amplitude": outer_tail_amplitude,
                    "net_signed_deficit": float(np.sum(deficit)),
                }
            )

    void = np.asarray([row["positive_void_volume"] for row in rows])
    flow = np.asarray([row["flow_energy_diagnostic"] for row in rows])
    compact = np.asarray([row["compact_potential_amplitude"] for row in rows])
    far = np.asarray([row["far_monopole_amplitude"] for row in rows])
    fixed_core_spread = []
    for core_sigma in core_sizes:
        group = compact[np.isclose([row["core_sigma"] for row in rows], core_sigma)]
        fixed_core_spread.append(float(np.ptp(group)))

    result = {
        "design": {
            "core_sigma_cells": core_sizes.tolist(),
            "windings": windings.tolist(),
            "factorial_configurations": len(rows),
            "flow_energy_definition": (
                "regularized sum of individual-phase |grad theta|^2/2; diagnostic only"
            ),
        },
        "rows": rows,
        "compact_response_predictors": {
            "void_volume_r_squared": predictor_r_squared(void, compact),
            "flow_energy_r_squared": predictor_r_squared(flow, compact),
            "maximum_winding_spread_at_fixed_core": max(fixed_core_spread),
        },
        "far_field_predictors": {
            "void_volume_r_squared": predictor_r_squared(void, far),
            "flow_energy_r_squared": predictor_r_squared(flow, far),
            "interpretation": (
                "undefined/irrelevant when the conserved far monopole is zero; "
                "neither predictor is selected"
            ),
        },
        "external_coupling_controls": {
            "void_sourced_poisson_amplitude_definition": "positive void volume",
            "energy_sourced_poisson_amplitude_definition": "flow energy diagnostic",
            "status": (
                "alternative postulates; the present lattice contains no equation "
                "that chooses between them"
            ),
        },
    }
    profiles = {
        "void": void,
        "flow": flow,
        "compact": compact,
        "far": far,
        "winding": np.asarray([row["winding"] for row in rows]),
    }
    return result, profiles


def validate_result(result: dict) -> None:
    stationary = result["stationary_far_field"]
    wake = result["moving_defect_wake"]
    sweep = result["source_sweep"]
    baseline_wake = wake["runs"]["speed_0.30"]
    checks = {
        "A density conserved": abs(stationary["conservation"]["A_total_error"]) < 1.0e-8,
        "B density conserved": abs(stationary["conservation"]["B_total_error"]) < 1.0e-8,
        "compensated monopole cancelled": abs(
            stationary["conservation"]["net_deficit_over_positive_void_volume"]
        ) < 1.0e-12,
        "uncompensated control resolves inverse radius": (
            stationary["uncompensated_control"]["inverse_r_fit"]["relative_rmse"]
            < 1.0e-5
        ),
        "inverse radius beats inverse square in control": (
            stationary["uncompensated_control"]["inverse_r_fit"]["rmse"]
            < stationary["uncompensated_control"]["inverse_r2_fit"]["rmse"]
        ),
        "compensated far field suppressed": (
            stationary["far_signal_fraction_of_uncompensated_control"] < 1.0e-6
        ),
        "wake tracks imposed velocity": (
            abs(baseline_wake["pattern_to_core_velocity_ratio"] - 1.0) < 0.03
        ),
        "wake lags moving core": baseline_wake["settled_mean_lag_cells"] > 0.1,
        "wake time-step convergence": (
            wake["time_step_convergence"]["pattern_velocity_relative_change"] < 0.01
            and wake["time_step_convergence"]["mean_lag_relative_change"] < 0.03
        ),
        "winding leaves compact scarcity unchanged": (
            sweep["compact_response_predictors"][
                "maximum_winding_spread_at_fixed_core"
            ]
            < 1.0e-12
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    result["validation"] = {"checks": checks, "all_passed": not failed}
    if failed:
        raise RuntimeError("validation failed: " + "; ".join(failed))


def make_plot(
    stationary: dict[str, np.ndarray],
    wake: dict[str, np.ndarray],
    sweep: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    radius = stationary["radius"]
    valid = (radius >= 1.0) & (radius <= 45.0)
    axes[0, 0].semilogy(
        radius[valid],
        np.maximum(np.abs(stationary["compensated_potential"][valid]), 1.0e-14),
        color="#ff6b7a",
        label="conserved core + halo",
    )
    axes[0, 0].semilogy(
        radius[valid],
        stationary["control_potential"][valid],
        color="#59c3ff",
        label="uncompensated core control",
    )
    reference = stationary["control_potential"][valid][-1] * radius[valid][-1] / radius[valid]
    axes[0, 0].semilogy(radius[valid], reference, "--", color="#cccccc", label="1/r")
    axes[0, 0].axvspan(STATIC_FIT_MIN, STATIC_FIT_MAX, color="#888888", alpha=0.12)
    axes[0, 0].set(
        title="Stationary spherical scarcity estimator",
        xlabel="radius (cells)",
        ylabel="|potential-like accumulated deficit|",
    )

    image = axes[0, 1].imshow(
        wake["deficit"].T,
        origin="lower",
        extent=[
            float(wake["x"].min()),
            float(wake["x"].max()),
            float(wake["y"].min()),
            float(wake["y"].max()),
        ],
        cmap="coolwarm",
        vmin=-float(np.max(np.abs(wake["deficit"]))),
        vmax=float(np.max(np.abs(wake["deficit"]))),
    )
    flux_magnitude = np.hypot(wake["flux_x"], wake["flux_y"])
    visible_flux = flux_magnitude >= 0.02 * float(np.max(flux_magnitude))
    plot_flux_x = np.where(visible_flux, wake["flux_x"], np.nan)
    plot_flux_y = np.where(visible_flux, wake["flux_y"], np.nan)
    skip = (slice(None, None, 4), slice(None, None, 4))
    axes[0, 1].quiver(
        wake["x"][skip],
        wake["y"][skip],
        plot_flux_x[skip],
        plot_flux_y[skip],
        color="#202020",
        alpha=0.65,
        angles="xy",
        scale_units="xy",
        scale=0.01,
    )
    axes[0, 1].plot(wake["core_x"][-1], 0.0, "ko", ms=4, label="imposed core")
    axes[0, 1].set_xlim(wake["core_x"][-1] - 25, wake["core_x"][-1] + 25)
    axes[0, 1].set_ylim(-22, 22)
    axes[0, 1].set(title="Moving compensated core: final wake", xlabel="x", ylabel="y")
    fig.colorbar(image, ax=axes[0, 1], label="signed depletion")

    axes[1, 0].plot(wake["time"], wake["core_x"], color="#eeeeee", label="imposed core")
    axes[1, 0].plot(
        wake["time"], wake["centroid_x"], color="#74d99f", label="depletion centroid"
    )
    axes[1, 0].set(
        title="Wake velocity and lag (v = 0.30)",
        xlabel="time",
        ylabel="x position (cells)",
    )

    # Small horizontal display offset exposes the four exactly overlapping
    # winding cases without changing the measured values in results.json.
    display_void = sweep["void"] * (1.0 + 0.015 * (sweep["winding"] - 1.5))
    scatter = axes[1, 1].scatter(
        display_void,
        sweep["compact"],
        c=sweep["flow"],
        cmap="viridis",
        s=55,
        edgecolor="#333333",
        linewidth=0.4,
    )
    axes[1, 1].set(
        title="Compact response: core size vs winding",
        xlabel="positive void volume",
        ylabel="compact potential amplitude",
    )
    fig.colorbar(scatter, ax=axes[1, 1], label="flow-energy diagnostic")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
        if axis is not axes[1, 1]:
            axis.legend(fontsize=8)
    fig.suptitle("Two-substance defect gravity bridge")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("assets")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stationary_result, stationary_profiles = stationary_experiment()
    wake_result, wake_profiles = wake_experiment()
    sweep_result, sweep_profiles = source_sweep()
    result = {
        "schema_version": 1,
        "testable_question": (
            "Does the conserved two-substance defect lattice itself generate a 1/r "
            "scarcity far field, a velocity-tracking wake, and a unique void- or "
            "flow-energy gravitational charge?"
        ),
        "stationary_far_field": stationary_result,
        "moving_defect_wake": wake_result,
        "source_sweep": sweep_result,
        "conclusions": {
            "far_field": (
                "The compensated void has zero monopole and no resolved far field; "
                "it is neither 1/r nor 1/r^2. The uncompensated control is 1/r."
            ),
            "wake": (
                "A forced moving equilibrium core produces a lagging depletion and "
                "replenishment-flow wake whose pattern velocity tracks the imposed core."
            ),
            "which_quantity_gravitates": (
                "Neither is selected. Core size controls the compact density response "
                "and winding changes flow energy independently, but conservation leaves "
                "the far monopole zero and the lattice defines no gravity-source coupling."
            ),
            "model_boundary": (
                "The defect trajectory and relaxation target are imposed, and the "
                "flow-energy measure is diagnostic. A nonconserving sink or explicit "
                "energy-to-scarcity equation would be new microphysics, not a measured "
                "consequence of this lattice."
            ),
        },
    }
    validate_result(result)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    make_plot(
        stationary_profiles,
        wake_profiles,
        sweep_profiles,
        args.output_dir / "gravity-bridge.png",
    )

    stationary = result["stationary_far_field"]
    wake = result["moving_defect_wake"]["runs"]["speed_0.30"]
    print(
        "stationary compensated/control far-signal ratio: "
        f"{stationary['far_signal_fraction_of_uncompensated_control']:.3e}"
    )
    print(
        "uncompensated 1/r relative RMSE: "
        f"{stationary['uncompensated_control']['inverse_r_fit']['relative_rmse']:.3e}"
    )
    print(
        f"wake velocity ratio: {wake['pattern_to_core_velocity_ratio']:.6f}; "
        f"lag {wake['settled_mean_lag_cells']:.3f} cells"
    )
    print("gravitating quantity selected by lattice: neither")


if __name__ == "__main__":
    main()
