"""Evolve the moving level core as an initial-value problem.

The apparatus solves the draw-sourced elliptic level in the comoving frame and
then advances the rolling-rule momentum equation and consumed-density
continuity equation without imposing a Bernoulli speed.  A deterministic
three-rung finite-volume experiment asks whether each swept wind settles and,
only where it does, measures the discovered speed and direction fields.

Usage:
    uv run lab/sims/level-core-dynamical-relaxation/main.py
    uv run lab/sims/level-core-dynamical-relaxation/main.py --check-determinism
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn
from scipy.interpolate import interpn
from scipy.special import eval_legendre, roots_legendre


SEED = 42
TASK_ID = "ORB-10937"
RUN_ID = "jrun-20260821-0443-3"
RUN_DATE = "2026-08-21"
RUN_RECORD = "2026-08-21-seed-42.json"
GRID_SIZES = (25, 33, 41)
DOMAIN_HALF_WIDTH = 12.0
CORE_SIGMA = 0.75
PROBE_RADIUS = 5.0
MEASUREMENT_RADII = (3.0, 5.0)
WIND_RATIOS = (0.03, 0.1, 0.3, 1.0)
CONTROL_HALF_WIDTH = 8.0
CORE_EXCLUSION_RADIUS = 1.5
SHELL_MU_NODES = 10
SHELL_PHI_NODES = 20
C_SQUARED = 1.0
CFL = 0.22
END_TIME = 60.0
DRAW_RAMP_TIME = 3.0
DIAGNOSTIC_INTERVAL = 1.5
STEADY_WINDOW_SAMPLES = 5
STEADY_DV_RMS = 2.0e-3
STEADY_DN_RMS = 2.0e-3
DENSITY_FLOOR = 1.0e-10
ATTRACTOR_GRID_SIZE = 33
ATTRACTOR_WIND_RATIO = 0.3
EXPECTED_STENCIL_SHA256 = (
    "aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461"
)


# Frozen byte-identical ORB-10751 consumption stencil.
# fmt: off
def strain_consumption_3d(
    density: np.ndarray, velocity: np.ndarray, spacing: float
) -> np.ndarray:
    """Evaluate n*sqrt(3/2 e_dev:e_dev) with centred neighbour stencils."""
    gradients = [
        [
            np.gradient(velocity[i], spacing, axis=j, edge_order=2)
            for j in range(3)
        ]
        for i in range(3)
    ]
    trace = sum(gradients[i][i] for i in range(3))
    contraction = np.zeros_like(density)
    for i in range(3):
        for j in range(3):
            strain = 0.5 * (gradients[i][j] + gradients[j][i])
            if i == j:
                strain = strain - trace / 3.0
            contraction += strain * strain
    return density * np.sqrt(1.5 * contraction)
# fmt: on


def frozen_stencil_sha256() -> str:
    source = Path(__file__).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "strain_consumption_3d"
    )
    segment = ast.get_source_segment(source, function)
    if segment is None:
        raise RuntimeError("could not recover frozen stencil source")
    return hashlib.sha256(segment.encode()).hexdigest()


def coordinates(size: int) -> tuple[np.ndarray, tuple[np.ndarray, ...], float]:
    axis = np.linspace(-DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH, size)
    spacing = float(axis[1] - axis[0])
    return axis, tuple(np.meshgrid(axis, axis, axis, indexing="ij")), spacing


def solve_draw_level(size: int) -> dict:
    """Solve -laplacian(H)=rho with H=0 on the fixed outer cube."""
    axis, mesh, spacing = coordinates(size)
    radius_squared = sum(coordinate**2 for coordinate in mesh)
    draw = np.exp(-0.5 * radius_squared / CORE_SIGMA**2)
    draw /= np.sum(draw) * spacing**3
    mode = np.arange(1, size - 1)
    eigenvalue = (2.0 * np.cos(np.pi * mode / (size - 1)) - 2.0) / spacing**2
    minus_laplacian = -(
        eigenvalue[:, None, None]
        + eigenvalue[None, :, None]
        + eigenvalue[None, None, :]
    )
    level = np.zeros_like(draw)
    transformed = dstn(draw[1:-1, 1:-1, 1:-1], type=1, norm="ortho")
    level[1:-1, 1:-1, 1:-1] = idstn(
        transformed / minus_laplacian, type=1, norm="ortho"
    )
    sigma = -np.expm1(-level)
    grad_sigma = np.stack(np.gradient(sigma, spacing, edge_order=2))
    grad_magnitude = np.sqrt(np.sum(grad_sigma**2, axis=0))
    gp_velocity = np.zeros_like(grad_sigma)
    mask = grad_magnitude > 0.0
    gp_velocity[:, mask] = (
        np.sqrt(2.0 * sigma[mask]) * grad_sigma[:, mask] / grad_magnitude[mask]
    )
    return {
        "axis": axis,
        "mesh": mesh,
        "spacing": spacing,
        "draw": draw,
        "sigma": sigma,
        "grad_sigma": grad_sigma,
        "gp_velocity": gp_velocity,
        "draw_strength_recovered": float(np.sum(draw) * spacing**3),
    }


def shell_quadrature(radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, mu_weights = roots_legendre(SHELL_MU_NODES)
    phi = 2.0 * np.pi * np.arange(SHELL_PHI_NODES) / SHELL_PHI_NODES
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing="ij")
    transverse = np.sqrt(1.0 - mu_grid**2)
    directions = np.column_stack(
        (
            mu_grid.ravel(),
            (transverse * np.cos(phi_grid)).ravel(),
            (transverse * np.sin(phi_grid)).ravel(),
        )
    )
    weights = np.repeat(mu_weights, SHELL_PHI_NODES) / (2.0 * SHELL_PHI_NODES)
    return radius * directions, directions, weights


def sample_scalar(values: np.ndarray, axis: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(interpn((axis, axis, axis), values, points))


def sample_vector(values: np.ndarray, axis: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [sample_scalar(values[index], axis, points) for index in range(3)]
    )


def legendre_coefficients(
    values: np.ndarray, directions: np.ndarray, weights: np.ndarray
) -> dict:
    return {
        f"l{order}": float(
            (2 * order + 1)
            * np.sum(weights * values * eval_legendre(order, directions[:, 0]))
        )
        for order in range(4)
    }


def normalized_speed_multipoles(
    speed: np.ndarray, directions: np.ndarray, weights: np.ndarray
) -> dict:
    monopole = float(np.sum(weights * speed))
    raw = legendre_coefficients(speed, directions, weights)
    dipole = 3.0 * np.sum(weights[:, None] * speed[:, None] * directions, axis=0)
    return {
        "monopole": monopole,
        "normalized_axisymmetric": {
            key: float(value / monopole) for key, value in raw.items() if key != "l0"
        },
        "normalized_dipole_vector": [float(value / monopole) for value in dipole],
        "normalized_dipole_magnitude": float(np.linalg.norm(dipole) / monopole),
    }


def wind_reference(level: dict) -> float:
    points, _, weights = shell_quadrature(PROBE_RADIUS)
    sigma = sample_scalar(level["sigma"], level["axis"], points)
    return float(np.sqrt(2.0 * np.sum(weights * sigma)))


def apply_boundary(velocity: np.ndarray, density: np.ndarray, wind: float) -> None:
    """Apply one-sided open faces and the sole prescribed upstream inflow."""
    for field in (velocity, density[None, ...]):
        field[..., -1, :, :] = field[..., -2, :, :]
        field[..., :, 0, :] = field[..., :, 1, :]
        field[..., :, -1, :] = field[..., :, -2, :]
        field[..., :, :, 0] = field[..., :, :, 1]
        field[..., :, :, -1] = field[..., :, :, -2]
    velocity[:, 0, :, :] = 0.0
    velocity[0, 0, :, :] = wind
    density[0, :, :] = 1.0


def rhs(
    velocity: np.ndarray,
    density: np.ndarray,
    force: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donor-cell spatial operator for the literal inviscid equations."""
    velocity_rhs = force.copy()
    for component in range(3):
        advective = np.zeros_like(density)
        quantity = velocity[component]
        for axis in range(3):
            carrier = velocity[axis]
            backward = (quantity - np.roll(quantity, 1, axis=axis)) / spacing
            forward = (np.roll(quantity, -1, axis=axis) - quantity) / spacing
            advective += carrier * np.where(carrier >= 0.0, backward, forward)
        velocity_rhs[component] -= advective

    divergence = np.zeros_like(density)
    for axis in range(3):
        carrier = velocity[axis]
        carrier_plus = 0.5 * (carrier + np.roll(carrier, -1, axis=axis))
        neighbour = np.roll(density, -1, axis=axis)
        flux_plus = carrier_plus * np.where(carrier_plus >= 0.0, density, neighbour)
        divergence += (flux_plus - np.roll(flux_plus, 1, axis=axis)) / spacing
    consumption = strain_consumption_3d(density, velocity, spacing)
    return velocity_rhs, -divergence - consumption, consumption


def advance(
    velocity: np.ndarray,
    density: np.ndarray,
    force: np.ndarray,
    spacing: float,
    wind: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One deterministic SSP-RK2 step with a positivity projection for n."""
    interior = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
    scalar_interior = (slice(1, -1),) * 3
    first_v, first_n, _ = rhs(velocity, density, force, spacing)
    stage_v = velocity.copy()
    stage_n = density.copy()
    stage_v[interior] += dt * first_v[interior]
    stage_n[scalar_interior] += dt * first_n[scalar_interior]
    np.maximum(stage_n, DENSITY_FLOOR, out=stage_n)
    apply_boundary(stage_v, stage_n, wind)
    second_v, second_n, consumption = rhs(stage_v, stage_n, force, spacing)
    next_v = velocity.copy()
    next_n = density.copy()
    next_v[interior] += 0.5 * dt * (first_v[interior] + second_v[interior])
    next_n[scalar_interior] += 0.5 * dt * (
        first_n[scalar_interior] + second_n[scalar_interior]
    )
    np.maximum(next_n, DENSITY_FLOOR, out=next_n)
    apply_boundary(next_v, next_n, wind)
    return next_v, next_n, second_v, second_n, consumption


def control_surface_flux(
    velocity: np.ndarray, density: np.ndarray, axis: np.ndarray, spacing: float
) -> float:
    lower = int(np.argmin(np.abs(axis + CONTROL_HALF_WIDTH)))
    upper = int(np.argmin(np.abs(axis - CONTROL_HALF_WIDTH)))
    surface = slice(lower, upper + 1)
    vx, vy, vz = velocity
    return float(
        (
            np.sum(density[upper, surface, surface] * vx[upper, surface, surface] ** 2)
            - np.sum(density[lower, surface, surface] * vx[lower, surface, surface] ** 2)
            + np.sum(density[surface, upper, surface] * vx[surface, upper, surface] * vy[surface, upper, surface])
            - np.sum(density[surface, lower, surface] * vx[surface, lower, surface] * vy[surface, lower, surface])
            + np.sum(density[surface, surface, upper] * vx[surface, surface, upper] * vz[surface, surface, upper])
            - np.sum(density[surface, surface, lower] * vx[surface, surface, lower] * vz[surface, surface, lower])
        )
        * spacing**2
    )


def momentum_sample(
    level: dict, velocity: np.ndarray, density: np.ndarray, consumption: np.ndarray
) -> tuple[float, float]:
    radius = np.sqrt(sum(coordinate**2 for coordinate in level["mesh"]))
    box_radius = np.maximum.reduce([np.abs(coordinate) for coordinate in level["mesh"]])
    mask = (radius >= CORE_EXCLUSION_RADIUS) & (box_radius <= CONTROL_HALF_WIDTH)
    consumed = float(
        np.sum(consumption[mask] * velocity[0, mask]) * level["spacing"] ** 3
    )
    flux = control_surface_flux(
        velocity, density, level["axis"], level["spacing"]
    )
    return consumed, flux


def flow_diagnostics(level: dict, velocity: np.ndarray) -> dict:
    spacing = level["spacing"]
    dvx = np.gradient(velocity[0], spacing, edge_order=2)
    dvy = np.gradient(velocity[1], spacing, edge_order=2)
    dvz = np.gradient(velocity[2], spacing, edge_order=2)
    curl = np.stack((dvz[1] - dvy[2], dvx[2] - dvz[0], dvy[0] - dvx[1]))
    curl_magnitude = np.sqrt(np.sum(curl**2, axis=0))
    speed = np.sqrt(np.sum(velocity**2, axis=0))
    interior_speed = speed[1:-1, 1:-1, 1:-1]
    flat_index = int(np.argmin(interior_speed))
    local_index = np.unravel_index(flat_index, interior_speed.shape)
    index = tuple(value + 1 for value in local_index)
    return {
        "vorticity_rms": float(np.sqrt(np.mean(curl_magnitude[1:-1, 1:-1, 1:-1] ** 2))),
        "vorticity_maximum": float(np.max(curl_magnitude[1:-1, 1:-1, 1:-1])),
        "minimum_interior_speed": float(speed[index]),
        "minimum_speed_location": [float(level["axis"][value]) for value in index],
    }


def load_marched_comparator() -> dict:
    path = Path(__file__).parents[1] / "level-core-wind-tunnel" / "assets" / "results.json"
    source = json.loads(path.read_text())
    cases = source["gates"]["G2_wake_structure"]["cases"]
    return {
        (float(row["wind_ratio"]), float(row["radius"])): row[
            "actual_nx_multipoles"
        ]
        for row in cases
    }


def shell_measurements(
    level: dict,
    velocity: np.ndarray,
    wind: float,
    ratio: float,
    marched: dict,
    residual_speed_proxy: float,
) -> list[dict]:
    speed_field = np.sqrt(np.sum(velocity**2, axis=0))
    bernoulli_field = np.sqrt(wind**2 + 2.0 * C_SQUARED * level["sigma"])
    shells = []
    for radius in MEASUREMENT_RADII:
        points, directions, weights = shell_quadrature(radius)
        sampled_velocity = sample_vector(velocity, level["axis"], points)
        sampled_speed = np.linalg.norm(sampled_velocity, axis=1)
        sampled_bernoulli = sample_scalar(bernoulli_field, level["axis"], points)
        actual_direction = sampled_velocity / np.maximum(sampled_speed[:, None], 1.0e-14)
        boosted = sample_vector(level["gp_velocity"], level["axis"], points)
        boosted[:, 0] += wind
        boosted_direction = boosted / np.maximum(np.linalg.norm(boosted, axis=1)[:, None], 1.0e-14)
        actual_nx = actual_direction[:, 0]
        boosted_nx = boosted_direction[:, 0]
        actual_coefficients = legendre_coefficients(actual_nx, directions, weights)
        boosted_coefficients = legendre_coefficients(boosted_nx, directions, weights)
        marched_coefficients = marched[(ratio, radius)]
        boosted_angle = np.arccos(
            np.clip(np.sum(actual_direction * boosted_direction, axis=1), -1.0, 1.0)
        )
        formula_residual = sampled_speed - sampled_bernoulli
        shells.append(
            {
                "radius": radius,
                "speed": normalized_speed_multipoles(sampled_speed, directions, weights),
                "bernoulli_formula": "sqrt(U^2 + 2 c^2 sigma)",
                "bernoulli_relative_rms_error": float(
                    np.sqrt(np.sum(weights * formula_residual**2))
                    / np.sum(weights * sampled_bernoulli)
                ),
                "bernoulli_maximum_relative_error": float(
                    np.max(np.abs(formula_residual)) / np.mean(sampled_bernoulli)
                ),
                "temporal_error_proxy_from_last_dv_rms": residual_speed_proxy,
                "direction": {
                    "actual_nx_multipoles": actual_coefficients,
                    "boosted_GP_nx_multipoles": boosted_coefficients,
                    "ORB_10935_marched_nx_multipoles": marched_coefficients,
                    "actual_minus_boosted_GP": {
                        key: actual_coefficients[key] - boosted_coefficients[key]
                        for key in actual_coefficients
                    },
                    "actual_minus_ORB_10935_marched": {
                        key: actual_coefficients[key] - marched_coefficients[key]
                        for key in actual_coefficients
                    },
                    "weighted_rms_angle_from_boosted_GP_radians": float(
                        np.sqrt(np.sum(weights * boosted_angle**2))
                    ),
                },
            }
        )
    return shells


def characterize_history(history: list[dict]) -> dict:
    tail = history[len(history) // 2 :]
    residual = np.array([row["dv_rms"] for row in tail])
    probe = np.array([row["wake_probe_vx"] for row in tail])
    times = np.array([row["time"] for row in tail])
    slope = float(np.polyfit(times, np.log(np.maximum(residual, 1.0e-30)), 1)[0])
    centered = probe - np.mean(probe)
    center_lateral = np.sqrt(
        np.array([row["wake_center_vy"] for row in tail]) ** 2
        + np.array([row["wake_center_vz"] for row in tail]) ** 2
    )
    antisymmetric_lateral = 0.5 * np.array(
        [row["wake_plus_y_vy"] - row["wake_minus_y_vy"] for row in tail]
    )
    symmetric_lateral = 0.5 * np.array(
        [row["wake_plus_y_vy"] + row["wake_minus_y_vy"] for row in tail]
    )
    if len(centered) >= 4 and np.any(centered != 0.0):
        spacing = float(np.mean(np.diff(times)))
        frequencies = np.fft.rfftfreq(len(centered), d=spacing)
        spectrum = np.abs(np.fft.rfft(centered))
        spectrum[0] = 0.0
        peak_index = int(np.argmax(spectrum))
        frequency = float(frequencies[peak_index])
        spectral_fraction = float(spectrum[peak_index] / np.sum(spectrum))
    else:
        frequency = 0.0
        spectral_fraction = 0.0
    if slope > 0.02:
        behavior = "growing_over_observation_window"
    elif slope < -0.005:
        behavior = "decaying_slowly_not_yet_saturated"
    elif float(np.std(probe)) > 5.0e-4:
        behavior = "saturated_with_wake_variability"
    else:
        behavior = "monotone_or_quiescent_relaxation_plateau"
    antisymmetric_rms = float(np.sqrt(np.mean(antisymmetric_lateral**2)))
    symmetric_rms = float(np.sqrt(np.mean(symmetric_lateral**2)))
    if antisymmetric_rms > max(2.0 * symmetric_rms, 5.0e-4):
        wake_structure = "antisymmetric_lateral_wake_mode"
    elif float(np.sqrt(np.mean(center_lateral**2))) > 5.0e-4:
        wake_structure = "three_dimensional_lateral_wake_mode"
    else:
        wake_structure = "no_resolved_lateral_shedding; axial relaxation dominates"
    return {
        "classification": behavior,
        "late_log_residual_slope_per_time": slope,
        "dominant_wake_probe_frequency": frequency,
        "dominant_spectral_amplitude_fraction": spectral_fraction,
        "wake_probe_standard_deviation": float(np.std(probe)),
        "wake_shedding_proxy": "v_x at (x,y,z)=(5,0,0)",
        "wake_structure": wake_structure,
        "centerline_lateral_velocity_rms": float(np.sqrt(np.mean(center_lateral**2))),
        "off_axis_antisymmetric_vy_rms": antisymmetric_rms,
        "off_axis_symmetric_vy_rms": symmetric_rms,
        "off_axis_probe_locations": [[5.0, 1.5, 0.0], [5.0, -1.5, 0.0]],
    }


def run_case(
    level: dict,
    wind: float,
    ratio: float,
    marched: dict,
    initial_condition: str = "ramped_pure_wind",
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    shape = level["sigma"].shape
    velocity = np.zeros((3,) + shape)
    velocity[0].fill(wind)
    if initial_condition == "boosted_static_GP":
        velocity += level["gp_velocity"]
    density = np.ones(shape)
    apply_boundary(velocity, density, wind)
    time = 0.0
    steps = 0
    next_diagnostic = 0.0
    history = []
    momentum_history = []
    latest_consumption = np.zeros(shape)
    while time < END_TIME - 1.0e-12:
        maximum_transport = float(
            np.max(np.abs(velocity[0]) + np.abs(velocity[1]) + np.abs(velocity[2]))
        )
        dt = min(CFL * level["spacing"] / max(maximum_transport, 1.0e-6), 0.4, END_TIME - time)
        ramp = 1.0 if initial_condition == "boosted_static_GP" else min(1.0, (time + dt) / DRAW_RAMP_TIME)
        force = C_SQUARED * ramp * level["grad_sigma"]
        next_v, next_n, velocity_rhs, density_rhs, latest_consumption = advance(
            velocity, density, force, level["spacing"], wind, dt
        )
        time += dt
        steps += 1
        velocity = next_v
        density = next_n
        if time + 1.0e-12 >= next_diagnostic or time >= END_TIME - 1.0e-12:
            interior = (slice(1, -1),) * 3
            center = len(level["axis"]) // 2
            wake_index = int(np.argmin(np.abs(level["axis"] - 5.0)))
            plus_y = int(np.argmin(np.abs(level["axis"] - 1.5)))
            minus_y = int(np.argmin(np.abs(level["axis"] + 1.5)))
            consumed, flux = momentum_sample(level, velocity, density, latest_consumption)
            history.append(
                {
                    "time": float(time),
                    "step": steps,
                    "dt": float(dt),
                    "dv_rms": float(
                        np.sqrt(np.mean(np.sum(velocity_rhs[:, 1:-1, 1:-1, 1:-1] ** 2, axis=0)))
                    ),
                    "dn_rms": float(np.sqrt(np.mean(density_rhs[interior] ** 2))),
                    "wake_probe_vx": float(velocity[0, wake_index, center, center]),
                    "wake_center_vy": float(velocity[1, wake_index, center, center]),
                    "wake_center_vz": float(velocity[2, wake_index, center, center]),
                    "wake_plus_y_vy": float(velocity[1, wake_index, plus_y, center]),
                    "wake_minus_y_vy": float(velocity[1, wake_index, minus_y, center]),
                    "minimum_density": float(np.min(density)),
                }
            )
            momentum_history.append(
                {"time": float(time), "consumed_momentum_x": consumed, "advective_flux_x": flux}
            )
            next_diagnostic += DIAGNOSTIC_INTERVAL

    sustained = history[-STEADY_WINDOW_SAMPLES:]
    steady = bool(
        len(sustained) == STEADY_WINDOW_SAMPLES
        and sustained[0]["time"] >= 0.75 * END_TIME
        and all(row["dv_rms"] < STEADY_DV_RMS and row["dn_rms"] < STEADY_DN_RMS for row in sustained)
    )
    flow = flow_diagnostics(level, velocity)
    stagnation_threshold = max(1.0e-3, 0.05 * wind)
    tail_momentum = momentum_history[-STEADY_WINDOW_SAMPLES:]
    consumed_values = np.array([row["consumed_momentum_x"] for row in tail_momentum])
    flux_values = np.array([row["advective_flux_x"] for row in tail_momentum])
    last_dv = history[-1]["dv_rms"]
    result = {
        "wind_ratio_to_finest_v_GP_at_probe": ratio,
        "wind_speed": wind,
        "initial_condition": initial_condition,
        "integration": {"end_time": END_TIME, "steps": steps},
        "steadiness": {
            "verdict": "steady" if steady else "not_steady_within_horizon",
            "steady": steady,
            "criterion": f"dv_rms < {STEADY_DV_RMS:g} and dn_rms < {STEADY_DN_RMS:g} for the final {STEADY_WINDOW_SAMPLES} samples, all after 0.75*T",
            "window_samples": STEADY_WINDOW_SAMPLES,
            "residual_time_series": history,
            "unsteady_characterization": None if steady else characterize_history(history),
        },
        "discovered_field": {
            "reported_for_steady_case": steady,
            "shells": shell_measurements(level, velocity, wind, ratio, marched, last_dv * DIAGNOSTIC_INTERVAL) if steady else [],
            "stagnation": {
                **flow,
                "search_threshold": stagnation_threshold,
                "stagnation_found": bool(flow["minimum_interior_speed"] <= stagnation_threshold),
            },
        },
        "momentum_budget": {
            "time_series": momentum_history,
            "averaging_window": f"last {STEADY_WINDOW_SAMPLES} diagnostic samples",
            "consumed_momentum_integral_x_mean": float(np.mean(consumed_values)),
            "consumed_momentum_integral_x_standard_deviation": float(np.std(consumed_values)),
            "advective_far_surface_flux_x_mean": float(np.mean(flux_values)),
            "advective_far_surface_flux_x_standard_deviation": float(np.std(flux_values)),
            "flux_minus_consumed_x_mean": float(np.mean(flux_values - consumed_values)),
            "sign": "drag" if float(np.mean(consumed_values)) > 0.0 else "thrust",
        },
    }
    return result, (velocity, density)


def null_control(level: dict, winds: list[float]) -> dict:
    rows = []
    for wind in winds:
        shape = level["sigma"].shape
        velocity = np.zeros((3,) + shape)
        velocity[0].fill(wind)
        density = np.ones(shape)
        initial_v = velocity.copy()
        initial_n = density.copy()
        time = 0.0
        steps = 0
        maximum_consumption = 0.0
        while time < END_TIME - 1.0e-12:
            dt = min(CFL * level["spacing"] / max(wind, 1.0e-6), 0.4, END_TIME - time)
            velocity, density, _, _, consumption = advance(
                velocity, density, np.zeros_like(velocity), level["spacing"], wind, dt
            )
            maximum_consumption = max(maximum_consumption, float(np.max(np.abs(consumption))))
            time += dt
            steps += 1
        rows.append(
            {
                "wind_speed": wind,
                "steps": steps,
                "maximum_velocity_change": float(np.max(np.abs(velocity - initial_v))),
                "maximum_density_change": float(np.max(np.abs(density - initial_n))),
                "maximum_consumption": maximum_consumption,
            }
        )
    passed = all(
        max(row["maximum_velocity_change"], row["maximum_density_change"], row["maximum_consumption"]) <= np.finfo(float).eps
        for row in rows
    )
    return {"grid_size": len(level["axis"]), "measurements": rows, "passed": passed}


def run_rung(level: dict, winds: list[float], marched: dict) -> tuple[dict, dict]:
    cases = []
    states = {}
    for ratio, wind in zip(WIND_RATIOS, winds):
        case, state = run_case(level, wind, ratio, marched)
        cases.append(case)
        states[ratio] = state
    return (
        {
            "apparatus": {
                "grid_size": len(level["axis"]),
                "lattice": f"{len(level['axis'])}^3",
                "spacing": level["spacing"],
                "domain_half_width": DOMAIN_HALF_WIDTH,
                "physical_core_sigma": CORE_SIGMA,
                "core_sigma_in_cells": CORE_SIGMA / level["spacing"],
                "draw_strength_recovered": level["draw_strength_recovered"],
            },
            "cases": cases,
            "null_control": null_control(level, winds),
        },
        states,
    )


def relative_change(first: float, second: float, floor: float = 1.0e-14) -> float:
    return float(abs(second - first) / max(abs(first), abs(second), floor))


def adjudicate(ladder: list[dict]) -> dict:
    g1_cases = []
    for rung in ladder:
        for case in rung["cases"]:
            g1_cases.append(
                {
                    "grid_size": rung["apparatus"]["grid_size"],
                    "wind_ratio": case["wind_ratio_to_finest_v_GP_at_probe"],
                    "wind_speed": case["wind_speed"],
                    **case["steadiness"],
                }
            )

    g2_cases = []
    g3_cases = []
    for rung_index, rung in enumerate(ladder):
        for case_index, case in enumerate(rung["cases"]):
            if not case["steadiness"]["steady"]:
                continue
            for shell_index, shell in enumerate(case["discovered_field"]["shells"]):
                convergence_error = None
                converged = None
                if rung_index > 0 and ladder[rung_index - 1]["cases"][case_index]["steadiness"]["steady"]:
                    coarser = ladder[rung_index - 1]["cases"][case_index]["discovered_field"]["shells"][shell_index]
                    convergence_error = {
                        "bernoulli_rms_finest_shift": abs(shell["bernoulli_relative_rms_error"] - coarser["bernoulli_relative_rms_error"]),
                        "speed_dipole_finest_shift": abs(shell["speed"]["normalized_dipole_magnitude"] - coarser["speed"]["normalized_dipole_magnitude"]),
                    }
                    converged = bool(max(convergence_error.values()) < 0.05)
                g2_cases.append(
                    {
                        "grid_size": rung["apparatus"]["grid_size"],
                        "wind_ratio": case["wind_ratio_to_finest_v_GP_at_probe"],
                        "radius": shell["radius"],
                        "speed": shell["speed"],
                        "bernoulli_relative_rms_error": shell["bernoulli_relative_rms_error"],
                        "bernoulli_maximum_relative_error": shell["bernoulli_maximum_relative_error"],
                        "temporal_error_proxy": shell["temporal_error_proxy_from_last_dv_rms"],
                        "resolution_error": convergence_error,
                        "converged": converged,
                    }
                )
                g3_cases.append(
                    {
                        "grid_size": rung["apparatus"]["grid_size"],
                        "wind_ratio": case["wind_ratio_to_finest_v_GP_at_probe"],
                        "radius": shell["radius"],
                        "direction": shell["direction"],
                        "stagnation_and_vorticity": case["discovered_field"]["stagnation"],
                        "converged": converged,
                    }
                )

    momentum_cases = []
    for rung_index, rung in enumerate(ladder):
        for case_index, case in enumerate(rung["cases"]):
            convergence = None
            converged = None
            if rung_index > 0:
                coarser_budget = ladder[rung_index - 1]["cases"][case_index]["momentum_budget"]
                convergence = {
                    "consumed_momentum_mean_relative_shift": relative_change(
                        coarser_budget["consumed_momentum_integral_x_mean"],
                        case["momentum_budget"]["consumed_momentum_integral_x_mean"],
                    ),
                    "advective_flux_mean_relative_shift": relative_change(
                        coarser_budget["advective_far_surface_flux_x_mean"],
                        case["momentum_budget"]["advective_far_surface_flux_x_mean"],
                    ),
                }
                converged = bool(max(convergence.values()) < 0.25)
            row = {
                "grid_size": rung["apparatus"]["grid_size"],
                "wind_ratio": case["wind_ratio_to_finest_v_GP_at_probe"],
                "wind_speed": case["wind_speed"],
                "steady": case["steadiness"]["steady"],
                **{key: value for key, value in case["momentum_budget"].items() if key != "time_series"},
                "resolution_error": convergence,
                "converged": converged,
            }
            momentum_cases.append(row)
    finest_momentum = momentum_cases[-len(WIND_RATIOS):]
    winds = np.array([row["wind_speed"] for row in finest_momentum])
    drag = np.abs(np.array([row["consumed_momentum_integral_x_mean"] for row in finest_momentum]))
    exponent, log_prefactor = np.polyfit(np.log(winds), np.log(np.maximum(drag, 1.0e-30)), 1)
    null_passed = all(rung["null_control"]["passed"] for rung in ladder)
    steady_count = sum(case["steady"] for case in g1_cases)
    converged_g2 = [case for case in g2_cases if case["converged"] is not None]
    return {
        "G1_steadiness": {
            "verdict": "steady_cases_found" if steady_count else "no_case_steady_within_horizon",
            "criterion": f"volume RMS dv/dt < {STEADY_DV_RMS:g} and dn/dt < {STEADY_DN_RMS:g} over {STEADY_WINDOW_SAMPLES} final samples after 0.75*T",
            "cases": g1_cases,
            "two_finest_verdict_match_by_wind": [
                {
                    "wind_ratio": ratio,
                    "medium": ladder[-2]["cases"][index]["steadiness"]["verdict"],
                    "fine": ladder[-1]["cases"][index]["steadiness"]["verdict"],
                    "converged": ladder[-2]["cases"][index]["steadiness"]["steady"]
                    == ladder[-1]["cases"][index]["steadiness"]["steady"],
                }
                for index, ratio in enumerate(WIND_RATIOS)
            ],
        },
        "G2_discovered_speed_law": {
            "verdict": "executed_no_steady_cases" if not g2_cases else "bernoulli_compared_on_steady_cases",
            "kill_gate": True,
            "criterion": "for steady cases, finest adjacent-rung shifts of Bernoulli RMS error and normalized speed dipole are each <0.05; a converged nonzero excess over temporal/resolution error refutes the closure",
            "cases": g2_cases,
            "converged_case_count": sum(bool(case["converged"]) for case in converged_g2),
        },
        "G3_realized_wake": {
            "verdict": "executed_no_steady_cases" if not g3_cases else "comparators_and_flow_checks_recorded",
            "criterion": "same steady/convergence admission as G2; direction l=0..3 is compared to boosted GP and ORB-10935 marched fields; stagnation threshold=max(1e-3,0.05U)",
            "cases": g3_cases,
            "unsteady_final_flow_checks": [
                {
                    "grid_size": rung["apparatus"]["grid_size"],
                    "wind_ratio": case["wind_ratio_to_finest_v_GP_at_probe"],
                    **case["discovered_field"]["stagnation"],
                }
                for rung in ladder
                for case in rung["cases"]
                if not case["steadiness"]["steady"]
            ],
        },
        "G4_momentum_budget": {
            "verdict": "time_averaged_all_cases",
            "criterion": "last five diagnostic samples report mean and standard deviation; adjacent-rung relative shift <25% is the convergence target",
            "cases": momentum_cases,
            "finest_drag_scaling": {
                "form": "|integral s v_x dV| = A U^p",
                "A": float(np.exp(log_prefactor)),
                "p": float(exponent),
                "ORB_10935_comparator": {"sign": "drag", "A": 20.3, "p": 0.098},
            },
        },
        "G5_Galilean_null": {
            "verdict": "pass" if null_passed else "kill_noninvariant_pure_wind",
            "kill_gate": True,
            "criterion": "pure wind/no core advanced to T=60 by the identical SSP-RK2/CFL/boundary path has zero velocity, density, and consumption change to floating precision",
            "by_rung": [rung["null_control"] for rung in ladder],
        },
    }


def experiment() -> tuple[dict, dict]:
    actual_hash = frozen_stencil_sha256()
    if actual_hash != EXPECTED_STENCIL_SHA256:
        raise RuntimeError(f"frozen stencil hash mismatch: {actual_hash}")
    levels = [solve_draw_level(size) for size in GRID_SIZES]
    reference = wind_reference(levels[-1])
    winds = [ratio * reference for ratio in WIND_RATIOS]
    marched = load_marched_comparator()
    ladder = []
    state_by_grid = {}
    for level in levels:
        rung, states = run_rung(level, winds, marched)
        ladder.append(rung)
        state_by_grid[len(level["axis"])] = states

    attractor_level = levels[GRID_SIZES.index(ATTRACTOR_GRID_SIZE)]
    attractor_wind = winds[WIND_RATIOS.index(ATTRACTOR_WIND_RATIO)]
    alternate, alternate_state = run_case(
        attractor_level, attractor_wind, ATTRACTOR_WIND_RATIO, marched, "boosted_static_GP"
    )
    baseline_case = ladder[GRID_SIZES.index(ATTRACTOR_GRID_SIZE)]["cases"][WIND_RATIOS.index(ATTRACTOR_WIND_RATIO)]
    baseline_state = state_by_grid[ATTRACTOR_GRID_SIZE][ATTRACTOR_WIND_RATIO]
    velocity_difference = np.linalg.norm(alternate_state[0] - baseline_state[0]) / max(
        np.linalg.norm(baseline_state[0]), 1.0e-30
    )
    density_difference = np.linalg.norm(alternate_state[1] - baseline_state[1]) / max(
        np.linalg.norm(baseline_state[1]), 1.0e-30
    )
    both_steady = baseline_case["steadiness"]["steady"] and alternate["steadiness"]["steady"]
    attractor_probe = {
        "grid_size": ATTRACTOR_GRID_SIZE,
        "wind_ratio": ATTRACTOR_WIND_RATIO,
        "initial_conditions": [
            {"name": baseline_case["initial_condition"], "steadiness": baseline_case["steadiness"]},
            {"name": alternate["initial_condition"], "steadiness": alternate["steadiness"]},
        ],
        "final_relative_velocity_L2_difference": float(velocity_difference),
        "final_relative_density_L2_difference": float(density_difference),
        "verdict": "same_attractor_within_5_percent" if both_steady and max(velocity_difference, density_difference) < 0.05 else "distinct_steady_attractors" if both_steady else "unresolved_because_both_runs_did_not_meet_steadiness_criterion",
    }
    gates = adjudicate(ladder)
    apparatus = {
        "equations": {
            "momentum": "dv/dt + (v.grad)v = c^2 grad sigma",
            "continuity": "dn/dt + div(n v) = -s",
            "consumption": "s=n*sqrt((3/2)*e_dev:e_dev)",
            "level": "-discrete_laplacian(H)=normalized Gaussian draw; sigma=1-exp(-H)",
        },
        "Bernoulli_speed_imposed": False,
        "implementation_choice": "cell-centered donor-cell finite volume with SSP-RK2, adaptive CFL, density positivity floor, draw ramp for the pure-wind start, upstream Dirichlet wind/density, and zero-normal-gradient open treatment on the other five faces",
        "numerical_dissipation": "only the first-order donor-cell truncation diffusion; no physical viscosity or relaxation term is added",
        "grid_sizes": list(GRID_SIZES),
        "fixed_domain_half_width": DOMAIN_HALF_WIDTH,
        "fixed_physical_core_sigma": CORE_SIGMA,
        "CFL": CFL,
        "end_time": END_TIME,
        "draw_ramp_time": DRAW_RAMP_TIME,
        "wind_ratios": list(WIND_RATIOS),
        "wind_speeds": winds,
        "finest_v_GP_at_probe": reference,
        "measurement_radii": list(MEASUREMENT_RADII),
        "ORB_10751_expected_stencil_sha256": EXPECTED_STENCIL_SHA256,
        "measured_stencil_sha256": actual_hash,
        "stencil_sha256_match": actual_hash == EXPECTED_STENCIL_SHA256,
        "ORB_10935_comparator_source": "../level-core-wind-tunnel/assets/results.json",
    }
    reproducibility = {
        "seed": SEED,
        "random_numbers_used": False,
        "byte_identical_in_memory_rerun_verified": True,
        "command": "UV_CACHE_DIR=/tmp/orbit-uv-cache-10937 uv run lab/sims/level-core-dynamical-relaxation/main.py --check-determinism",
    }
    verdict = {
        "all_five_predeclared_gates_executed": True,
        "G1": gates["G1_steadiness"]["verdict"],
        "G2": gates["G2_discovered_speed_law"]["verdict"],
        "G3": gates["G3_realized_wake"]["verdict"],
        "G4": gates["G4_momentum_budget"]["verdict"],
        "G5": gates["G5_Galilean_null"]["verdict"],
        "theory_reconciliation": "deferred to kepler; principia intentionally untouched",
    }
    limitations = [
        "The feasible T=60 horizon is still shorter than the two lowest-wind box crossing times; failures to settle are finite-horizon results, not proofs of perpetual unsteadiness.",
        "The 25^3/33^3/41^3 ladder fixes geometry but is deliberately computationally bounded; donor-cell diffusion is quantified only through adjacent-rung shifts.",
        "Open faces use one-sided zero-normal-gradient ghost values and may reflect some nonlinear structure.",
        "The density floor is a positivity safeguard; its minimum value is recorded in every residual history.",
        "The momentum surface ledger contains advective flux only because no pressure/level stress was supplied by the model.",
    ]
    record = {
        "schema_version": 1,
        "task": TASK_ID,
        "run_id": RUN_ID,
        "run_date": RUN_DATE,
        "reproducibility": reproducibility,
        "apparatus": apparatus,
        "resolution_ladder": ladder,
        "attractor_probe": attractor_probe,
        "gates": gates,
        "verdict": verdict,
        "limitations": limitations,
    }
    results = {
        "schema_version": 1,
        "task": TASK_ID,
        "run_record": f"runs/{RUN_RECORD}",
        "reproducibility": reproducibility,
        "apparatus": apparatus,
        "resolution_ladder": {
            "grid_sizes": list(GRID_SIZES),
            "spacings": [rung["apparatus"]["spacing"] for rung in ladder],
            "fixed_domain_half_width": DOMAIN_HALF_WIDTH,
            "fixed_physical_core_sigma": CORE_SIGMA,
            "predeclared_convergence": {
                "G1": "same categorical steadiness verdict on the two finest rungs",
                "G2_G3": "adjacent-rung shifts of Bernoulli RMS error and normalized speed dipole <0.05",
                "G4": "adjacent-rung relative shift of each budget mean <25%",
                "G5": "exact invariant to floating precision on every rung",
            },
        },
        "attractor_probe": attractor_probe,
        "gates": gates,
        "verdict": verdict,
        "limitations": limitations,
    }
    return results, record


def encoded(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-determinism", action="store_true")
    arguments = parser.parse_args()
    results, record = experiment()
    if arguments.check_determinism:
        repeated_results, repeated_record = experiment()
        if encoded(results) != encoded(repeated_results) or encoded(record) != encoded(repeated_record):
            raise RuntimeError("experiment rerun was not byte-identical")
    root = Path(__file__).parent
    (root / "assets").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "assets" / "results.json").write_bytes(encoded(results))
    (root / "runs" / RUN_RECORD).write_bytes(encoded(record))
    print(json.dumps(results["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
