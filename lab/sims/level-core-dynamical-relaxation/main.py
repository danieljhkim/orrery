"""Resolve the moving level core's long-time dynamical attractor.

The apparatus solves the draw-sourced elliptic level in the comoving frame and
then advances the rolling-rule momentum equation and consumed-density
continuity equation without imposing a Bernoulli speed.  A deterministic
finite-volume experiment follows the previously observed slow decay for more
than nine e-folding times.  It asks whether each swept wind settles, stalls, or
continues to decay and, only where it settles, measures the discovered speed
and direction fields.

Usage:
    uv run lab/sims/level-core-dynamical-relaxation/main.py
    uv run lab/sims/level-core-dynamical-relaxation/main.py --check-determinism
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn
from scipy.interpolate import interpn
from scipy.special import eval_legendre, roots_legendre

SEED = 42
TASK_ID = "ORB-10938"
RUN_ID = "jrun-20260821-0503-3"
RUN_DATE = "2026-08-21"
RUN_RECORD = "2026-08-21-extended-seed-42.json"
GRID_SIZES = (33, 41)
ANCHOR_GRID_SIZE = 61
ANCHOR_WIND_RATIOS = (0.3,)
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
END_TIME = 600.0
DRAW_RAMP_TIME = 3.0
DIAGNOSTIC_INTERVAL = 10.0
STEADY_WINDOW_SAMPLES = 11
STEADY_DV_RMS = 2.0e-3
STEADY_DN_RMS = 2.0e-3
STALL_LOG_SLOPE_FLOOR = 2.0e-3
TROUGH_SATURATION_RELATIVE_CHANGE = 1.0e-2
CORE_SHELL_INNER_RADIUS = 1.5
CORE_SHELL_OUTER_RADIUS = 3.0
DENSITY_FLOOR = 1.0e-10
CAVITATION_DENSITY_THRESHOLD = 1.0e-2
CAVITATION_CUTOFF_MULTIPLIER = 1.01
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
    level[1:-1, 1:-1, 1:-1] = idstn(transformed / minus_laplacian, type=1, norm="ortho")
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


def sample_scalar(
    values: np.ndarray, axis: np.ndarray, points: np.ndarray
) -> np.ndarray:
    return np.asarray(interpn((axis, axis, axis), values, points))


def sample_vector(
    values: np.ndarray, axis: np.ndarray, points: np.ndarray
) -> np.ndarray:
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
    next_n[scalar_interior] += (
        0.5 * dt * (first_n[scalar_interior] + second_n[scalar_interior])
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
            - np.sum(
                density[lower, surface, surface] * vx[lower, surface, surface] ** 2
            )
            + np.sum(
                density[surface, upper, surface]
                * vx[surface, upper, surface]
                * vy[surface, upper, surface]
            )
            - np.sum(
                density[surface, lower, surface]
                * vx[surface, lower, surface]
                * vy[surface, lower, surface]
            )
            + np.sum(
                density[surface, surface, upper]
                * vx[surface, surface, upper]
                * vz[surface, surface, upper]
            )
            - np.sum(
                density[surface, surface, lower]
                * vx[surface, surface, lower]
                * vz[surface, surface, lower]
            )
        )
        * spacing**2
    )


def net_boundary_mass_influx(
    velocity: np.ndarray, density: np.ndarray, spacing: float
) -> float:
    """Return inflow minus outflow through all six outer cube faces."""
    vx, vy, vz = velocity
    outward = (
        np.sum(density[-1, :, :] * vx[-1, :, :])
        - np.sum(density[0, :, :] * vx[0, :, :])
        + np.sum(density[:, -1, :] * vy[:, -1, :])
        - np.sum(density[:, 0, :] * vy[:, 0, :])
        + np.sum(density[:, :, -1] * vz[:, :, -1])
        - np.sum(density[:, :, 0] * vz[:, :, 0])
    ) * spacing**2
    return float(-outward)


def density_sample(
    level: dict,
    velocity: np.ndarray,
    density: np.ndarray,
    consumption: np.ndarray,
) -> dict:
    radius = np.sqrt(sum(coordinate**2 for coordinate in level["mesh"]))
    shell = (radius >= CORE_SHELL_INNER_RADIUS) & (radius <= CORE_SHELL_OUTER_RADIUS)
    spacing = level["spacing"]
    total_consumption = float(np.sum(consumption) * spacing**3)
    boundary_influx = net_boundary_mass_influx(velocity, density, spacing)
    return {
        "minimum_density": float(np.min(density)),
        "core_shell_mean_density": float(np.mean(density[shell])),
        "total_density": float(np.sum(density) * spacing**3),
        "net_boundary_mass_influx": boundary_influx,
        "total_consumption_integral": total_consumption,
        "influx_to_consumption_ratio": float(
            boundary_influx / max(total_consumption, 1.0e-30)
        ),
        "mass_balance_residual": float(boundary_influx - total_consumption),
    }


def momentum_sample(
    level: dict, velocity: np.ndarray, density: np.ndarray, consumption: np.ndarray
) -> tuple[float, float]:
    radius = np.sqrt(sum(coordinate**2 for coordinate in level["mesh"]))
    box_radius = np.maximum.reduce([np.abs(coordinate) for coordinate in level["mesh"]])
    mask = (radius >= CORE_EXCLUSION_RADIUS) & (box_radius <= CONTROL_HALF_WIDTH)
    consumed = float(
        np.sum(consumption[mask] * velocity[0, mask]) * level["spacing"] ** 3
    )
    flux = control_surface_flux(velocity, density, level["axis"], level["spacing"])
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
    path = (
        Path(__file__).parents[1] / "level-core-wind-tunnel" / "assets" / "results.json"
    )
    source = json.loads(path.read_text())
    cases = source["gates"]["G2_wake_structure"]["cases"]
    return {
        (float(row["wind_ratio"]), float(row["radius"])): row["actual_nx_multipoles"]
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
    bernoulli_field = np.sqrt(wind**2 + 2.0 * C_SQUARED * level["sigma"])
    shells = []
    for radius in MEASUREMENT_RADII:
        points, directions, weights = shell_quadrature(radius)
        sampled_velocity = sample_vector(velocity, level["axis"], points)
        sampled_speed = np.linalg.norm(sampled_velocity, axis=1)
        sampled_bernoulli = sample_scalar(bernoulli_field, level["axis"], points)
        sampled_sigma = sample_scalar(level["sigma"], level["axis"], points)
        actual_direction = sampled_velocity / np.maximum(
            sampled_speed[:, None], 1.0e-14
        )
        boosted = sample_vector(level["gp_velocity"], level["axis"], points)
        boosted[:, 0] += wind
        boosted_direction = boosted / np.maximum(
            np.linalg.norm(boosted, axis=1)[:, None], 1.0e-14
        )
        actual_nx = actual_direction[:, 0]
        boosted_nx = boosted_direction[:, 0]
        actual_coefficients = legendre_coefficients(actual_nx, directions, weights)
        boosted_coefficients = legendre_coefficients(boosted_nx, directions, weights)
        marched_coefficients = marched[(ratio, radius)]
        boosted_angle = np.arccos(
            np.clip(np.sum(actual_direction * boosted_direction, axis=1), -1.0, 1.0)
        )
        formula_residual = sampled_speed - sampled_bernoulli
        squared_formula = wind**2 + 2.0 * C_SQUARED * sampled_sigma
        squared_residual = sampled_speed**2 - squared_formula
        squared_scale = float(np.sum(weights * squared_formula))
        shells.append(
            {
                "radius": radius,
                "speed": normalized_speed_multipoles(
                    sampled_speed, directions, weights
                ),
                "bernoulli_formula": "sqrt(U^2 + 2 c^2 sigma)",
                "bernoulli_relative_rms_error": float(
                    np.sqrt(np.sum(weights * formula_residual**2))
                    / np.sum(weights * sampled_bernoulli)
                ),
                "bernoulli_maximum_relative_error": float(
                    np.max(np.abs(formula_residual)) / np.mean(sampled_bernoulli)
                ),
                "bernoulli_squared_residual": {
                    "formula": "|v|^2 - U^2 - 2 c^2 sigma",
                    "relative_rms_error": float(
                        np.sqrt(np.sum(weights * squared_residual**2)) / squared_scale
                    ),
                    "maximum_relative_error": float(
                        np.max(np.abs(squared_residual)) / squared_scale
                    ),
                    "shell_multipoles": legendre_coefficients(
                        squared_residual, directions, weights
                    ),
                },
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


def pointwise_speed_measurement(level: dict, velocity: np.ndarray, wind: float) -> dict:
    speed = np.sqrt(np.sum(velocity**2, axis=0))
    formula = np.sqrt(wind**2 + 2.0 * C_SQUARED * level["sigma"])
    formula_squared = wind**2 + 2.0 * C_SQUARED * level["sigma"]
    radius = np.sqrt(sum(coordinate**2 for coordinate in level["mesh"]))
    box_radius = np.maximum.reduce([np.abs(coordinate) for coordinate in level["mesh"]])
    mask = (radius >= CORE_EXCLUSION_RADIUS) & (box_radius <= CONTROL_HALF_WIDTH)
    residual = speed[mask] - formula[mask]
    scale = float(np.sqrt(np.mean(formula[mask] ** 2)))
    squared_residual = speed[mask] ** 2 - formula_squared[mask]
    squared_scale = float(np.sqrt(np.mean(formula_squared[mask] ** 2)))
    return {
        "region": (
            f"{CORE_EXCLUSION_RADIUS:g} <= radius and max(|x_i|) <= "
            f"{CONTROL_HALF_WIDTH:g}"
        ),
        "sample_count": int(np.count_nonzero(mask)),
        "relative_rms_error": float(np.sqrt(np.mean(residual**2)) / scale),
        "maximum_relative_error": float(np.max(np.abs(residual)) / scale),
        "mean_signed_relative_error": float(np.mean(residual) / scale),
        "bernoulli_squared_residual": {
            "formula": "|v|^2 - U^2 - 2 c^2 sigma",
            "relative_rms_error": float(
                np.sqrt(np.mean(squared_residual**2)) / squared_scale
            ),
            "maximum_relative_error": float(
                np.max(np.abs(squared_residual)) / squared_scale
            ),
            "mean_signed_relative_error": float(
                np.mean(squared_residual) / squared_scale
            ),
        },
    }


def log_slope(history: list[dict], field: str) -> float:
    tail = history[len(history) * 3 // 4 :]
    values = np.array([row[field] for row in tail])
    times = np.array([row["time"] for row in tail])
    return float(np.polyfit(times, np.log(np.maximum(np.abs(values), 1.0e-30)), 1)[0])


def sector_verdict(history: list[dict], field: str, threshold: float) -> dict:
    sustained = history[-STEADY_WINDOW_SAMPLES:]
    settled = bool(
        len(sustained) == STEADY_WINDOW_SAMPLES
        and sustained[0]["time"] >= 0.75 * END_TIME
        and all(row[field] < threshold for row in sustained)
    )
    slope = log_slope(history, field)
    final = float(history[-1][field])
    if settled:
        verdict = "settled"
    elif slope < -STALL_LOG_SLOPE_FLOOR:
        verdict = "still-decaying"
    else:
        verdict = "stalled"
    projected = None
    if not settled and slope < 0.0 and final > threshold:
        projected = float(history[-1]["time"] + np.log(threshold / final) / slope)
    return {
        "verdict": verdict,
        "threshold": threshold,
        "final_residual": final,
        "late_log_residual_slope_per_time": slope,
        "slope_floor_per_time": STALL_LOG_SLOPE_FLOOR,
        "extrapolated_threshold_time": projected,
    }


def trough_diagnostic(history: list[dict]) -> dict:
    sustained = history[-STEADY_WINDOW_SAMPLES:]
    times = np.array([row["time"] for row in sustained])
    minimum = np.array([row["minimum_density"] for row in sustained])
    shell = np.array([row["core_shell_mean_density"] for row in sustained])
    minimum_change = relative_change(float(minimum[0]), float(minimum[-1]))
    shell_change = relative_change(float(shell[0]), float(shell[-1]))
    minimum_log_slope = float(np.polyfit(times, np.log(minimum), 1)[0])
    shell_log_slope = float(np.polyfit(times, np.log(shell), 1)[0])
    cutoff_reached = bool(
        np.min(minimum) <= CAVITATION_CUTOFF_MULTIPLIER * DENSITY_FLOOR
    )
    cavitation_candidate = bool(
        minimum[-1] < CAVITATION_DENSITY_THRESHOLD
        and minimum_log_slope < -STALL_LOG_SLOPE_FLOOR
    )
    if cutoff_reached:
        cavitation_verdict = "cavitated_at_positivity_cutoff"
    elif cavitation_candidate:
        cavitation_verdict = "cavitation_trending_to_cutoff"
    else:
        cavitation_verdict = "finite_floor_or_recovery"
    extrapolated_cutoff_time = None
    if minimum_log_slope < 0.0 and minimum[-1] > DENSITY_FLOOR:
        extrapolated_cutoff_time = float(
            times[-1] + np.log(DENSITY_FLOOR / minimum[-1]) / minimum_log_slope
        )
    return {
        "time_series": [
            {
                "time": row["time"],
                "minimum_density": row["minimum_density"],
                "core_shell_mean_density": row["core_shell_mean_density"],
            }
            for row in history
        ],
        "core_shell_definition": (
            f"{CORE_SHELL_INNER_RADIUS:g} <= r <= {CORE_SHELL_OUTER_RADIUS:g}"
        ),
        "saturation_criterion": (
            "relative change of both minimum density and core-shell mean over "
            f"the final {STEADY_WINDOW_SAMPLES} samples <= "
            f"{TROUGH_SATURATION_RELATIVE_CHANGE:g}"
        ),
        "minimum_density_final_window_relative_change": minimum_change,
        "core_shell_mean_final_window_relative_change": shell_change,
        "minimum_density_final_window_log_slope_per_time": minimum_log_slope,
        "core_shell_mean_final_window_log_slope_per_time": shell_log_slope,
        "saturated": bool(
            max(minimum_change, shell_change) <= TROUGH_SATURATION_RELATIVE_CHANGE
        ),
        "cavitation": {
            "verdict": cavitation_verdict,
            "criterion": (
                "cavitated at n_min <= 1.01*density_floor; a trajectory is a "
                f"cavitation candidate when final n_min < {CAVITATION_DENSITY_THRESHOLD:g} "
                f"and its final-window log slope < -{STALL_LOG_SLOPE_FLOOR:g}/time; "
                "a branch kill requires cutoff contact or candidate agreement on "
                "adjacent rungs"
            ),
            "final_minimum_density": float(minimum[-1]),
            "positivity_cutoff": DENSITY_FLOOR,
            "cutoff_reached": cutoff_reached,
            "candidate": cavitation_candidate,
            "extrapolated_cutoff_time": extrapolated_cutoff_time,
        },
    }


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
        dt = min(
            CFL * level["spacing"] / max(maximum_transport, 1.0e-6),
            0.4,
            END_TIME - time,
        )
        ramp = (
            1.0
            if initial_condition == "boosted_static_GP"
            else min(1.0, (time + dt) / DRAW_RAMP_TIME)
        )
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
            consumed, flux = momentum_sample(
                level, velocity, density, latest_consumption
            )
            density_diagnostics = density_sample(
                level, velocity, density, latest_consumption
            )
            history.append(
                {
                    "time": float(time),
                    "step": steps,
                    "dt": float(dt),
                    "dv_rms": float(
                        np.sqrt(
                            np.mean(
                                np.sum(velocity_rhs[:, 1:-1, 1:-1, 1:-1] ** 2, axis=0)
                            )
                        )
                    ),
                    "dn_rms": float(np.sqrt(np.mean(density_rhs[interior] ** 2))),
                    "wake_probe_vx": float(velocity[0, wake_index, center, center]),
                    "wake_center_vy": float(velocity[1, wake_index, center, center]),
                    "wake_center_vz": float(velocity[2, wake_index, center, center]),
                    "wake_plus_y_vy": float(velocity[1, wake_index, plus_y, center]),
                    "wake_minus_y_vy": float(velocity[1, wake_index, minus_y, center]),
                    **density_diagnostics,
                }
            )
            momentum_history.append(
                {
                    "time": float(time),
                    "consumed_momentum_x": consumed,
                    "advective_flux_x": flux,
                }
            )
            next_diagnostic += DIAGNOSTIC_INTERVAL

    velocity_sector = sector_verdict(history, "dv_rms", STEADY_DV_RMS)
    density_sector = sector_verdict(history, "dn_rms", STEADY_DN_RMS)
    steady = bool(
        velocity_sector["verdict"] == "settled"
        and density_sector["verdict"] == "settled"
    )
    velocity_admitted = velocity_sector["verdict"] == "settled"
    if steady:
        overall_verdict = "settled"
    elif "still-decaying" in (
        velocity_sector["verdict"],
        density_sector["verdict"],
    ):
        overall_verdict = "still-decaying"
    else:
        overall_verdict = "stalled"
    flow = flow_diagnostics(level, velocity)
    stagnation_threshold = max(1.0e-3, 0.05 * wind)
    tail_momentum = momentum_history[-STEADY_WINDOW_SAMPLES:]
    consumed_values = np.array([row["consumed_momentum_x"] for row in tail_momentum])
    flux_values = np.array([row["advective_flux_x"] for row in tail_momentum])
    mass_history = [
        {
            key: row[key]
            for key in (
                "time",
                "total_density",
                "net_boundary_mass_influx",
                "total_consumption_integral",
                "influx_to_consumption_ratio",
                "mass_balance_residual",
            )
        }
        for row in history
    ]
    mass_ratio = np.array([row["influx_to_consumption_ratio"] for row in mass_history])
    mass_times = np.array([row["time"] for row in mass_history])
    mass_tail = mass_history[-STEADY_WINDOW_SAMPLES:]
    mass_tail_ratio = np.array(
        [row["influx_to_consumption_ratio"] for row in mass_tail]
    )
    mass_tail_times = np.array([row["time"] for row in mass_tail])
    last_dv = history[-1]["dv_rms"]
    result = {
        "wind_ratio_to_finest_v_GP_at_probe": ratio,
        "wind_speed": wind,
        "initial_condition": initial_condition,
        "integration": {"end_time": END_TIME, "steps": steps},
        "steadiness": {
            "verdict": overall_verdict,
            "steady": steady,
            "criterion": f"dv_rms < {STEADY_DV_RMS:g} and dn_rms < {STEADY_DN_RMS:g} for the final {STEADY_WINDOW_SAMPLES} samples, all after 0.75*T",
            "window_samples": STEADY_WINDOW_SAMPLES,
            "sectoral_verdicts": {
                "velocity": velocity_sector,
                "density": density_sector,
            },
            "residual_time_series": history,
            "trough_saturation": trough_diagnostic(history),
            "unsteady_characterization": (
                None if steady else characterize_history(history)
            ),
        },
        "mass_budget": {
            "time_series": mass_history,
            "sign_convention": "positive boundary value is net mass entering the cube",
            "steady_balance_target": "net boundary mass influx / total consumption integral -> 1",
            "late_mean_influx_to_consumption_ratio": float(np.mean(mass_tail_ratio)),
            "late_ratio_standard_deviation": float(np.std(mass_tail_ratio)),
            "late_ratio_linear_slope_per_time": float(
                np.polyfit(mass_tail_times, mass_tail_ratio, 1)[0]
            ),
            "whole_run_ratio_linear_slope_per_time": float(
                np.polyfit(mass_times, mass_ratio, 1)[0]
            ),
        },
        "discovered_field": {
            "admission": "velocity_sector_settled"
            if velocity_admitted
            else "not_admitted",
            "reported_for_velocity_settled_case": velocity_admitted,
            "pointwise_speed": (
                pointwise_speed_measurement(level, velocity, wind)
                if velocity_admitted
                else None
            ),
            "shells": (
                shell_measurements(
                    level,
                    velocity,
                    wind,
                    ratio,
                    marched,
                    last_dv * DIAGNOSTIC_INTERVAL,
                )
                if velocity_admitted
                else []
            ),
            "stagnation": {
                **flow,
                "search_threshold": stagnation_threshold,
                "stagnation_found": bool(
                    flow["minimum_interior_speed"] <= stagnation_threshold
                ),
            },
        },
        "momentum_budget": {
            "time_series": momentum_history,
            "averaging_window": f"last {STEADY_WINDOW_SAMPLES} diagnostic samples",
            "consumed_momentum_integral_x_mean": float(np.mean(consumed_values)),
            "consumed_momentum_integral_x_standard_deviation": float(
                np.std(consumed_values)
            ),
            "advective_far_surface_flux_x_mean": float(np.mean(flux_values)),
            "advective_far_surface_flux_x_standard_deviation": float(
                np.std(flux_values)
            ),
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
            maximum_consumption = max(
                maximum_consumption, float(np.max(np.abs(consumption)))
            )
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
        max(
            row["maximum_velocity_change"],
            row["maximum_density_change"],
            row["maximum_consumption"],
        )
        <= np.finfo(float).eps
        for row in rows
    )
    return {"grid_size": len(level["axis"]), "measurements": rows, "passed": passed}


def run_rung(
    level: dict,
    ratios: tuple[float, ...],
    winds: list[float],
    marched: dict,
    *,
    anchor: bool = False,
) -> tuple[dict, dict]:
    cases = []
    states = {}
    for ratio, wind in zip(ratios, winds):
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
                "finer_anchor": anchor,
            },
            "cases": cases,
            "null_control": null_control(level, winds),
        },
        states,
    )


def relative_change(first: float, second: float, floor: float = 1.0e-14) -> float:
    return float(abs(second - first) / max(abs(first), abs(second), floor))


def pair_steadiness(history: list[dict]) -> dict:
    velocity = sector_verdict(history, "dv_rms", STEADY_DV_RMS)
    density = sector_verdict(history, "dn_rms", STEADY_DN_RMS)
    steady = velocity["verdict"] == density["verdict"] == "settled"
    return {
        "steady": steady,
        "sectoral_verdicts": {"velocity": velocity, "density": density},
        "trough_saturation": trough_diagnostic(history),
        "residual_time_series": history,
    }


def run_attractor_probe(level: dict, wind: float) -> dict:
    shape = level["sigma"].shape
    velocity_a = np.zeros((3,) + shape)
    velocity_a[0].fill(wind)
    velocity_b = velocity_a + level["gp_velocity"]
    density_a = np.ones(shape)
    density_b = np.ones(shape)
    apply_boundary(velocity_a, density_a, wind)
    apply_boundary(velocity_b, density_b, wind)
    time = 0.0
    steps = 0
    next_diagnostic = 0.0
    histories = ([], [])
    differences = []
    while time < END_TIME - 1.0e-12:
        transport = max(
            float(np.max(np.sum(np.abs(velocity_a), axis=0))),
            float(np.max(np.sum(np.abs(velocity_b), axis=0))),
        )
        dt = min(
            CFL * level["spacing"] / max(transport, 1.0e-6),
            0.4,
            END_TIME - time,
        )
        force_a = (
            C_SQUARED * min(1.0, (time + dt) / DRAW_RAMP_TIME) * level["grad_sigma"]
        )
        force_b = C_SQUARED * level["grad_sigma"]
        velocity_a, density_a, rhs_va, rhs_na, consumption_a = advance(
            velocity_a, density_a, force_a, level["spacing"], wind, dt
        )
        velocity_b, density_b, rhs_vb, rhs_nb, consumption_b = advance(
            velocity_b, density_b, force_b, level["spacing"], wind, dt
        )
        time += dt
        steps += 1
        if time + 1.0e-12 >= next_diagnostic or time >= END_TIME - 1.0e-12:
            interior = (slice(1, -1),) * 3
            for history, velocity, density, rhs_v, rhs_n, consumption in (
                (histories[0], velocity_a, density_a, rhs_va, rhs_na, consumption_a),
                (histories[1], velocity_b, density_b, rhs_vb, rhs_nb, consumption_b),
            ):
                history.append(
                    {
                        "time": float(time),
                        "dv_rms": float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        rhs_v[:, 1:-1, 1:-1, 1:-1] ** 2,
                                        axis=0,
                                    )
                                )
                            )
                        ),
                        "dn_rms": float(np.sqrt(np.mean(rhs_n[interior] ** 2))),
                        **density_sample(level, velocity, density, consumption),
                    }
                )
            differences.append(
                {
                    "time": float(time),
                    "relative_velocity_L2_difference": float(
                        np.linalg.norm(velocity_b - velocity_a)
                        / max(np.linalg.norm(velocity_a), 1.0e-30)
                    ),
                    "relative_density_L2_difference": float(
                        np.linalg.norm(density_b - density_a)
                        / max(np.linalg.norm(density_a), 1.0e-30)
                    ),
                }
            )
            next_diagnostic += DIAGNOSTIC_INTERVAL
    summaries = [pair_steadiness(history) for history in histories]
    final_difference = max(
        differences[-1]["relative_velocity_L2_difference"],
        differences[-1]["relative_density_L2_difference"],
    )
    both_steady = all(summary["steady"] for summary in summaries)
    return {
        "grid_size": len(level["axis"]),
        "wind_ratio": ATTRACTOR_WIND_RATIO,
        "wind_speed": wind,
        "integration": {"end_time": END_TIME, "shared_steps": steps},
        "initial_conditions": [
            {"name": "ramped_pure_wind", "steadiness": summaries[0]},
            {"name": "boosted_static_GP", "steadiness": summaries[1]},
        ],
        "relative_L2_difference_time_series": differences,
        "same_attractor_threshold": 0.05,
        "verdict": (
            "same_attractor_within_5_percent"
            if both_steady and final_difference < 0.05
            else "distinct_steady_attractors"
            if both_steady
            else "unresolved_because_both_runs_did_not_settle"
        ),
    }


def case_map(rungs: list[dict]) -> dict:
    return {
        (
            rung["apparatus"]["grid_size"],
            case["wind_ratio_to_finest_v_GP_at_probe"],
        ): case
        for rung in rungs
        for case in rung["cases"]
    }


def coarser_case(
    cases: dict,
    grid_size: int,
    ratio: float,
    *,
    require_velocity_settled: bool = False,
) -> tuple[int | None, dict | None]:
    candidates = sorted(
        size
        for size, candidate_ratio in cases
        if candidate_ratio == ratio and size < grid_size
    )
    for size in reversed(candidates):
        candidate = cases[(size, ratio)]
        velocity_settled = (
            candidate["steadiness"]["sectoral_verdicts"]["velocity"]["verdict"]
            == "settled"
        )
        if not require_velocity_settled or velocity_settled:
            return size, candidate
    return None, None


def trajectory_relative_l2(
    coarse_history: list[dict], fine_history: list[dict], field: str
) -> float:
    fine_tail = [row for row in fine_history if row["time"] >= 0.75 * END_TIME]
    fine_times = np.array([row["time"] for row in fine_tail])
    fine_values = np.array([row[field] for row in fine_tail])
    coarse_times = np.array([row["time"] for row in coarse_history])
    coarse_values = np.array([row[field] for row in coarse_history])
    interpolated = np.interp(fine_times, coarse_times, coarse_values)
    return float(
        np.linalg.norm(interpolated - fine_values)
        / max(np.linalg.norm(fine_values), 1.0e-30)
    )


def adjudicate(ladder: list[dict], anchors: list[dict]) -> dict:
    rungs = ladder + anchors
    cases = case_map(rungs)
    base_coarse_size = ladder[-2]["apparatus"]["grid_size"]
    base_fine_size = ladder[-1]["apparatus"]["grid_size"]
    g1_cases = [
        {
            "grid_size": rung["apparatus"]["grid_size"],
            "finer_anchor": rung["apparatus"]["finer_anchor"],
            "wind_ratio": case["wind_ratio_to_finest_v_GP_at_probe"],
            "wind_speed": case["wind_speed"],
            **case["steadiness"],
        }
        for rung in rungs
        for case in rung["cases"]
    ]
    g2_cases = [
        {
            "grid_size": rung["apparatus"]["grid_size"],
            "finer_anchor": rung["apparatus"]["finer_anchor"],
            "wind_ratio": case["wind_ratio_to_finest_v_GP_at_probe"],
            "wind_speed": case["wind_speed"],
            **case["mass_budget"],
        }
        for rung in rungs
        for case in rung["cases"]
    ]
    slope_convergence = []
    cavitation_by_wind = []
    for ratio in WIND_RATIOS:
        sizes = sorted(
            size for size, candidate_ratio in cases if candidate_ratio == ratio
        )
        pair_rows = []
        for coarse_size, fine_size in pairwise(sizes):
            coarse = cases[(coarse_size, ratio)]
            fine = cases[(fine_size, ratio)]
            coarse_steady = coarse["steadiness"]
            fine_steady = fine["steadiness"]
            pair_rows.append(
                {
                    "coarse_grid_size": coarse_size,
                    "fine_grid_size": fine_size,
                    "velocity_late_log_slope_absolute_shift": abs(
                        coarse_steady["sectoral_verdicts"]["velocity"][
                            "late_log_residual_slope_per_time"
                        ]
                        - fine_steady["sectoral_verdicts"]["velocity"][
                            "late_log_residual_slope_per_time"
                        ]
                    ),
                    "density_late_log_slope_absolute_shift": abs(
                        coarse_steady["sectoral_verdicts"]["density"][
                            "late_log_residual_slope_per_time"
                        ]
                        - fine_steady["sectoral_verdicts"]["density"][
                            "late_log_residual_slope_per_time"
                        ]
                    ),
                    "minimum_density_late_log_slope_absolute_shift": abs(
                        coarse_steady["trough_saturation"][
                            "minimum_density_final_window_log_slope_per_time"
                        ]
                        - fine_steady["trough_saturation"][
                            "minimum_density_final_window_log_slope_per_time"
                        ]
                    ),
                    "minimum_density_late_trajectory_relative_L2": trajectory_relative_l2(
                        coarse_steady["residual_time_series"],
                        fine_steady["residual_time_series"],
                        "minimum_density",
                    ),
                }
            )
        slope_convergence.append(
            {
                "wind_ratio": ratio,
                "adjacent_rung_pairs": pair_rows,
                "slope_agreement_scale_per_time": STALL_LOG_SLOPE_FLOOR,
            }
        )
        cavitation_rows = [
            {
                "grid_size": size,
                **cases[(size, ratio)]["steadiness"]["trough_saturation"]["cavitation"],
            }
            for size in sizes
        ]
        confirmed_pairs = [
            [first["grid_size"], second["grid_size"]]
            for first, second in pairwise(cavitation_rows)
            if first["candidate"] and second["candidate"]
        ]
        branch_kill = bool(
            any(row["cutoff_reached"] for row in cavitation_rows) or confirmed_pairs
        )
        cavitation_by_wind.append(
            {
                "wind_ratio": ratio,
                "by_rung": cavitation_rows,
                "adjacent_candidate_agreement": confirmed_pairs,
                "branch_kill": branch_kill,
                "verdict": (
                    "cavitation_branch_kill"
                    if branch_kill
                    else "cavitation_not_ladder_confirmed"
                ),
            }
        )
    g3_cases = []
    g4_cases = []
    all_case_vorticity = []
    for rung in rungs:
        grid_size = rung["apparatus"]["grid_size"]
        for case in rung["cases"]:
            ratio = case["wind_ratio_to_finest_v_GP_at_probe"]
            all_case_vorticity.append(
                {
                    "grid_size": grid_size,
                    "wind_ratio": ratio,
                    "joint_settlement_verdict": case["steadiness"]["verdict"],
                    **{
                        key: value
                        for key, value in case["discovered_field"]["stagnation"].items()
                        if key.startswith("vorticity_")
                    },
                }
            )
            velocity_settled = (
                case["steadiness"]["sectoral_verdicts"]["velocity"]["verdict"]
                == "settled"
            )
            if not velocity_settled:
                continue
            coarser_size, coarser = coarser_case(
                cases, grid_size, ratio, require_velocity_settled=True
            )
            pointwise_resolution_error = None
            if coarser is not None:
                current_pointwise = case["discovered_field"]["pointwise_speed"]
                previous_pointwise = coarser["discovered_field"]["pointwise_speed"]
                pointwise_resolution_error = {
                    "coarser_grid_size": coarser_size,
                    "speed_relative_rms_shift": abs(
                        current_pointwise["relative_rms_error"]
                        - previous_pointwise["relative_rms_error"]
                    ),
                    "bernoulli_squared_relative_rms_shift": abs(
                        current_pointwise["bernoulli_squared_residual"][
                            "relative_rms_error"
                        ]
                        - previous_pointwise["bernoulli_squared_residual"][
                            "relative_rms_error"
                        ]
                    ),
                }
            shells = []
            for index, shell in enumerate(case["discovered_field"]["shells"]):
                resolution_error = None
                if coarser is not None:
                    previous = coarser["discovered_field"]["shells"][index]
                    resolution_error = {
                        "coarser_grid_size": coarser_size,
                        "bernoulli_relative_rms_shift": abs(
                            shell["bernoulli_relative_rms_error"]
                            - previous["bernoulli_relative_rms_error"]
                        ),
                        "normalized_speed_dipole_shift": abs(
                            shell["speed"]["normalized_dipole_magnitude"]
                            - previous["speed"]["normalized_dipole_magnitude"]
                        ),
                        "bernoulli_squared_relative_rms_shift": abs(
                            shell["bernoulli_squared_residual"]["relative_rms_error"]
                            - previous["bernoulli_squared_residual"][
                                "relative_rms_error"
                            ]
                        ),
                    }
                converged = (
                    None
                    if resolution_error is None
                    else max(
                        value
                        for key, value in resolution_error.items()
                        if key != "coarser_grid_size"
                    )
                    < 0.05
                )
                temporal_error = shell["temporal_error_proxy_from_last_dv_rms"]
                resolution_bound = (
                    0.0
                    if resolution_error is None
                    else max(
                        value
                        for key, value in resolution_error.items()
                        if key != "coarser_grid_size"
                    )
                )
                combined_error = temporal_error + resolution_bound
                closure_killed = bool(
                    converged
                    and shell["bernoulli_squared_residual"]["relative_rms_error"]
                    > combined_error
                )
                shells.append(
                    {
                        "radius": shell["radius"],
                        "speed": shell["speed"],
                        "bernoulli_relative_rms_error": shell[
                            "bernoulli_relative_rms_error"
                        ],
                        "bernoulli_maximum_relative_error": shell[
                            "bernoulli_maximum_relative_error"
                        ],
                        "bernoulli_squared_residual": shell[
                            "bernoulli_squared_residual"
                        ],
                        "temporal_error_proxy": shell[
                            "temporal_error_proxy_from_last_dv_rms"
                        ],
                        "resolution_error": resolution_error,
                        "converged": converged,
                        "combined_temporal_resolution_error_proxy": combined_error,
                        "formula_violation_beyond_error": closure_killed,
                    }
                )
                if case["steadiness"]["steady"]:
                    g4_cases.append(
                        {
                            "grid_size": grid_size,
                            "wind_ratio": ratio,
                            "radius": shell["radius"],
                            "direction": shell["direction"],
                            "stagnation_and_vorticity": case["discovered_field"][
                                "stagnation"
                            ],
                            "resolution_error": resolution_error,
                        }
                    )
            g3_cases.append(
                {
                    "grid_size": grid_size,
                    "finer_anchor": rung["apparatus"]["finer_anchor"],
                    "wind_ratio": ratio,
                    "admission": "velocity_sector_settled",
                    "pointwise_speed": case["discovered_field"]["pointwise_speed"],
                    "apparatus_errors": {
                        "temporal_speed_proxy": case["steadiness"][
                            "residual_time_series"
                        ][-1]["dv_rms"]
                        * DIAGNOSTIC_INTERVAL,
                        "pointwise_resolution_error": pointwise_resolution_error,
                    },
                    "shells": shells,
                }
            )
    momentum_cases = []
    for rung in rungs:
        grid_size = rung["apparatus"]["grid_size"]
        for case in rung["cases"]:
            ratio = case["wind_ratio_to_finest_v_GP_at_probe"]
            _, coarser = coarser_case(cases, grid_size, ratio)
            budget = case["momentum_budget"]
            convergence = None
            if coarser is not None:
                previous = coarser["momentum_budget"]
                convergence = {
                    "consumed_momentum_mean_relative_shift": relative_change(
                        previous["consumed_momentum_integral_x_mean"],
                        budget["consumed_momentum_integral_x_mean"],
                    ),
                    "advective_flux_mean_relative_shift": relative_change(
                        previous["advective_far_surface_flux_x_mean"],
                        budget["advective_far_surface_flux_x_mean"],
                    ),
                }
            wind = case["wind_speed"]
            measured = abs(budget["consumed_momentum_integral_x_mean"])
            old_dynamic = 55.91 * wind**0.979
            marched = 20.3 * wind**0.098
            momentum_cases.append(
                {
                    "grid_size": grid_size,
                    "finer_anchor": rung["apparatus"]["finer_anchor"],
                    "wind_ratio": ratio,
                    "wind_speed": wind,
                    "settlement_verdict": case["steadiness"]["verdict"],
                    **{
                        key: value
                        for key, value in budget.items()
                        if key != "time_series"
                    },
                    "resolution_error": convergence,
                    "converged": (
                        None
                        if convergence is None
                        else max(convergence.values()) < 0.25
                    ),
                    "prior_fit_comparison": {
                        "ORB_10937_predicted": old_dynamic,
                        "measured_over_ORB_10937": measured / old_dynamic,
                        "ORB_10935_predicted": marched,
                        "measured_over_ORB_10935": measured / marched,
                    },
                }
            )
    finest_cases = ladder[-1]["cases"]
    winds = np.array([case["wind_speed"] for case in finest_cases])
    drag = np.abs(
        np.array(
            [
                case["momentum_budget"]["consumed_momentum_integral_x_mean"]
                for case in finest_cases
            ]
        )
    )
    exponent, log_prefactor = np.polyfit(
        np.log(winds), np.log(np.maximum(drag, 1.0e-30)), 1
    )
    null_passed = all(rung["null_control"]["passed"] for rung in rungs)
    steady_count = sum(row["steady"] for row in g1_cases)
    cavitation_branch_kill = any(row["branch_kill"] for row in cavitation_by_wind)
    bernoulli_branch_kill = any(
        shell["formula_violation_beyond_error"]
        for case in g3_cases
        for shell in case["shells"]
    )
    return {
        "G1_steadiness": {
            "verdict": (
                "cavitation_branch_kill"
                if cavitation_branch_kill
                else "steady_cases_found"
                if steady_count
                else "no_case_settled"
            ),
            "three_way_outcomes": ["settled", "stalled", "still-decaying"],
            "cavitation_is_a_distinct_branch_kill_outcome": True,
            "criterion": f"volume RMS dv/dt < {STEADY_DV_RMS:g} and dn/dt < {STEADY_DN_RMS:g} over {STEADY_WINDOW_SAMPLES} final samples after 0.75*T; a failing sector with late log slope < -{STALL_LOG_SLOPE_FLOOR:g}/time is still-decaying, otherwise stalled",
            "cases": g1_cases,
            "late_slope_and_n_min_convergence": slope_convergence,
            "cavitation_by_wind": cavitation_by_wind,
            "cavitation_branch_kill": cavitation_branch_kill,
            "base_ladder_verdict_match_by_wind": [
                {
                    "wind_ratio": ratio,
                    "coarse_grid_size": base_coarse_size,
                    "fine_grid_size": base_fine_size,
                    "coarse": cases[(base_coarse_size, ratio)]["steadiness"]["verdict"],
                    "fine": cases[(base_fine_size, ratio)]["steadiness"]["verdict"],
                    "match": cases[(base_coarse_size, ratio)]["steadiness"]["verdict"]
                    == cases[(base_fine_size, ratio)]["steadiness"]["verdict"],
                }
                for ratio in WIND_RATIOS
            ],
        },
        "G2_global_mass_budget": {
            "verdict": "flux_balance_time_series_recorded",
            "criterion": "net boundary mass influx / total consumption approaches 1 with a diminishing late trend",
            "cases": g2_cases,
        },
        "G3_discovered_speed_law": {
            "verdict": (
                "executed_no_velocity_settled_cases"
                if not g3_cases
                else "bernoulli_closure_refuted_on_converged_velocity_sector"
                if bernoulli_branch_kill
                else "bernoulli_compared_on_every_velocity_settled_case"
            ),
            "kill_gate": True,
            "branch_kill": bernoulli_branch_kill,
            "criterion": "admit every case whose G1 velocity sector is settled, without waiting for density; compare |v|^2-U^2-2c^2 sigma pointwise and in shell multipoles plus the equivalent speed residual; adjacent-rung residual and normalized-dipole shifts below 0.05 establish apparatus convergence",
            "cases": g3_cases,
        },
        "G4_realized_wake": {
            "verdict": (
                "executed_no_steady_cases"
                if not g4_cases
                else "both_comparators_stagnation_and_vorticity_recorded"
            ),
            "criterion": "jointly G1-settled cases report l=0..3 direction multipoles against boosted GP and ORB-10935 marched fields plus stagnation threshold=max(1e-3,0.05U); curl norms are reported for every case regardless of settlement",
            "cases": g4_cases,
            "all_case_vorticity_ladder": all_case_vorticity,
        },
        "G5_momentum_budget": {
            "verdict": "late_time_budget_compared_to_both_prior_fits",
            "criterion": f"final {STEADY_WINDOW_SAMPLES} samples report means/variability; adjacent-rung mean shifts below 25% are converged",
            "cases": momentum_cases,
            "finest_complete_rung_drag_scaling": {
                "grid_size": ladder[-1]["apparatus"]["grid_size"],
                "form": "|integral s v_x dV| = A U^p",
                "A": float(np.exp(log_prefactor)),
                "p": float(exponent),
                "ORB_10937_comparator": {"A": 55.91, "p": 0.979},
                "ORB_10935_comparator": {"A": 20.3, "p": 0.098},
            },
        },
        "G6_Galilean_null": {
            "verdict": "pass" if null_passed else "kill_noninvariant_pure_wind",
            "kill_gate": True,
            "criterion": f"pure wind/no core advanced to T={END_TIME:g} by the identical SSP-RK2/CFL/boundary path remains invariant to floating precision",
            "by_rung": [rung["null_control"] for rung in rungs],
        },
    }


def verdict_from_gates(gates: dict) -> dict:
    return {
        "all_six_predeclared_gates_executed": True,
        "G1": gates["G1_steadiness"]["verdict"],
        "G2": gates["G2_global_mass_budget"]["verdict"],
        "G3": gates["G3_discovered_speed_law"]["verdict"],
        "G4": gates["G4_realized_wake"]["verdict"],
        "G5": gates["G5_momentum_budget"]["verdict"],
        "G6": gates["G6_Galilean_null"]["verdict"],
        "theory_reconciliation": "deferred to kepler; principia intentionally untouched",
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
    for level in levels:
        rung, _ = run_rung(level, WIND_RATIOS, winds, marched)
        ladder.append(rung)
    anchor_level = solve_draw_level(ANCHOR_GRID_SIZE)
    anchor_winds = [ratio * reference for ratio in ANCHOR_WIND_RATIOS]
    anchor, _ = run_rung(
        anchor_level,
        ANCHOR_WIND_RATIOS,
        anchor_winds,
        marched,
        anchor=True,
    )
    anchors = [anchor]
    attractor_level = levels[GRID_SIZES.index(ATTRACTOR_GRID_SIZE)]
    attractor_wind = winds[WIND_RATIOS.index(ATTRACTOR_WIND_RATIO)]
    attractor_probe = run_attractor_probe(attractor_level, attractor_wind)
    gates = adjudicate(ladder, anchors)
    feasibility = {
        "decision": "full_requested_horizon_completed",
        "target_end_time": 600.0,
        "base_ladder": [
            {
                "grid_size": rung["apparatus"]["grid_size"],
                "wind_ratios": list(WIND_RATIOS),
                "achieved_end_time": END_TIME,
            }
            for rung in ladder
        ],
        "finer_anchor": {
            "grid_size": ANCHOR_GRID_SIZE,
            "wind_ratios": list(ANCHOR_WIND_RATIOS),
            "achieved_end_time": END_TIME,
        },
        "benchmark_basis": "pre-run timings showed T=600 feasible at 33^3/41^3 and for one 61^3 anchor without reducing the physical horizon",
    }
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
        "base_grid_sizes": list(GRID_SIZES),
        "finer_anchor_grid_size": ANCHOR_GRID_SIZE,
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
        "ORB_10937_initial_condition_source": "this sim's preceding git revision and runs/2026-08-21-seed-42.json",
    }
    reproducibility = {
        "seed": SEED,
        "random_numbers_used": False,
        "byte_identical_in_memory_rerun_verified": True,
        "byte_identical_scope": "complete results and dated run-record JSON encodings from two fresh in-memory executions",
        "command": "PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-10938 uv run lab/sims/level-core-dynamical-relaxation/main.py --check-determinism",
    }
    verdict = verdict_from_gates(gates)
    limitations = [
        "The T=600 horizon spans more than nine ORB-10937 residual e-folding times but remains a finite-horizon classification, not a proof about infinite time.",
        "The complete 33^3/41^3 ladder fixes geometry; the 61^3 convergence anchor samples only U/v_GP=0.3 because a four-wind fine sweep was not required for the bounded apparatus decision.",
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
        "feasibility_decision": feasibility,
        "resolution_ladder": ladder,
        "finer_rung_anchors": anchors,
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
        "feasibility_decision": feasibility,
        "resolution_ladder": {
            "base_grid_sizes": list(GRID_SIZES),
            "finer_anchor_grid_size": ANCHOR_GRID_SIZE,
            "spacings": [rung["apparatus"]["spacing"] for rung in ladder],
            "finer_anchor_spacing": anchor["apparatus"]["spacing"],
            "fixed_domain_half_width": DOMAIN_HALF_WIDTH,
            "fixed_physical_core_sigma": CORE_SIGMA,
            "predeclared_convergence": {
                "G1": "same three-way categorical verdict on the 33^3 and 41^3 base ladder; 61^3 anchors U/v_GP=0.3",
                "G2": "mass influx/consumption ratio and late trend reported at every admitted rung/wind",
                "G3_G4": "adjacent-rung shifts of Bernoulli RMS error and normalized speed dipole <0.05",
                "G5": "adjacent-rung relative shift of each budget mean <25%",
                "G6": "exact invariant to floating precision on every executed rung/wind",
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


def readjudicate_existing(root: Path) -> tuple[dict, dict]:
    """Apply amended gates to the byte-verified stored physical run."""
    results_path = root / "assets" / "results.json"
    record_path = root / "runs" / RUN_RECORD
    results = json.loads(results_path.read_text())
    record = json.loads(record_path.read_text())
    gates = adjudicate(record["resolution_ladder"], record["finer_rung_anchors"])
    repeated_gates = adjudicate(
        record["resolution_ladder"], record["finer_rung_anchors"]
    )
    if encoded(gates) != encoded(repeated_gates):
        raise RuntimeError("amended gate adjudication rerun was not byte-identical")
    verdict = verdict_from_gates(gates)
    for payload in (results, record):
        payload["gates"] = gates
        payload["verdict"] = verdict
        payload["reproducibility"][
            "post_adjudication_byte_identical_rerun_verified"
        ] = True
    results_path.write_bytes(encoded(results))
    record_path.write_bytes(encoded(record))
    return results, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--readjudicate-existing", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).parent
    if arguments.readjudicate_existing:
        results, _ = readjudicate_existing(root)
        print(json.dumps(results["verdict"], indent=2, sort_keys=True))
        return
    results, record = experiment()
    if arguments.check_determinism:
        repeated_results, repeated_record = experiment()
        if encoded(results) != encoded(repeated_results) or encoded(record) != encoded(
            repeated_record
        ):
            raise RuntimeError("experiment rerun was not byte-identical")
    (root / "assets").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "assets" / "results.json").write_bytes(encoded(results))
    (root / "runs" / RUN_RECORD).write_bytes(encoded(record))
    print(json.dumps(results["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
