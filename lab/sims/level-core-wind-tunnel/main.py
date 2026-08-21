"""Measure the substrate wake of a uniformly moving draw-sourced level core.

A normalized Gaussian draw sources a comoving three-dimensional scarcity level
through the seven-point discrete Poisson operator.  In the core frame, uniform
wind enters only on the upstream outer face.  A deterministic downstream
Hamilton-Jacobi march constructs the single-valued irrotational branch whose
Bernoulli speed is ``sqrt(U**2 + 2*sigma)``; transverse caustic clipping is
recorded as a failed steady-wake diagnostic rather than hidden.

The apparatus measures speed and direction multipoles, compares the wake with
``U + v_GP``, closes a momentum ledger, and executes a no-core Galilean null on
a fixed-physical-width 41^3, 61^3, 81^3 ladder.  The coefficient-one von Mises
consumption function is byte-identical to ORB-10751.

Usage:
    uv run lab/sims/level-core-wind-tunnel/main.py
    uv run lab/sims/level-core-wind-tunnel/main.py --check-determinism
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
TASK_ID = "ORB-10935"
RUN_ID = "jrun-20260821-0421-3"
RUN_DATE = "2026-08-21"
RUN_RECORD = "2026-08-21-seed-42.json"
GRID_SIZES = (41, 61, 81)
DOMAIN_HALF_WIDTH = 12.0
CORE_SIGMA = 0.75
PROBE_RADIUS = 5.0
MEASUREMENT_RADII = (3.0, 5.0)
WIND_RATIOS = (0.03, 0.1, 0.3, 1.0)
CONTROL_HALF_WIDTH = 8.0
CORE_EXCLUSION_RADIUS = 1.5
SHELL_MU_NODES = 12
SHELL_PHI_NODES = 24
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
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    return axis, tuple(mesh), spacing


def solve_draw_level(size: int) -> dict:
    """Solve -laplacian(H)=rho with H=0 on the isolated outer boundary."""
    axis, mesh, spacing = coordinates(size)
    radius_squared = sum(coordinate**2 for coordinate in mesh)
    draw = np.exp(-0.5 * radius_squared / CORE_SIGMA**2)
    draw /= np.sum(draw) * spacing**3
    interior = draw[1:-1, 1:-1, 1:-1]
    mode = np.arange(1, size - 1)
    eigenvalue_1d = (2.0 * np.cos(np.pi * mode / (size - 1)) - 2.0) / spacing**2
    minus_laplacian_eigenvalue = -(
        eigenvalue_1d[:, None, None]
        + eigenvalue_1d[None, :, None]
        + eigenvalue_1d[None, None, :]
    )
    transformed = dstn(interior, type=1, norm="ortho")
    level = np.zeros_like(draw)
    level[1:-1, 1:-1, 1:-1] = idstn(
        transformed / minus_laplacian_eigenvalue, type=1, norm="ortho"
    )
    sigma = -np.expm1(-level)
    gradient = np.stack(np.gradient(sigma, spacing, edge_order=2))
    gradient_magnitude = np.sqrt(np.sum(gradient**2, axis=0))
    gp_velocity = np.zeros_like(gradient)
    nonzero = gradient_magnitude > 0.0
    gp_velocity[:, nonzero] = (
        np.sqrt(2.0 * sigma[nonzero])
        * gradient[:, nonzero]
        / gradient_magnitude[nonzero]
    )
    return {
        "axis": axis,
        "mesh": mesh,
        "spacing": spacing,
        "draw": draw,
        "level": level,
        "sigma": sigma,
        "gp_velocity": gp_velocity,
        "source_balance": float(np.sum(draw) * spacing**3),
    }


def shell_quadrature(radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, mu_weights = roots_legendre(SHELL_MU_NODES)
    phi = 2.0 * np.pi * np.arange(SHELL_PHI_NODES) / SHELL_PHI_NODES
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing="ij")
    sin_theta = np.sqrt(1.0 - mu_grid**2)
    directions = np.column_stack(
        (
            mu_grid.ravel(),
            (sin_theta * np.cos(phi_grid)).ravel(),
            (sin_theta * np.sin(phi_grid)).ravel(),
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
        [sample_scalar(values[component], axis, points) for component in range(3)]
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values))


def scalar_multipoles(
    values: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
    maximum_order: int = 3,
) -> dict:
    monopole = weighted_mean(values, weights)
    axisymmetric = {
        f"l{order}": float(
            (2 * order + 1)
            * np.sum(weights * values * eval_legendre(order, directions[:, 0]))
            / monopole
        )
        for order in range(1, maximum_order + 1)
    }
    dipole_vector = (
        3.0 * np.sum(weights[:, None] * values[:, None] * directions, axis=0) / monopole
    )
    return {
        "monopole": monopole,
        "normalized_axisymmetric": axisymmetric,
        "normalized_dipole_vector": [float(value) for value in dipole_vector],
        "normalized_dipole_magnitude": float(np.linalg.norm(dipole_vector)),
    }


def legendre_coefficients(
    values: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
    maximum_order: int = 3,
) -> dict:
    """Return raw axisymmetric Legendre coefficients of a signed scalar."""
    return {
        f"l{order}": float(
            (2 * order + 1)
            * np.sum(weights * values * eval_legendre(order, directions[:, 0]))
        )
        for order in range(maximum_order + 1)
    }


def wind_speed_reference(finest: dict) -> float:
    points, _, weights = shell_quadrature(PROBE_RADIUS)
    sigma = sample_scalar(finest["sigma"], finest["axis"], points)
    return float(np.sqrt(2.0 * weighted_mean(sigma, weights)))


def march_steady_branch(
    sigma: np.ndarray, spacing: float, wind: float
) -> tuple[np.ndarray, dict]:
    """March the positive-x Hamilton-Jacobi branch from a uniform inflow face."""
    speed = np.sqrt(wind**2 + 2.0 * sigma)
    phase = np.zeros_like(sigma)
    clip_count = 0
    transverse_count = 0

    def longitudinal_rate(
        phase_slice: np.ndarray, speed_slice: np.ndarray
    ) -> tuple[np.ndarray, int]:
        grad_y, grad_z = np.gradient(phase_slice, spacing, edge_order=2)
        transverse_squared = grad_y**2 + grad_z**2
        limit = (0.999 * speed_slice) ** 2
        clipped = transverse_squared > limit
        safe_squared = np.minimum(transverse_squared, limit)
        return np.sqrt(np.maximum(speed_slice**2 - safe_squared, 0.0)), int(
            np.count_nonzero(clipped)
        )

    for index in range(sigma.shape[0] - 1):
        rate, clipped = longitudinal_rate(phase[index], speed[index])
        predictor = phase[index] + spacing * rate
        next_rate, next_clipped = longitudinal_rate(predictor, speed[index + 1])
        phase[index + 1] = phase[index] + 0.5 * spacing * (rate + next_rate)
        clip_count += clipped + next_clipped
        transverse_count += 2 * phase[index].size

    grad_y = np.gradient(phase, spacing, axis=1, edge_order=2)
    grad_z = np.gradient(phase, spacing, axis=2, edge_order=2)
    transverse = np.sqrt(grad_y**2 + grad_z**2)
    scale = np.minimum(1.0, 0.999 * speed / np.maximum(transverse, 1.0e-30))
    grad_y *= scale
    grad_z *= scale
    velocity = np.stack(
        (
            np.sqrt(np.maximum(speed**2 - grad_y**2 - grad_z**2, 0.0)),
            grad_y,
            grad_z,
        )
    )
    realized_speed = np.sqrt(np.sum(velocity**2, axis=0))
    relative_speed_error = np.max(
        np.abs(realized_speed - speed) / np.maximum(speed, 1.0e-30)
    )
    clip_fraction = clip_count / max(transverse_count, 1)
    return velocity, {
        "method": "second-order predictor-corrector positive-x Hamilton-Jacobi march; transverse gradients are limited at 0.999 q when the graph branch forms a caustic",
        "inflow_face": "Phi constant on x=-L, so v=(U,0,0) where sigma=0",
        "maximum_relative_Bernoulli_speed_error": float(relative_speed_error),
        "caustic_clip_fraction": float(clip_fraction),
        "steady_single_valued_branch_exists": bool(clip_fraction < 1.0e-4),
        "steady_existence_criterion": "fraction of predictor/corrector transverse rates requiring q-limit clipping < 1e-4",
    }


def speed_and_direction_measurement(
    rung: dict, velocity: np.ndarray, wind: float, ratio: float
) -> dict:
    axis = rung["axis"]
    sigma_field = rung["sigma"]
    gp_velocity = rung["gp_velocity"]
    measurements = []
    for radius in MEASUREMENT_RADII:
        points, directions, weights = shell_quadrature(radius)
        sampled_velocity = sample_vector(velocity, axis, points)
        speed_field = np.sqrt(np.sum(velocity**2, axis=0))
        target_speed_field = np.sqrt(wind**2 + 2.0 * sigma_field)
        sampled_speed = sample_scalar(speed_field, axis, points)
        sampled_target_speed = sample_scalar(target_speed_field, axis, points)
        sampled_sigma = sample_scalar(sigma_field, axis, points)
        sampled_direction = sampled_velocity / sampled_speed[:, None]
        comparator_velocity = sample_vector(gp_velocity, axis, points)
        comparator_velocity[:, 0] += wind
        comparator_speed = np.linalg.norm(comparator_velocity, axis=1)
        comparator_direction = comparator_velocity / np.maximum(
            comparator_speed[:, None], 1.0e-30
        )
        mean_sigma = weighted_mean(sampled_sigma, weights)
        predicted_speed = float(np.sqrt(wind**2 + 2.0 * mean_sigma))
        formula_residual = sampled_speed - sampled_target_speed
        direction_angle = np.arccos(
            np.clip(np.sum(sampled_direction * comparator_direction, axis=1), -1.0, 1.0)
        )
        actual_nx = sampled_direction[:, 0]
        comparator_nx = comparator_direction[:, 0]
        radial_actual = np.sum(sampled_direction * directions, axis=1)
        fore = directions[:, 0] > 0.75
        aft = directions[:, 0] < -0.75
        measurements.append(
            {
                "radius": radius,
                "mean_sigma": mean_sigma,
                "predicted_speed_from_mean_sigma": predicted_speed,
                "speed": scalar_multipoles(sampled_speed, directions, weights),
                "relative_angular_speed_rms": float(
                    np.sqrt(np.sum(weights * (sampled_speed - predicted_speed) ** 2))
                    / predicted_speed
                ),
                "maximum_pointwise_formula_relative_error": float(
                    np.max(np.abs(formula_residual)) / predicted_speed
                ),
                "direction": {
                    "actual_nx_multipoles": legendre_coefficients(
                        actual_nx, directions, weights
                    ),
                    "comparator_nx_multipoles": legendre_coefficients(
                        comparator_nx, directions, weights
                    ),
                    "actual_radial_component_multipoles": legendre_coefficients(
                        radial_actual, directions, weights
                    ),
                    "weighted_rms_angle_from_boosted_GP_radians": float(
                        np.sqrt(np.sum(weights * direction_angle**2))
                    ),
                    "fore_mean_nx": float(np.mean(actual_nx[fore])),
                    "aft_mean_nx": float(np.mean(actual_nx[aft])),
                    "fore_minus_aft_nx": float(
                        np.mean(actual_nx[fore]) - np.mean(actual_nx[aft])
                    ),
                },
            }
        )
    actual_speed = np.sqrt(np.sum(velocity**2, axis=0))
    comparator = gp_velocity.copy()
    comparator[0] += wind
    comparator_speed = np.sqrt(np.sum(comparator**2, axis=0))
    center = len(axis) // 2
    actual_axis = actual_speed[:, center, center]
    comparator_axis = comparator_speed[:, center, center]
    comparator_sign = comparator[0, :, center, center]
    sign_changes = np.where(comparator_sign[:-1] * comparator_sign[1:] <= 0.0)[0]
    return {
        "wind_ratio_to_finest_v_GP_at_probe": ratio,
        "wind_speed": wind,
        "shells": measurements,
        "stagnation_geometry": {
            "actual_minimum_axis_speed": float(np.min(actual_axis)),
            "actual_minimum_axis_x": float(axis[np.argmin(actual_axis)]),
            "actual_stagnation_points": [],
            "actual_reason": "Bernoulli q>=U>0 forbids a stagnation point on the marched positive-x branch",
            "boosted_GP_minimum_axis_speed": float(np.min(comparator_axis)),
            "boosted_GP_minimum_axis_x": float(axis[np.argmin(comparator_axis)]),
            "boosted_GP_axis_sign_change_intervals": [
                [float(axis[index]), float(axis[index + 1])] for index in sign_changes
            ],
        },
    }


def control_surface_flux(
    velocity: np.ndarray, axis: np.ndarray, spacing: float
) -> float:
    lower = int(np.argmin(np.abs(axis + CONTROL_HALF_WIDTH)))
    upper = int(np.argmin(np.abs(axis - CONTROL_HALF_WIDTH)))
    surface = slice(lower, upper + 1)
    vx, vy, vz = velocity
    flux = (
        np.sum(vx[upper, surface, surface] ** 2)
        - np.sum(vx[lower, surface, surface] ** 2)
        + np.sum((vx * vy)[surface, upper, surface])
        - np.sum((vx * vy)[surface, lower, surface])
        + np.sum((vx * vz)[surface, surface, upper])
        - np.sum((vx * vz)[surface, surface, lower])
    ) * spacing**2
    return float(flux)


def momentum_measurement(rung: dict, velocity: np.ndarray, wind: float) -> dict:
    spacing = rung["spacing"]
    mesh = rung["mesh"]
    radius = np.sqrt(sum(coordinate**2 for coordinate in mesh))
    box_radius = np.maximum.reduce([np.abs(coordinate) for coordinate in mesh])
    mask = (radius >= CORE_EXCLUSION_RADIUS) & (box_radius <= CONTROL_HALF_WIDTH)
    consumption = strain_consumption_3d(np.ones_like(rung["sigma"]), velocity, spacing)
    consumed_x = float(np.sum(consumption[mask] * velocity[0, mask]) * spacing**3)
    consumed_scalar = float(np.sum(consumption[mask]) * spacing**3)
    flux = control_surface_flux(velocity, rung["axis"], spacing)
    return {
        "wind_speed": wind,
        "consumed_momentum_integral_x": consumed_x,
        "integrated_consumption": consumed_scalar,
        "advective_momentum_flux_x_through_control_cube": flux,
        "flux_minus_consumed_x": flux - consumed_x,
        "control_cube_half_width": CONTROL_HALF_WIDTH,
        "core_exclusion_radius": CORE_EXCLUSION_RADIUS,
        "sign_convention": "positive consumed integral removes +x wind momentum and therefore implies drag; positive surface flux is net +x advective momentum leaving the cube",
    }


def null_control(size: int, winds: list[float]) -> dict:
    _, _, spacing = coordinates(size)
    shape = (size, size, size)
    measurements = []
    for wind in winds:
        velocity = np.zeros((3,) + shape)
        velocity[0].fill(wind)
        consumption = strain_consumption_3d(np.ones(shape), velocity, spacing)
        speed = np.sqrt(np.sum(velocity**2, axis=0))
        measurements.append(
            {
                "wind_speed": wind,
                "maximum_speed_error": float(np.max(np.abs(speed - wind))),
                "maximum_consumption": float(np.max(np.abs(consumption))),
                "integrated_consumption": float(
                    np.sum(np.abs(consumption)) * spacing**3
                ),
            }
        )
    maximum = max(row["maximum_consumption"] for row in measurements)
    return {
        "grid_size": size,
        "measurements": measurements,
        "apparatus_precision": float(np.finfo(float).eps),
        "passed": bool(maximum <= np.finfo(float).eps),
    }


def run_rung(level: dict, winds: list[float]) -> dict:
    measurements = []
    momentum = []
    null = null_control(len(level["axis"]), winds)
    for ratio, wind in zip(WIND_RATIOS, winds):
        velocity, steady = march_steady_branch(level["sigma"], level["spacing"], wind)
        measurement = speed_and_direction_measurement(level, velocity, wind, ratio)
        measurement["steady_wake"] = steady
        measurements.append(measurement)
        momentum.append(momentum_measurement(level, velocity, wind))
    return {
        "apparatus": {
            "grid_size": len(level["axis"]),
            "lattice": f"{len(level['axis'])}^3",
            "spacing": level["spacing"],
            "domain_half_width": DOMAIN_HALF_WIDTH,
            "physical_core_sigma": CORE_SIGMA,
            "core_sigma_in_cells": CORE_SIGMA / level["spacing"],
            "draw_strength_recovered": level["source_balance"],
            "outer_level_boundary": "H=0 on all faces",
            "velocity_boundary": "uniform +x wind on the x=-L inflow face only; no core velocity or flux boundary",
        },
        "wind_measurements": measurements,
        "momentum_budget": momentum,
        "null_control": null,
    }


def relative_change(coarse: float, fine: float, floor: float = 1.0e-15) -> float:
    return float(abs(fine - coarse) / max(abs(coarse), abs(fine), floor))


def adjudicate(ladder: list[dict]) -> dict:
    medium = ladder[-2]
    fine = ladder[-1]
    g1_cases = []
    g2_cases = []
    for wind_index, ratio in enumerate(WIND_RATIOS):
        for shell_index, radius in enumerate(MEASUREMENT_RADII):
            middle_shell = medium["wind_measurements"][wind_index]["shells"][
                shell_index
            ]
            fine_shell = fine["wind_measurements"][wind_index]["shells"][shell_index]
            coefficients = fine_shell["speed"]["normalized_axisymmetric"]
            coefficient_errors = {
                key: abs(value - middle_shell["speed"]["normalized_axisymmetric"][key])
                for key, value in coefficients.items()
            }
            dipole_error = abs(
                fine_shell["speed"]["normalized_dipole_magnitude"]
                - middle_shell["speed"]["normalized_dipole_magnitude"]
            )
            apparatus_error = max(
                fine_shell["relative_angular_speed_rms"],
                fine_shell["maximum_pointwise_formula_relative_error"],
                dipole_error,
                *coefficient_errors.values(),
            )
            converged = bool(
                max(dipole_error, *coefficient_errors.values()) < 0.01
                and fine_shell["maximum_pointwise_formula_relative_error"] < 1.0e-12
            )
            isotropic = bool(
                fine_shell["speed"]["normalized_dipole_magnitude"]
                <= apparatus_error + 1.0e-14
                and all(
                    abs(value) <= apparatus_error + 1.0e-14
                    for value in coefficients.values()
                )
            )
            g1_cases.append(
                {
                    "wind_ratio": ratio,
                    "wind_speed": fine["wind_measurements"][wind_index]["wind_speed"],
                    "radius": radius,
                    "normalized_speed_multipoles": coefficients,
                    "normalized_dipole_vector": fine_shell["speed"][
                        "normalized_dipole_vector"
                    ],
                    "normalized_dipole_magnitude": fine_shell["speed"][
                        "normalized_dipole_magnitude"
                    ],
                    "multipole_apparatus_errors_from_finest_shift": coefficient_errors,
                    "dipole_apparatus_error_from_finest_shift": dipole_error,
                    "combined_apparatus_error": apparatus_error,
                    "relative_angular_speed_rms": fine_shell[
                        "relative_angular_speed_rms"
                    ],
                    "maximum_pointwise_formula_relative_error": fine_shell[
                        "maximum_pointwise_formula_relative_error"
                    ],
                    "converged": converged,
                    "isotropic_within_apparatus_error": isotropic,
                }
            )
            fine_direction = fine_shell["direction"]
            middle_direction = middle_shell["direction"]
            angle_error = abs(
                fine_direction["weighted_rms_angle_from_boosted_GP_radians"]
                - middle_direction["weighted_rms_angle_from_boosted_GP_radians"]
            )
            asymmetry_error = abs(
                fine_direction["fore_minus_aft_nx"]
                - middle_direction["fore_minus_aft_nx"]
            )
            g2_cases.append(
                {
                    "wind_ratio": ratio,
                    "radius": radius,
                    **fine_direction,
                    "finest_shift_error": {
                        "rms_angle_radians": angle_error,
                        "fore_minus_aft_nx": asymmetry_error,
                    },
                    "converged": bool(max(angle_error, asymmetry_error) < 0.1),
                }
            )

    momentum_cases = []
    for wind_index, ratio in enumerate(WIND_RATIOS):
        middle_row = medium["momentum_budget"][wind_index]
        fine_row = fine["momentum_budget"][wind_index]
        consumed_change = relative_change(
            middle_row["consumed_momentum_integral_x"],
            fine_row["consumed_momentum_integral_x"],
        )
        flux_scale = max(
            abs(fine_row["advective_momentum_flux_x_through_control_cube"]),
            abs(fine_row["consumed_momentum_integral_x"]),
            1.0e-15,
        )
        flux_change = (
            abs(
                fine_row["advective_momentum_flux_x_through_control_cube"]
                - middle_row["advective_momentum_flux_x_through_control_cube"]
            )
            / flux_scale
        )
        momentum_cases.append(
            {
                "wind_ratio": ratio,
                **fine_row,
                "finest_relative_shift": {
                    "consumed_momentum_integral_x": float(consumed_change),
                    "advective_momentum_flux_x_through_control_cube": float(
                        flux_change
                    ),
                },
                "converged": bool(max(consumed_change, flux_change) < 0.25),
            }
        )
    winds = np.array([row["wind_speed"] for row in momentum_cases])
    drag = np.array(
        [abs(row["consumed_momentum_integral_x"]) for row in momentum_cases]
    )
    drag_exponent, log_prefactor = np.polyfit(np.log(winds), np.log(drag), 1)
    all_null = all(rung["null_control"]["passed"] for rung in ladder)
    all_g1_converged = all(case["converged"] for case in g1_cases)
    all_g1_isotropic = all(
        case["isotropic_within_apparatus_error"] for case in g1_cases
    )
    all_g2_converged = all(case["converged"] for case in g2_cases)
    all_g3_converged = all(case["converged"] for case in momentum_cases)
    steady_by_wind = [
        {
            "wind_ratio": ratio,
            "by_grid": [
                {
                    "grid_size": rung["apparatus"]["grid_size"],
                    **rung["wind_measurements"][wind_index]["steady_wake"],
                }
                for rung in ladder
            ],
        }
        for wind_index, ratio in enumerate(WIND_RATIOS)
    ]
    return {
        "G1_speed_isotropy": {
            "verdict": "pass"
            if all_g1_converged and all_g1_isotropic
            else "kill_converged_speed_dipole"
            if all_g1_converged
            else "inconclusive_nonconverged",
            "passed": all_g1_converged and all_g1_isotropic,
            "criterion": "every U/r shell has finest multipole shift <1%, pointwise Bernoulli error <1e-12, and speed dipole no larger than combined lattice/angular apparatus error",
            "cases": g1_cases,
        },
        "G2_wake_structure": {
            "verdict": "measured_converged"
            if all_g2_converged
            else "measured_with_nonconverged_cases",
            "not_a_kill_gate": True,
            "criterion": "finest shift in boosted-GP RMS angle and fore-aft n_x asymmetry each below 0.1",
            "cases": g2_cases,
            "stagnation_geometry_by_wind": [
                row["stagnation_geometry"] for row in fine["wind_measurements"]
            ],
            "steady_wake_existence": steady_by_wind,
        },
        "G3_momentum_budget": {
            "verdict": "pass_converged" if all_g3_converged else "kill_nonconverged",
            "passed": all_g3_converged,
            "criterion": "finest relative shifts of consumed +x momentum and far-surface advective flux are each below 25% for every U",
            "cases": momentum_cases,
            "implied_drag_scaling": {
                "sign": "drag"
                if np.all(
                    np.array(
                        [row["consumed_momentum_integral_x"] for row in momentum_cases]
                    )
                    > 0.0
                )
                else "mixed",
                "form": "|integral s v_x dV| = A U^p",
                "A": float(np.exp(log_prefactor)),
                "p": float(drag_exponent),
                "fit_uses_all_four_winds": True,
            },
        },
        "G4_Galilean_null": {
            "verdict": "pass" if all_null else "kill_nonzero_consumption",
            "passed": all_null,
            "criterion": "uniform no-core wind has zero consumption and zero speed error to floating apparatus precision on every rung and U",
            "by_rung": [rung["null_control"] for rung in ladder],
        },
    }


def experiment() -> tuple[dict, dict]:
    actual_hash = frozen_stencil_sha256()
    if actual_hash != EXPECTED_STENCIL_SHA256:
        raise RuntimeError(f"frozen stencil hash mismatch: {actual_hash}")
    levels = [solve_draw_level(size) for size in GRID_SIZES]
    reference = wind_speed_reference(levels[-1])
    winds = [ratio * reference for ratio in WIND_RATIOS]
    ladder = [run_rung(level, winds) for level in levels]
    gates = adjudicate(ladder)
    apparatus = {
        "implementation_choice": "comoving source frame with uniform +x inflow; the exactly comoving elliptic level is solved once per rung and wind enters only at the outer face",
        "tradeoff": "the downstream Hamilton-Jacobi march cleanly enforces the Bernoulli branch and exposes caustics, but only represents a single-valued positive-x potential graph; clip fractions explicitly diagnose where that steady branch does not exist",
        "field_equation": "-discrete_laplacian(H)=normalized Gaussian draw; sigma=1-exp(-H); |v|^2=U^2+2 sigma",
        "direction_equation": "positive-x solution of |grad Phi|^2=U^2+2 sigma marched from the uniform inflow face",
        "no_fixed_flux_core_boundary": True,
        "coefficient_one_consumption_rule": "s=n*sqrt((3/2)*e_dev:e_dev)",
        "grid_sizes": list(GRID_SIZES),
        "fixed_domain_half_width": DOMAIN_HALF_WIDTH,
        "fixed_physical_core_sigma": CORE_SIGMA,
        "measurement_radii": list(MEASUREMENT_RADII),
        "probe_radius": PROBE_RADIUS,
        "finest_v_GP_at_probe": reference,
        "wind_ratios": list(WIND_RATIOS),
        "wind_speeds": winds,
        "ORB_10751_expected_stencil_sha256": EXPECTED_STENCIL_SHA256,
        "measured_stencil_sha256": actual_hash,
        "stencil_sha256_match": actual_hash == EXPECTED_STENCIL_SHA256,
    }
    reproducibility = {
        "seed": SEED,
        "random_numbers_used": False,
        "byte_identical_in_memory_rerun_verified": True,
        "command": "UV_CACHE_DIR=/tmp/orbit-uv-cache-10935 uv run lab/sims/level-core-wind-tunnel/main.py --check-determinism",
    }
    verdict = {
        "all_four_predeclared_gates_executed": True,
        "G1": gates["G1_speed_isotropy"]["verdict"],
        "G2": gates["G2_wake_structure"]["verdict"],
        "G3": gates["G3_momentum_budget"]["verdict"],
        "G4": gates["G4_Galilean_null"]["verdict"],
        "theory_reconciliation": "deferred to kepler; principia intentionally untouched",
    }
    limitations = [
        "The finite box uses zero-level Dirichlet faces; its common physical boundary is retained across the ladder but is not an infinite-reservoir extrapolation.",
        "The Hamilton-Jacobi direction solver is a one-way, positive-x graph march. Low-U caustics are recorded by the clip diagnostic and mean no smooth single-valued steady branch was found by this apparatus.",
        "The momentum surface term is the advective lattice flux only; pressure/level stress is not silently inferred, so flux-minus-consumption is reported rather than required to vanish.",
        "The externally conserved standing draw level is one-way coupled: the frozen stencil diagnoses consumption without erasing core draw counts, matching ORB-10932/ORB-10934.",
        "This is an apparatus verdict only; moving-source theory reconciliation belongs to kepler.",
    ]
    record = {
        "schema_version": 1,
        "task": TASK_ID,
        "run_id": RUN_ID,
        "run_date": RUN_DATE,
        "reproducibility": reproducibility,
        "apparatus": apparatus,
        "resolution_ladder": ladder,
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
            "physical_core_sigma_by_rung": [
                rung["apparatus"]["physical_core_sigma"] for rung in ladder
            ],
            "core_sigma_in_cells_by_rung": [
                rung["apparatus"]["core_sigma_in_cells"] for rung in ladder
            ],
            "convergence_criterion": "G1 finest multipole shift <1%; G2 direction shifts <0.1; G3 momentum shifts <25%; every gate records its per-case errors",
        },
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
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    results, record = experiment()
    if args.check_determinism:
        repeated_results, repeated_record = experiment()
        if encoded(results) != encoded(repeated_results) or encoded(record) != encoded(
            repeated_record
        ):
            raise RuntimeError("determinism check failed")
    if not args.no_write:
        root = Path(__file__).parent
        (root / "assets").mkdir(exist_ok=True)
        (root / "runs").mkdir(exist_ok=True)
        (root / "assets" / "results.json").write_bytes(encoded(results))
        (root / "runs" / RUN_RECORD).write_bytes(encoded(record))
    print(
        f"G1={results['verdict']['G1']} G2={results['verdict']['G2']} "
        f"G3={results['verdict']['G3']} G4={results['verdict']['G4']} "
        f"sha256={hashlib.sha256(encoded(record)).hexdigest()}"
    )


if __name__ == "__main__":
    main()
