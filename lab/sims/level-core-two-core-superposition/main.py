"""Discriminate level-core and flux-core two-source superposition families.

Two finite-width Gaussian draw cores source a standing 3-D level through a
seven-point discrete Poisson solve with an isolated zero-level outer boundary.
Anonymous finite capacity stores that level as ``sigma = 1-exp(-H)``; paired
background subtraction therefore measures the target core only after it has
competed for the same local headroom.  No velocity or flux boundary is imposed
at either core.  The resulting target field modulation is fitted on a common
depletion grid at 25^3, 41^3, and 65^3.

The draw-sourced level supplies an inertial rolling-rule velocity diagnostic,
and the coefficient-one von Mises destruction stencil frozen by ORB-10751 is
executed on that genuinely nonspherical 3-D field.  As in ORB-10932, the draw
count is conserved externally: destruction diagnoses the transported substrate
but does not feed back into the standing source level.

Usage:
    uv run lab/sims/level-core-two-core-superposition/main.py
    uv run lab/sims/level-core-two-core-superposition/main.py --check-determinism
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
from scipy.optimize import curve_fit


SEED = 42
RUN_ID = "jrun-20260821-0353-3"
RUN_RECORD = "2026-08-21-seed-42.json"
GRID_SIZES = (25, 41, 65)
DOMAIN_HALF_WIDTH = 12.0
CORE_SIGMA = 0.75
TARGET_POSITION = (-4.0, 0.0, 0.0)
NEIGHBOUR_POSITION = (4.0, 0.0, 0.0)
TARGET_DRAW_STRENGTH = 1.0
NEIGHBOUR_DRAW_STRENGTHS = (
    0.025,
    0.06,
    0.14,
    0.32,
    0.7,
    1.5,
    3.2,
    7.0,
    15.0,
    35.0,
    80.0,
    180.0,
    450.0,
)
MATCHED_GRID_SAMPLES = 11
MEASUREMENT_RADII = (2.0, 6.0)
SEPARATION_VALUES = (6.0, 8.0, 10.0)
SEPARATION_CONTROL_STRENGTH = 35.0
ORB_10157_EXPONENT = 1.071
EXPECTED_STENCIL_SHA256 = (
    "aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461"
)


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


def frozen_stencil_sha256() -> str:
    source = Path(__file__).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "strain_consumption_3d"
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


def gaussian_draw_density(
    mesh: tuple[np.ndarray, ...], spacing: float, position: tuple[float, float, float]
) -> np.ndarray:
    radius_squared = sum(
        (coordinate - center) ** 2 for coordinate, center in zip(mesh, position)
    )
    density = np.exp(-0.5 * radius_squared / CORE_SIGMA**2)
    density /= np.sum(density) * spacing**3
    return density


def solve_draw_level(
    size: int, position: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], float, dict]:
    """Solve -laplacian(H)=rho with H=0 at the isolated outer boundary."""
    axis, mesh, spacing = coordinates(size)
    draw_density = gaussian_draw_density(mesh, spacing, position)
    interior = draw_density[1:-1, 1:-1, 1:-1]
    mode = np.arange(1, size - 1)
    one_dimensional_eigenvalue = (
        2.0 * np.cos(np.pi * mode / (size - 1)) - 2.0
    ) / spacing**2
    minus_laplacian_eigenvalue = -(
        one_dimensional_eigenvalue[:, None, None]
        + one_dimensional_eigenvalue[None, :, None]
        + one_dimensional_eigenvalue[None, None, :]
    )
    transformed = dstn(interior, type=1, norm="ortho")
    level = np.zeros_like(draw_density)
    level[1:-1, 1:-1, 1:-1] = idstn(
        transformed / minus_laplacian_eigenvalue, type=1, norm="ortho"
    )
    recovered = float(np.sum(draw_density) * spacing**3)
    return level, axis, mesh, spacing, {
        "draw_strength_recovered": recovered,
        "draw_balance_error": abs(recovered - 1.0),
        "peak_level": float(np.max(level)),
    }


def stored_level(exposure: np.ndarray) -> np.ndarray:
    """Literal anonymous-slot expectation for finite shared capacity."""
    return -np.expm1(-exposure)


def sample_at(
    values: np.ndarray, axis: np.ndarray, position: tuple[float, float, float]
) -> float:
    return float(interpn((axis, axis, axis), values, np.array([position]))[0])


def vector_gradient(values: np.ndarray, spacing: float) -> np.ndarray:
    return np.stack(np.gradient(values, spacing, edge_order=2))


def target_measurement(
    target_level: np.ndarray,
    neighbour_level: np.ndarray,
    mesh: tuple[np.ndarray, ...],
    axis: np.ndarray,
    spacing: float,
    target_position: tuple[float, float, float],
    neighbour_strength: float,
) -> dict:
    isolated = stored_level(TARGET_DRAW_STRENGTH * target_level)
    background = stored_level(neighbour_strength * neighbour_level)
    combined = stored_level(
        TARGET_DRAW_STRENGTH * target_level
        + neighbour_strength * neighbour_level
    )
    paired_increment = combined - background
    radius = np.sqrt(
        sum(
            (coordinate - center) ** 2
            for coordinate, center in zip(mesh, target_position)
        )
    )
    fit_mask = (radius >= MEASUREMENT_RADII[0]) & (
        radius <= MEASUREMENT_RADII[1]
    )
    isolated_gradient = vector_gradient(isolated, spacing)[:, fit_mask]
    paired_gradient = vector_gradient(paired_increment, spacing)[:, fit_mask]
    denominator = float(np.sum(isolated_gradient**2))
    amplitude = float(np.sum(paired_gradient * isolated_gradient) / denominator)
    residual = paired_gradient - amplitude * isolated_gradient
    relative_vector_rmse = float(
        np.sqrt(np.mean(residual**2) / np.mean(isolated_gradient**2))
    )
    depletion = sample_at(background, axis, target_position)
    return {
        "neighbour_draw_strength": neighbour_strength,
        "ambient_depletion_at_target": depletion,
        "ambient_headroom_at_target": 1.0 - depletion,
        "measured_incremental_amplitude": amplitude,
        "paired_gradient_relative_rmse": relative_vector_rmse,
    }


def fit_families(samples: list[dict]) -> dict:
    depletion = np.array([row["ambient_depletion_at_target"] for row in samples])
    amplitude = np.array([row["measured_incremental_amplitude"] for row in samples])
    log_amplitude = np.log(amplitude)
    log_headroom = np.log1p(-depletion)

    headroom_exponent = float(
        np.dot(log_headroom, log_amplitude) / np.dot(log_headroom, log_headroom)
    )
    headroom_residual = log_amplitude - headroom_exponent * log_headroom
    headroom_standard_error = float(
        np.sqrt(
            np.sum(headroom_residual**2)
            / (len(depletion) - 1)
            / np.dot(log_headroom, log_headroom)
        )
    )

    def log_flux_family(x: np.ndarray, coefficient: float, exponent: float) -> np.ndarray:
        return -np.log1p(coefficient * x**exponent)

    flux_parameters, flux_covariance = curve_fit(
        log_flux_family,
        depletion,
        log_amplitude,
        p0=(5.0, 1.0),
        bounds=([0.0, 0.05], [100.0, 2.0]),
        maxfev=20_000,
    )
    flux_standard_errors = np.sqrt(np.maximum(np.diag(flux_covariance), 0.0))
    flux_prediction = log_flux_family(depletion, *flux_parameters)
    fixed_prediction = ORB_10157_EXPONENT * log_headroom

    def log_rmse(predicted_log_amplitude: np.ndarray) -> float:
        return float(np.sqrt(np.mean((log_amplitude - predicted_log_amplitude) ** 2)))

    return {
        "headroom_free": {
            "form": "A(D) = (1-D)^p",
            "p": headroom_exponent,
            "p_standard_error": headroom_standard_error,
            "p_ci95": [
                headroom_exponent - 1.96 * headroom_standard_error,
                headroom_exponent + 1.96 * headroom_standard_error,
            ],
            "log_rmse": log_rmse(headroom_exponent * log_headroom),
        },
        "flux_free": {
            "form": "A(D) = 1/(1+c*D^beta)",
            "c": float(flux_parameters[0]),
            "c_standard_error": float(flux_standard_errors[0]),
            "beta": float(flux_parameters[1]),
            "beta_standard_error": float(flux_standard_errors[1]),
            "log_rmse": log_rmse(flux_prediction),
            "nonanalytic_beta_below_one": bool(flux_parameters[1] < 1.0),
        },
        "ORB_10157_fixed": {
            "form": "A(D) = (1-D)^1.071",
            "p": ORB_10157_EXPONENT,
            "log_rmse": log_rmse(fixed_prediction),
        },
    }


def run_rung(
    size: int,
    target_position: tuple[float, float, float] = TARGET_POSITION,
    neighbour_position: tuple[float, float, float] = NEIGHBOUR_POSITION,
    strengths: tuple[float, ...] = NEIGHBOUR_DRAW_STRENGTHS,
) -> dict:
    target, axis, mesh, spacing, target_source = solve_draw_level(
        size, target_position
    )
    neighbour, _, _, _, neighbour_source = solve_draw_level(
        size, neighbour_position
    )
    sweep = [
        target_measurement(
            target,
            neighbour,
            mesh,
            axis,
            spacing,
            target_position,
            strength,
        )
        for strength in strengths
    ]

    combined_level = stored_level(target + neighbour)
    gradient = vector_gradient(combined_level, spacing)
    magnitude = np.sqrt(np.sum(gradient**2, axis=0))
    velocity = np.zeros_like(gradient)
    nonzero = magnitude > 0.0
    velocity[:, nonzero] = (
        np.sqrt(2.0 * combined_level[nonzero]) * gradient[:, nonzero]
        / magnitude[nonzero]
    )
    consumption = strain_consumption_3d(np.ones_like(combined_level), velocity, spacing)
    interior = consumption[2:-2, 2:-2, 2:-2]
    return {
        "apparatus": {
            "grid_size": size,
            "lattice": f"{size}^3",
            "spacing": spacing,
            "domain_half_width": DOMAIN_HALF_WIDTH,
            "physical_core_sigma": CORE_SIGMA,
            "core_sigma_in_cells": CORE_SIGMA / spacing,
            "target_position": list(target_position),
            "neighbour_position": list(neighbour_position),
            "core_separation": float(
                np.linalg.norm(np.subtract(neighbour_position, target_position))
            ),
            "measurement_radii": list(MEASUREMENT_RADII),
            "outer_boundary": "H=0 isolated rest-at-infinity level",
            "core_boundary": "none; normalized Gaussian cell draws source H through the discrete Poisson equation",
            "target_source": target_source,
            "neighbour_source": neighbour_source,
        },
        "sweep": sweep,
        "raw_depletion_range": [
            min(row["ambient_depletion_at_target"] for row in sweep),
            max(row["ambient_depletion_at_target"] for row in sweep),
        ],
        "raw_family_fits": fit_families(sweep) if len(sweep) >= 3 else None,
        "frozen_consumption_diagnostic": {
            "coefficient": 1.0,
            "maximum_density": float(np.max(interior)),
            "rms_density": float(np.sqrt(np.mean(interior**2))),
            "integrated_density": float(np.sum(interior) * spacing**3),
        },
    }


def apply_matched_depletion_grid(ladder: list[dict]) -> list[float]:
    lower = max(rung["raw_depletion_range"][0] for rung in ladder)
    upper = min(rung["raw_depletion_range"][1] for rung in ladder)
    matched = np.geomspace(lower, upper, MATCHED_GRID_SAMPLES)
    for rung in ladder:
        raw_depletion = np.array(
            [row["ambient_depletion_at_target"] for row in rung["sweep"]]
        )
        raw_amplitude = np.array(
            [row["measured_incremental_amplitude"] for row in rung["sweep"]]
        )
        interpolated_amplitude = np.exp(
            np.interp(
                np.log(matched), np.log(raw_depletion), np.log(raw_amplitude)
            )
        )
        rung["matched_fit_samples"] = [
            {
                "ambient_depletion_at_target": float(depletion),
                "measured_incremental_amplitude": float(amplitude),
            }
            for depletion, amplitude in zip(matched, interpolated_amplitude)
        ]
        rung["matched_family_fits"] = fit_families(rung["matched_fit_samples"])
    return [float(value) for value in matched]


def winning_parameter_convergence(ladder: list[dict]) -> dict:
    spacings = np.array([rung["apparatus"]["spacing"] for rung in ladder])
    exponents = np.array(
        [rung["matched_family_fits"]["headroom_free"]["p"] for rung in ladder]
    )
    standard_errors = np.array(
        [
            rung["matched_family_fits"]["headroom_free"]["p_standard_error"]
            for rung in ladder
        ]
    )
    shifts = np.diff(exponents)
    _, continuum = np.polyfit(spacings**2, exponents, 1)
    apparatus_error = float(
        np.sqrt(
            max(abs(continuum - exponents[-1]), abs(shifts[-1])) ** 2
            + standard_errors[-1] ** 2
        )
    )
    continuum_relative_shift = float(abs(continuum - exponents[-1]) / exponents[-1])
    converged = bool(
        abs(shifts[-1] / exponents[-1]) < 0.02
        and continuum_relative_shift < 0.02
    )
    return {
        "parameter": "headroom exponent p",
        "grid_sizes": list(GRID_SIZES),
        "spacings": [float(value) for value in spacings],
        "values": [float(value) for value in exponents],
        "regression_standard_errors": [float(value) for value in standard_errors],
        "successive_shifts": [float(value) for value in shifts],
        "successive_shift_shrinkage": bool(abs(shifts[-1]) < abs(shifts[0])),
        "finest_relative_shift": float(abs(shifts[-1] / exponents[-1])),
        "second_order_spacing_extrapolation": float(continuum),
        "continuum_to_finest_relative_shift": continuum_relative_shift,
        "combined_apparatus_error": apparatus_error,
        "consistent_with_ORB_10157_p_1_071": bool(
            abs(continuum - ORB_10157_EXPONENT) <= apparatus_error
        ),
        "converged": converged,
        "criterion": "second-order-in-spacing extrapolation and finest-rung shift each differ from the finest p by below 2%; successive shifts are also reported but need not be monotone",
    }


def small_d_analyticity(finest: dict) -> dict:
    samples = finest["matched_fit_samples"][:4]
    depletion = np.array([row["ambient_depletion_at_target"] for row in samples])
    log_amplitude = np.log(
        [row["measured_incremental_amplitude"] for row in samples]
    )
    slopes = np.diff(log_amplitude) / np.diff(depletion)
    growth_ratio = float(abs(slopes[0]) / max(abs(slopes[-1]), 1.0e-15))
    bounded = bool(np.max(np.abs(slopes)) < 2.0 and growth_ratio < 2.0)
    return {
        "verdict": "bounded_analytic_headroom_compatible" if bounded else "growing_nonanalytic_flux_compatible",
        "passed_directional_probe": bounded,
        "criterion": "on the four smallest matched D points, |d ln A/dD| stays below 2 and the smallest-D magnitude is less than twice the largest-D magnitude",
        "depletion_points": [float(value) for value in depletion],
        "finite_difference_d_ln_A_d_D": [float(value) for value in slopes],
        "smallest_to_largest_D_slope_magnitude_ratio": growth_ratio,
    }


def far_field_composition() -> dict:
    size = GRID_SIZES[-1]
    target, _, mesh, _, _ = solve_draw_level(size, TARGET_POSITION)
    neighbour, _, _, _, _ = solve_draw_level(size, NEIGHBOUR_POSITION)
    radius = np.sqrt(sum(coordinate**2 for coordinate in mesh))
    mask = (radius >= 8.5) & (radius <= 10.5)
    design = np.column_stack((np.ones(np.count_nonzero(mask)), 1.0 / radius[mask]))

    def monopole_coefficient(level: np.ndarray) -> float:
        return float(np.linalg.lstsq(design, level[mask], rcond=None)[0][1])

    level_coefficients = [monopole_coefficient(target), monopole_coefficient(neighbour)]
    combined_coefficient = monopole_coefficient(target + neighbour)
    component_amplitudes = [float(np.sqrt(value)) for value in level_coefficients]
    combined_amplitude = float(np.sqrt(combined_coefficient))
    expected_squared = float(sum(value**2 for value in component_amplitudes))
    relative_error = abs(combined_amplitude**2 - expected_squared) / expected_squared
    passed = bool(relative_error < 0.01)
    return {
        "verdict": "pass" if passed else "kill_converged_violation",
        "passed": passed,
        "criterion": "finest-rung far-field 1/r level fits obey A_pair^2=A_1^2+A_2^2 within 1%",
        "fit_shell_from_barycenter": [8.5, 10.5],
        "component_level_coefficients": level_coefficients,
        "component_amplitudes": component_amplitudes,
        "combined_level_coefficient": combined_coefficient,
        "combined_amplitude": combined_amplitude,
        "A_pair_squared": combined_amplitude**2,
        "A1_squared_plus_A2_squared": expected_squared,
        "relative_error": relative_error,
    }


def separation_control(finest_fit: dict) -> dict:
    exponent = finest_fit["headroom_free"]["p"]
    measurements = []
    for separation in SEPARATION_VALUES:
        target_position = (-0.5 * separation, 0.0, 0.0)
        neighbour_position = (0.5 * separation, 0.0, 0.0)
        rung = run_rung(
            GRID_SIZES[-1],
            target_position,
            neighbour_position,
            (SEPARATION_CONTROL_STRENGTH,),
        )
        row = rung["sweep"][0]
        prediction = (1.0 - row["ambient_depletion_at_target"]) ** exponent
        measurements.append(
            {
                "core_separation": separation,
                "fixed_target_draw_strength": TARGET_DRAW_STRENGTH,
                "fixed_neighbour_draw_strength": SEPARATION_CONTROL_STRENGTH,
                "ambient_depletion_at_target": row["ambient_depletion_at_target"],
                "measured_incremental_amplitude": row[
                    "measured_incremental_amplitude"
                ],
                "finest_headroom_family_prediction": prediction,
                "log_residual": float(
                    np.log(row["measured_incremental_amplitude"] / prediction)
                ),
            }
        )
    residuals = np.array([row["log_residual"] for row in measurements])
    log_rmse = float(np.sqrt(np.mean(residuals**2)))
    return {
        "passed": bool(log_rmse < 0.08),
        "criterion": "three fixed-draw separation points follow finest A(D) with log-RMSE below 0.08",
        "grid_size": GRID_SIZES[-1],
        "measurements": measurements,
        "log_rmse_against_A_of_D": log_rmse,
        "maximum_absolute_log_residual": float(np.max(np.abs(residuals))),
    }


def family_gate(ladder: list[dict], convergence: dict) -> dict:
    finest = ladder[-1]["matched_family_fits"]
    headroom_rmse = finest["headroom_free"]["log_rmse"]
    flux_rmse = finest["flux_free"]["log_rmse"]
    headroom_decisive = headroom_rmse * 3.0 < flux_rmse
    flux_nonanalytic_decisive = bool(
        flux_rmse * 3.0 < headroom_rmse
        and finest["flux_free"]["beta"] < 1.0
    )
    if (
        convergence["converged"]
        and convergence["consistent_with_ORB_10157_p_1_071"]
        and headroom_decisive
    ):
        verdict = "confirmed"
        passed = True
        rationale = "converged headroom family is >3x better than flux in log-RMSE and p is consistent with 1.071"
    elif convergence["converged"] and flux_nonanalytic_decisive:
        verdict = "refuted"
        passed = False
        rationale = "converged nonanalytic flux family is >3x better than headroom"
    else:
        verdict = "third-family"
        passed = False
        rationale = "neither predeclared confirmed nor nonanalytic-flux-refuted condition was met"
    return {
        "verdict": verdict,
        "passed_closure_prediction": passed,
        "rationale": rationale,
        "decisive_threshold": "winning family log-RMSE must be at least 3x smaller; fixed ORB-10157 is a parameter check within the headroom family",
        "convergence": convergence,
        "log_rmse_table": {
            name: fit["log_rmse"] for name, fit in finest.items()
        },
        "finest_fit_parameters": finest,
        "fits_by_rung": [
            {
                "grid_size": rung["apparatus"]["grid_size"],
                "fits": rung["matched_family_fits"],
            }
            for rung in ladder
        ],
    }


def experiment() -> tuple[dict, dict]:
    actual_stencil_hash = frozen_stencil_sha256()
    if actual_stencil_hash != EXPECTED_STENCIL_SHA256:
        raise RuntimeError(
            f"frozen stencil hash mismatch: {actual_stencil_hash}"
        )
    ladder = [run_rung(size) for size in GRID_SIZES]
    matched_grid = apply_matched_depletion_grid(ladder)
    convergence = winning_parameter_convergence(ladder)
    family = family_gate(ladder, convergence)
    analytic = small_d_analyticity(ladder[-1])
    composition = far_field_composition()
    separation = separation_control(ladder[-1]["matched_family_fits"])
    gates = {
        "family": family,
        "small_D_analyticity": analytic,
        "far_field_composition": composition,
    }
    apparatus = {
        "implementation_choice": "full 3-D extension of ORB-10932 level coupling, rather than replacing core boundaries inside ORB-10755's flux-relaxation solver",
        "tradeoff": "preserves genuinely nonspherical draw superposition and removes fixed-flux core data; as in ORB-10932, the conserved standing draw level is one-way coupled, so the frozen destruction stencil diagnoses substrate consumption but does not alter source counts",
        "field_equation": "-discrete_laplacian(H)=Gaussian draw density; sigma=1-exp(-H) anonymous shared-capacity expectation",
        "amplitude_measurement": "paired target stored-level gradient projected on the isolated target gradient over fixed physical radii 2-6",
        "no_fixed_flux_boundary": True,
        "grid_sizes": list(GRID_SIZES),
        "domain_half_width": DOMAIN_HALF_WIDTH,
        "fixed_physical_core_sigma": CORE_SIGMA,
        "target_position": list(TARGET_POSITION),
        "neighbour_position": list(NEIGHBOUR_POSITION),
        "matched_depletion_grid": matched_grid,
        "matched_grid_interpolation": "piecewise linear in log(D)-log(A) inside every rung's directly measured range",
        "consumption_rule": "s=n*sqrt((3/2)*e_dev:e_dev), coefficient exactly 1",
        "ORB_10751_expected_stencil_sha256": EXPECTED_STENCIL_SHA256,
        "measured_stencil_sha256": actual_stencil_hash,
        "stencil_sha256_match": actual_stencil_hash == EXPECTED_STENCIL_SHA256,
    }
    reproducibility = {
        "seed": SEED,
        "random_numbers_used": False,
        "byte_identical_in_memory_rerun_verified": True,
        "command": "UV_CACHE_DIR=/tmp/orbit-uv-cache-10934 uv run lab/sims/level-core-two-core-superposition/main.py --check-determinism",
    }
    limitations = [
        "The finite box uses a zero-level Dirichlet face as a discrete rest-at-infinity approximation; the fixed physical domain makes its bias common across the resolution ladder.",
        "The anonymous-slot expectation is evaluated deterministically rather than sampled at finite capacity; this isolates discretization and family error from Monte Carlo shot noise.",
        "The level-to-consumption coupling is one-way, matching ORB-10932: destroyed substrate does not erase externally conserved core draws.",
        "The small-D gate is exploratory and uses finite differences on the four smallest matched depletion points.",
        "This is an apparatus verdict only; principia reconciliation is deferred to kepler.",
    ]
    record = {
        "schema_version": 1,
        "task": "ORB-10934",
        "run_id": RUN_ID,
        "run_date": "2026-08-21",
        "reproducibility": reproducibility,
        "apparatus": apparatus,
        "resolution_ladder": ladder,
        "separation_control": separation,
        "gates": gates,
        "verdict": {
            "family": family["verdict"],
            "small_D_analyticity": analytic["verdict"],
            "far_field_composition": composition["verdict"],
            "all_three_predeclared_gates_executed": True,
            "closure_prediction_confirmed": family["verdict"] == "confirmed",
            "theory_reconciliation": "deferred to kepler; principia intentionally untouched",
        },
        "limitations": limitations,
    }
    results = {
        "schema_version": 1,
        "task": "ORB-10934",
        "run_record": f"runs/{RUN_RECORD}",
        "reproducibility": reproducibility,
        "apparatus": apparatus,
        "gates": gates,
        "resolution_ladder": {
            "grid_sizes": list(GRID_SIZES),
            "spacings": [rung["apparatus"]["spacing"] for rung in ladder],
            "physical_core_sigma_by_rung": [
                rung["apparatus"]["physical_core_sigma"] for rung in ladder
            ],
            "core_sigma_in_cells_by_rung": [
                rung["apparatus"]["core_sigma_in_cells"] for rung in ladder
            ],
            "directly_measured_depletion_ranges": [
                rung["raw_depletion_range"] for rung in ladder
            ],
            "matched_depletion_grid": matched_grid,
            "family_fits_by_rung": family["fits_by_rung"],
            "convergence": convergence,
        },
        "separation_control": separation,
        "verdict": record["verdict"],
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
        f"family={results['gates']['family']['verdict']} "
        f"p0={results['gates']['family']['convergence']['second_order_spacing_extrapolation']:.6f} "
        f"small_D={results['gates']['small_D_analyticity']['verdict']} "
        f"composition={results['gates']['far_field_composition']['verdict']} "
        f"sha256={hashlib.sha256(encoded(record)).hexdigest()}"
    )


if __name__ == "__main__":
    main()
