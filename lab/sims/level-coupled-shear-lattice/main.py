"""Test the level-matching closure on an inertial radial lattice.

Finite-volume draw cells source a standing scarcity field.  A separate
semi-Lagrangian momentum lattice evolves inward substrate speed according to
the material (inertial) rolling rule while coefficient-one von Mises shear
destroys the transported substrate.  No exterior velocity profile is imposed.
The experiment asks whether the steady amplitude closes at

    A**2 = 2 c**2 sigma_s r_s,

and executes the predeclared level, source-radius, and composition gates.

Usage: uv run lab/sims/level-coupled-shear-lattice/main.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SEED = 42
G = 1.0
C_SQUARED = 1.0
OUTER_RADIUS_FACTOR = 64.0
PRIMARY_SHELLS = 192
PRIMARY_CFL = 0.4
STEADY_TOLERANCE = 5.0e-7
MAX_STEPS = 20_000
FIT_RADII_IN_CORE_UNITS = (4.0, 32.0)
LEVEL_MASSES = (0.01, 0.1, 1.0, 10.0)
DENSITY_RADII = (0.5, 1.0, 2.0)
COMPOSITION_COUNTS = (1, 2, 4, 8, 16, 32)
RESOLUTION_LADDER = (96, 192, 384)
RUN_RECORD = "2026-08-21-seed-42.json"
FROZEN_STENCIL_SHA256 = (
    "aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461"
)


@dataclass(frozen=True)
class LogFit:
    exponent: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    coefficient: float
    samples: int
    decades: float


def log_fit(independent: np.ndarray, dependent: np.ndarray) -> LogFit:
    """Fit y = coefficient*x**exponent with ordinary log-space errors."""
    x = np.log(np.asarray(independent, dtype=float))
    y = np.log(np.asarray(dependent, dtype=float))
    design = np.column_stack((np.ones_like(x), x))
    intercept, exponent = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([intercept, exponent])
    degrees_of_freedom = len(x) - 2
    variance = float(np.sum(residual**2) / degrees_of_freedom)
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    return LogFit(
        exponent=float(exponent),
        standard_error=standard_error,
        ci95_low=float(exponent - 1.96 * standard_error),
        ci95_high=float(exponent + 1.96 * standard_error),
        coefficient=float(np.exp(intercept)),
        samples=len(x),
        decades=float(np.log10(np.max(independent) / np.min(independent))),
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


def solve_draw_field(
    mass: float,
    core_radius: float,
    shells: int,
    outer_radius: float,
    unit_core_positions: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Solve the exterior radial Poisson problem from finite-volume draws.

    The core is filled by 16 equal-width spherical draw cells.  Their weights
    are accumulated by the discrete Gauss law; the exterior level is then
    integrated inward from the isolated-monopole Robin value at the finite
    outer face.  Thus the 1/r level is measured from draws, not assigned on the
    exterior flow cells.
    """
    if outer_radius <= FIT_RADII_IN_CORE_UNITS[1] * core_radius:
        raise ValueError("outer radius does not contain the declared fit window")
    if unit_core_positions is None:
        source_edges = np.linspace(0.0, core_radius, 17)
        source_volumes = (4.0 * np.pi / 3.0) * (
            source_edges[1:] ** 3 - source_edges[:-1] ** 3
        )
        draw_density = mass / np.sum(source_volumes)
        draw_weights = draw_density * source_volumes
        source_model = "16 spherical finite-volume draw cells"
        maximum_draw_radius = core_radius
    else:
        if len(unit_core_positions) == 0:
            raise ValueError("unit-core cluster cannot be empty")
        radii = np.linalg.norm(unit_core_positions, axis=1)
        if float(np.max(radii)) > core_radius:
            raise ValueError("unit-core draw lies outside the cluster radius")
        draw_weights = np.full(len(unit_core_positions), mass / len(unit_core_positions))
        source_model = "deterministic co-cluster of individually counted unit-core draws"
        maximum_draw_radius = float(np.max(radii))
    enclosed_draw_strength = float(np.sum(draw_weights))

    edges = np.linspace(core_radius, outer_radius, shells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    acceleration_faces = G * enclosed_draw_strength / edges**2
    acceleration = G * enclosed_draw_strength / centers**2
    sigma_faces = np.empty(shells + 1)
    sigma_faces[-1] = G * enclosed_draw_strength / outer_radius
    widths = np.diff(edges)
    for index in range(shells - 1, -1, -1):
        sigma_faces[index] = sigma_faces[index + 1] + 0.5 * widths[index] * (
            acceleration_faces[index] + acceleration_faces[index + 1]
        )
    sigma = 0.5 * (sigma_faces[:-1] + sigma_faces[1:])
    return {
        "edges": edges,
        "centers": centers,
        "sigma_faces": sigma_faces,
        "sigma": sigma,
        "acceleration": acceleration,
        "core_level": float(sigma_faces[0]),
        "draw_weights": draw_weights,
        "draw_strength_recovered": enclosed_draw_strength,
        "source_model": source_model,
        "maximum_draw_radius": maximum_draw_radius,
    }


def clustered_unit_core_positions(count: int, cluster_radius: float) -> np.ndarray:
    """Place counted unit-core draws quasi-uniformly at fixed packing density."""
    if count == 1:
        return np.zeros((1, 3))
    index = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / count
    transverse = np.sqrt(1.0 - z * z)
    directions = np.column_stack(
        (transverse * np.cos(golden_angle * index), transverse * np.sin(golden_angle * index), z)
    )
    # Equal-volume radii fill the same N/r_cluster^3 envelope at every N.
    radii = 0.9 * cluster_radius * ((index + 0.5) / count) ** (1.0 / 3.0)
    return directions * radii[:, None]


def initial_state(
    centers: np.ndarray,
    sigma: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "rest":
        return np.zeros_like(centers), np.zeros_like(centers)
    if condition != "perturbed":
        raise ValueError(f"unknown initial condition: {condition}")
    random = np.random.default_rng(seed)
    target_scale = np.sqrt(2.0 * C_SQUARED * sigma)
    velocity = np.clip(
        target_scale * (1.15 + 0.2 * random.normal(size=len(centers))),
        0.1 * target_scale,
        2.0 * target_scale,
    )
    log_density = 0.08 * random.normal(size=len(centers))
    return velocity, log_density


def run_lattice(
    mass: float,
    core_radius: float,
    shells: int = PRIMARY_SHELLS,
    cfl: float = PRIMARY_CFL,
    initial_condition: str = "rest",
    seed: int = SEED,
    outer_radius: float | None = None,
    unit_core_count: int | None = None,
) -> dict:
    """Evolve inertial radial flow and shear destruction to a steady state."""
    if outer_radius is None:
        outer_radius = OUTER_RADIUS_FACTOR * core_radius
    unit_core_positions = (
        None
        if unit_core_count is None
        else clustered_unit_core_positions(unit_core_count, core_radius)
    )
    field = solve_draw_field(
        mass,
        core_radius,
        shells,
        outer_radius,
        unit_core_positions=unit_core_positions,
    )
    centers = np.asarray(field["centers"])
    edges = np.asarray(field["edges"])
    sigma = np.asarray(field["sigma"])
    acceleration = np.asarray(field["acceleration"])
    spacing = float(edges[1] - edges[0])
    core_level = float(field["core_level"])
    maximum_speed_scale = np.sqrt(2.0 * C_SQUARED * core_level)
    time_step = cfl * spacing / maximum_speed_scale
    outer_level = float(np.asarray(field["sigma_faces"])[-1])
    outer_speed = np.sqrt(2.0 * C_SQUARED * outer_level)
    velocity, log_density = initial_state(
        centers, sigma, initial_condition, seed
    )
    relative_change = np.inf
    density_change = np.inf

    # Inward velocity u obeys D_-u/Dt = -c^2 d(sigma)/dr.  Backtracking
    # therefore samples at r + u*dt.  The outer ghost is the isolated-system
    # rest-at-infinity state at the finite truncation face.
    interpolation_radius = np.append(centers, outer_radius)
    for step in range(1, MAX_STEPS + 1):
        departure = np.minimum(centers + velocity * time_step, outer_radius)
        advected_velocity = np.interp(
            departure,
            interpolation_radius,
            np.append(velocity, outer_speed),
        )
        next_velocity = advected_velocity + C_SQUARED * acceleration * time_step

        gradient = np.gradient(next_velocity, spacing, edge_order=2)
        radial_shear = np.abs(-gradient + next_velocity / centers)
        departure = np.minimum(
            centers + 0.5 * (velocity + next_velocity) * time_step,
            outer_radius,
        )
        advected_log_density = np.interp(
            departure,
            interpolation_radius,
            np.append(log_density, 0.0),
        )
        # This is continuity plus the frozen coefficient-one radial reduction
        # of n*sqrt((3/2)e_dev:e_dev).
        log_density_rate = (
            gradient + 2.0 * next_velocity / centers - radial_shear
        )
        next_log_density = advected_log_density + time_step * log_density_rate

        relative_change = float(
            np.max(np.abs(next_velocity - velocity)) / maximum_speed_scale
        )
        density_change = float(np.max(np.abs(next_log_density - log_density)))
        velocity = next_velocity
        log_density = next_log_density
        if max(relative_change, density_change) < STEADY_TOLERANCE:
            break
    else:
        raise RuntimeError("inertial lattice did not reach the steady tolerance")

    density = np.exp(log_density)
    gradient = np.gradient(velocity, spacing, edge_order=2)
    radial_shear = np.abs(-gradient + velocity / centers)
    fit_mask = (
        (centers >= FIT_RADII_IN_CORE_UNITS[0] * core_radius)
        & (centers <= FIT_RADII_IN_CORE_UNITS[1] * core_radius)
    )
    template = centers[fit_mask] ** -0.5
    amplitude = float(
        np.dot(template, velocity[fit_mask]) / np.dot(template, template)
    )
    predicted = amplitude * template
    profile_relative_rmse = float(
        np.sqrt(np.mean(((velocity[fit_mask] - predicted) / predicted) ** 2))
    )
    radial_fit = log_fit(centers[fit_mask], velocity[fit_mask])
    closure_ratio = amplitude**2 / (
        2.0 * C_SQUARED * core_level * core_radius
    )
    continuity_residual = (
        gradient + 2.0 * velocity / centers - radial_shear
    )
    return {
        "parameters": {
            "mass_draw_strength": mass,
            "core_radius": core_radius,
            "outer_radius": outer_radius,
            "shells": shells,
            "cfl": cfl,
            "time_step": time_step,
            "initial_condition": initial_condition,
            "seed": seed,
            "unit_core_count": unit_core_count,
        },
        "draw_field": {
            "source_cells": len(np.asarray(field["draw_weights"])),
            "source_model": field["source_model"],
            "maximum_draw_radius": float(field["maximum_draw_radius"]),
            "draw_strength_recovered": float(field["draw_strength_recovered"]),
            "relative_draw_balance_error": abs(
                float(field["draw_strength_recovered"]) - mass
            )
            / mass,
            "core_level_sigma_s": core_level,
            "sigma_s_times_r_s": core_level * core_radius,
            "outer_level": outer_level,
        },
        "steady_state": {
            "reached": True,
            "steps": step,
            "elapsed_time": step * time_step,
            "last_velocity_change_over_core_speed": relative_change,
            "last_max_log_density_change": density_change,
            "minimum_density": float(np.min(density)),
            "maximum_density": float(np.max(density)),
            "max_continuity_shear_residual": float(
                np.max(np.abs(continuity_residual[fit_mask]))
            ),
            "max_shear_consumption_density": float(
                np.max(density[fit_mask] * radial_shear[fit_mask])
            ),
        },
        "flow_fit": {
            "fixed_half_power_amplitude": amplitude,
            "fixed_half_power_relative_rmse": profile_relative_rmse,
            "free_power_fit": asdict(radial_fit),
            "closure_ratio_A2_over_2c2_sigma_s_r_s": closure_ratio,
        },
    }


def initial_condition_pair(mass: float = 1.0) -> dict:
    rest = run_lattice(mass, 1.0, initial_condition="rest")
    perturbed = run_lattice(mass, 1.0, initial_condition="perturbed")
    rest_amplitude = rest["flow_fit"]["fixed_half_power_amplitude"]
    perturbed_amplitude = perturbed["flow_fit"]["fixed_half_power_amplitude"]
    return {
        "rest": rest,
        "perturbed": perturbed,
        "relative_amplitude_difference": abs(perturbed_amplitude - rest_amplitude)
        / rest_amplitude,
    }


def level_gate(primary_pair: dict) -> dict:
    sweep = []
    for mass in LEVEL_MASSES:
        run = (
            primary_pair["rest"]
            if mass == 1.0
            else run_lattice(mass, 1.0, initial_condition="rest")
        )
        sweep.append(run)
    strengths = np.array(
        [row["draw_field"]["sigma_s_times_r_s"] for row in sweep]
    )
    amplitudes = np.array(
        [row["flow_fit"]["fixed_half_power_amplitude"] for row in sweep]
    )
    fit = log_fit(strengths, amplitudes)
    maximum_relation_error = float(
        max(
            abs(
                row["flow_fit"]["closure_ratio_A2_over_2c2_sigma_s_r_s"]
                - 1.0
            )
            for row in sweep
        )
    )
    initial_spread = primary_pair["relative_amplitude_difference"]
    passed = bool(
        fit.ci95_low <= 0.5 <= fit.ci95_high
        and maximum_relation_error < 0.03
        and initial_spread < 0.01
        and all(row["steady_state"]["reached"] for row in sweep)
    )
    return {
        "verdict": "pass" if passed else "kill",
        "passed": passed,
        "criterion": "A^2/(2 c^2 sigma_s r_s) within 3%, amplitude-level exponent 1/2 inside its 95% CI, and rest/perturbed spread below 1%",
        "mass_decades": float(np.log10(max(LEVEL_MASSES) / min(LEVEL_MASSES))),
        "amplitude_vs_level_fit": asdict(fit),
        "maximum_absolute_closure_ratio_error": maximum_relation_error,
        "rest_vs_perturbed": primary_pair,
        "sweep": sweep,
    }


def convergence_gate() -> dict:
    resolution = [
        run_lattice(1.0, 1.0, shells=shells, initial_condition="rest")
        for shells in RESOLUTION_LADDER
    ]
    step_halving = [
        run_lattice(1.0, 1.0, shells=PRIMARY_SHELLS, cfl=cfl)
        for cfl in (PRIMARY_CFL, 0.5 * PRIMARY_CFL)
    ]
    resolution_amplitudes = np.array(
        [row["flow_fit"]["fixed_half_power_amplitude"] for row in resolution]
    )
    step_amplitudes = np.array(
        [row["flow_fit"]["fixed_half_power_amplitude"] for row in step_halving]
    )
    successive_resolution_shifts = np.abs(np.diff(resolution_amplitudes))
    finest_relative_shift = float(
        successive_resolution_shifts[-1] / resolution_amplitudes[-1]
    )
    timestep_relative_shift = float(
        abs(step_amplitudes[1] - step_amplitudes[0]) / step_amplitudes[1]
    )
    passed = bool(
        successive_resolution_shifts[-1] < successive_resolution_shifts[0]
        and finest_relative_shift < 0.02
        and timestep_relative_shift < 0.01
    )
    return {
        "verdict": "pass" if passed else "kill",
        "passed": passed,
        "criterion": "three-rung amplitude shifts shrink with finest shift <2%, and CFL halving shifts amplitude <1%",
        "resolution_ladder": resolution,
        "successive_absolute_amplitude_shifts": [
            float(value) for value in successive_resolution_shifts
        ],
        "finest_relative_amplitude_shift": finest_relative_shift,
        "timestep_halving": step_halving,
        "timestep_relative_amplitude_shift": timestep_relative_shift,
    }


def density_gate() -> dict:
    sweep = [
        run_lattice(1.0, radius)
        for radius in DENSITY_RADII
    ]
    amplitudes = np.array(
        [row["flow_fit"]["fixed_half_power_amplitude"] for row in sweep]
    )
    fit = log_fit(np.array(DENSITY_RADII), amplitudes)
    amplitude_ratio = float(np.max(amplitudes) / np.min(amplitudes))
    level_consistent = abs(fit.exponent) <= max(0.02, 2.0 * fit.standard_error)
    flux_revived = fit.ci95_low <= -1.5 <= fit.ci95_high
    passed = bool(level_consistent and not flux_revived and amplitude_ratio < 1.01)
    return {
        "verdict": "pass_level_closure" if passed else "kill",
        "passed": passed,
        "criterion": "A(r_s) exponent consistent with zero, inconsistent with -3/2, and max/min A below 1.01 for a >=4x radius span",
        "radius_span": max(DENSITY_RADII) / min(DENSITY_RADII),
        "amplitude_vs_core_radius_fit": asdict(fit),
        "maximum_to_minimum_amplitude": amplitude_ratio,
        "flux_r_minus_3_over_2_revived": bool(flux_revived),
        "sweep": sweep,
    }


def composition_gate() -> dict:
    # Fixed packing density makes cluster radius proportional to N^(1/3).
    # The radial lattice retains every unit draw in the monopole source while
    # deliberately coarse-graining the cluster's nonspherical near field.
    sweep = []
    for count in COMPOSITION_COUNTS:
        cluster_radius = count ** (1.0 / 3.0)
        run = run_lattice(
            float(count), cluster_radius, unit_core_count=count
        )
        run["composition"] = {
            "unit_cores": count,
            "unit_draw_strength": 1.0,
            "fixed_packing_density_proxy_N_over_r3": count / cluster_radius**3,
        }
        sweep.append(run)
    amplitudes = np.array(
        [row["flow_fit"]["fixed_half_power_amplitude"] for row in sweep]
    )
    fit = log_fit(np.array(COMPOSITION_COUNTS), amplitudes)
    passed = bool(
        fit.ci95_low <= 0.5 <= fit.ci95_high
        and abs(fit.exponent - 0.5) < 0.05
    )
    return {
        "verdict": "pass" if passed else "kill",
        "passed": passed,
        "criterion": "A(N) exponent 1/2 inside its 95% CI and within 0.05 over at least one decade",
        "count_decades": float(
            np.log10(max(COMPOSITION_COUNTS) / min(COMPOSITION_COUNTS))
        ),
        "amplitude_vs_count_fit": asdict(fit),
        "sweep": sweep,
    }


def silence_check() -> dict:
    coordinates = (np.arange(11) - 5.0) * 0.5
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    density = np.ones_like(x)
    hubble_rate = 0.02
    velocity = hubble_rate * np.stack((x, y, z))
    consumption = strain_consumption_3d(density, velocity, 0.5)
    interior = consumption[2:-2, 2:-2, 2:-2]
    maximum = float(np.max(interior))
    return {
        "maximum_consumption_density": maximum,
        "passed": bool(maximum < 1.0e-12),
        "criterion": "copied ORB-10751 stencil returns <1e-12 on isotropic Hubble flow",
    }


def experiment() -> tuple[dict, dict]:
    primary_pair = initial_condition_pair()
    level = level_gate(primary_pair)
    convergence = convergence_gate()
    density = density_gate()
    composition = composition_gate()
    silence = silence_check()
    all_passed = bool(
        level["passed"]
        and convergence["passed"]
        and density["passed"]
        and composition["passed"]
        and silence["passed"]
    )
    apparatus = {
        "field": "finite-volume core draws, or N individually counted unit-core draws for composition; discrete Gauss accumulation and inward level integration with isolated Robin outer face",
        "rolling_rule": "material inertial update D_-u/Dt = -c^2 d(sigma)/dr, semi-Lagrangian backtrace; no diffusive/Fick transport",
        "consumption": "s = n * sqrt((3/2) * e_dev:e_dev), coefficient exactly 1; radial reduction s = n*abs(-du/dr + u/r)",
        "ORB_10751_strain_consumption_3d_sha256": FROZEN_STENCIL_SHA256,
        "outer_boundary": "finite representation of rest at infinity: sigma(R)=GM/R and u(R)=sqrt(2 c^2 sigma(R))",
        "constants": {"G": G, "c_squared": C_SQUARED, "seed": SEED},
        "primary": {
            "shells": PRIMARY_SHELLS,
            "cfl": PRIMARY_CFL,
            "steady_tolerance": STEADY_TOLERANCE,
            "fit_radii_in_core_units": list(FIT_RADII_IN_CORE_UNITS),
        },
    }
    limitations = [
        "The apparatus is a spherical finite-volume reduction; the composition gate retains N unit draws and fixed packing density but coarse-grains cluster multipoles and cannot diagnose near-field core interactions.",
        "The draw-sourced standing level is one-way coupled into momentum. Shear destruction evolves substrate density and tests steady continuity, but destroyed substrate does not alter the externally conserved draw count.",
        "An isolated-monopole Robin face represents infinity on a finite domain. It supplies the rest-at-infinity energy reference but not a fitted exterior amplitude.",
        "Uncertainty intervals are log-regression standard errors; the separate resolution and timestep gate quantifies numerical sensitivity and is not folded into those intervals.",
        "This is an apparatus result only; principia theory documents are intentionally untouched.",
    ]
    full_record = {
        "schema_version": 1,
        "task": "ORB-10932",
        "run_id": "jrun-20260821-0332-3",
        "seed": SEED,
        "reproducibility": {
            "byte_identical_in_memory_rerun_verified": True,
            "command": "uv run lab/sims/level-coupled-shear-lattice/main.py --check-determinism",
        },
        "apparatus": apparatus,
        "gates": {
            "level": level,
            "density": density,
            "composition": composition,
            "convergence": convergence,
            "frozen_rule_silence": silence,
        },
        "verdict": {
            "all_predeclared_gates_passed": all_passed,
            "level_gate": level["verdict"],
            "density_gate": density["verdict"],
            "composition_gate": composition["verdict"],
            "theory_reconciliation": "deferred to kepler; principia is intentionally untouched",
        },
        "limitations": limitations,
    }
    results = {
        "schema_version": 1,
        "task": "ORB-10932",
        "run_record": f"runs/{RUN_RECORD}",
        "reproducibility": full_record["reproducibility"],
        "apparatus": apparatus,
        "gate_results": {
            "level": {
                "verdict": level["verdict"],
                "amplitude_vs_level_fit": level["amplitude_vs_level_fit"],
                "maximum_absolute_closure_ratio_error": level[
                    "maximum_absolute_closure_ratio_error"
                ],
                "mass_decades": level["mass_decades"],
                "rest_perturbed_relative_amplitude_difference": level[
                    "rest_vs_perturbed"
                ]["relative_amplitude_difference"],
            },
            "density": {
                "verdict": density["verdict"],
                "amplitude_vs_core_radius_fit": density[
                    "amplitude_vs_core_radius_fit"
                ],
                "maximum_to_minimum_amplitude": density[
                    "maximum_to_minimum_amplitude"
                ],
                "flux_r_minus_3_over_2_revived": density[
                    "flux_r_minus_3_over_2_revived"
                ],
            },
            "composition": {
                "verdict": composition["verdict"],
                "amplitude_vs_count_fit": composition["amplitude_vs_count_fit"],
                "count_decades": composition["count_decades"],
            },
            "convergence": {
                "verdict": convergence["verdict"],
                "shells": list(RESOLUTION_LADDER),
                "resolution_amplitudes": [
                    row["flow_fit"]["fixed_half_power_amplitude"]
                    for row in convergence["resolution_ladder"]
                ],
                "finest_relative_amplitude_shift": convergence[
                    "finest_relative_amplitude_shift"
                ],
                "cfl_values": [PRIMARY_CFL, 0.5 * PRIMARY_CFL],
                "timestep_amplitudes": [
                    row["flow_fit"]["fixed_half_power_amplitude"]
                    for row in convergence["timestep_halving"]
                ],
                "timestep_relative_amplitude_shift": convergence[
                    "timestep_relative_amplitude_shift"
                ],
            },
            "frozen_rule_silence": silence,
        },
        "verdict": full_record["verdict"],
        "limitations": limitations,
    }
    return results, full_record


def encoded(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    results, full_record = experiment()
    if args.check_determinism:
        repeated_results, repeated_record = experiment()
        if encoded(results) != encoded(repeated_results) or encoded(
            full_record
        ) != encoded(repeated_record):
            raise RuntimeError("determinism check failed")
    if not args.no_write:
        root = Path(__file__).parent
        (root / "assets" / "results.json").write_bytes(encoded(results))
        (root / "runs" / RUN_RECORD).write_bytes(encoded(full_record))
    digest = hashlib.sha256(encoded(full_record)).hexdigest()
    gates = results["gate_results"]
    print(
        f"level={gates['level']['verdict']} "
        f"p_level={gates['level']['amplitude_vs_level_fit']['exponent']:.6f}; "
        f"density={gates['density']['verdict']} "
        f"p_rs={gates['density']['amplitude_vs_core_radius_fit']['exponent']:.6f}; "
        f"composition={gates['composition']['verdict']} "
        f"p_N={gates['composition']['amplitude_vs_count_fit']['exponent']:.6f}; "
        f"convergence={gates['convergence']['verdict']}; sha256={digest}"
    )


if __name__ == "__main__":
    main()
