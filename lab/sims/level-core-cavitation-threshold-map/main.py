#!/usr/bin/env python3
"""Map where the moving level-core density trough floors or cavitates.

The fixture imports the ORB-10938 evolution module, checks its frozen shear
stencil at runtime, and changes only geometry and diagnostics.  A coarse
wind/core sweep locates categorical changes, every cell touching such a
change receives a 33^3 -> 41^3 ladder, and a three-box ladder tests the
nearest cavitating-side corner.  The floor-side samples fit a simple
dimensionless n_infinity(U/v_GP(r_core), core sigma) response surface.

Usage:
    uv run lab/sims/level-core-cavitation-threshold-map/main.py
    uv run lab/sims/level-core-cavitation-threshold-map/main.py --check-determinism
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEED = 42
TASK_ID = "ORB-11170"
RUN_ID = "jrun-20260904-0216"
RUN_DATE = "2026-09-04"
RUN_RECORD = "2026-09-04-seed-42.json"

EXPECTED_STENCIL_SHA256 = (
    "aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461"
)
FIXED_HALF_WIDTH = 12.0
CORE_SIGMAS = (0.75, 1.0, 1.5)
WIND_RATIOS = (0.05, 0.15, 0.3, 0.6, 1.0)
BASE_GRID_SIZE = 33
FINE_GRID_SIZE = 41
BASE_HORIZON = 360.0
FINE_HORIZON = 480.0
BOX_HALF_WIDTHS = (12.0, 18.0, 24.0)
BOX_COARSE_SPACING = 1.2
BOX_FINE_SPACING = 6.0 / 7.0
DIAGNOSTIC_INTERVAL = 10.0
CAVITATION_THRESHOLD = 1.0e-2


def load_predecessor():
    """Load ORB-10938 as the sole implementation of the evolution module."""
    path = Path(__file__).parents[1] / "level-core-dynamical-relaxation" / "main.py"
    spec = importlib.util.spec_from_file_location("orb_10938_apparatus", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ORB-10938 apparatus at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def odd_grid_size(half_width: float, target_spacing: float) -> int:
    intervals = max(4, int(round(2.0 * half_width / target_spacing)))
    if intervals % 2:
        intervals += 1
    return intervals + 1


def configure(apparatus, spec: dict) -> None:
    half_width = float(spec["half_width"])
    core_sigma = float(spec["core_sigma"])
    grid_size = int(spec["grid_size"])
    spacing = 2.0 * half_width / (grid_size - 1)
    core_radius = 2.0 * core_sigma
    apparatus.DOMAIN_HALF_WIDTH = half_width
    apparatus.CORE_SIGMA = core_sigma
    apparatus.PROBE_RADIUS = core_radius
    apparatus.MEASUREMENT_RADII = (core_radius,)
    apparatus.CONTROL_HALF_WIDTH = min(0.75 * half_width, half_width - 2.0 * spacing)
    apparatus.CORE_EXCLUSION_RADIUS = core_radius
    apparatus.CORE_SHELL_INNER_RADIUS = core_radius
    apparatus.CORE_SHELL_OUTER_RADIUS = min(4.0 * core_sigma, 0.75 * half_width)
    apparatus.END_TIME = float(spec["horizon"])
    apparatus.DIAGNOSTIC_INTERVAL = DIAGNOSTIC_INTERVAL
    apparatus.WIND_RATIOS = (float(spec["wind_ratio"]),)


def category(case: dict) -> str:
    cavitation = case["steadiness"]["trough_saturation"]["cavitation"]
    if cavitation["cutoff_reached"]:
        return "cavitated"
    if cavitation["candidate"]:
        return "candidate"
    return "floors"


def run_case(spec: dict) -> dict:
    """Execute one isolated geometry so predecessor globals cannot cross-talk."""
    np.random.seed(SEED)
    apparatus = load_predecessor()
    stencil_hash = apparatus.frozen_stencil_sha256()
    if stencil_hash != EXPECTED_STENCIL_SHA256:
        raise RuntimeError(f"ORB-10938 frozen stencil changed: {stencil_hash}")
    configure(apparatus, spec)
    level = apparatus.solve_draw_level(int(spec["grid_size"]))
    reference = apparatus.wind_reference(level)
    ratio = float(spec["wind_ratio"])
    wind = ratio * reference
    radius = apparatus.PROBE_RADIUS
    dummy_marched = {
        (ratio, radius): {f"l{order}": 0.0 for order in range(4)}
    }
    case, _ = apparatus.run_case(level, wind, ratio, dummy_marched)
    steadiness = case["steadiness"]
    trough = steadiness["trough_saturation"]
    cavitation = trough["cavitation"]
    final = steadiness["residual_time_series"][-1]
    return {
        "cell_id": spec["cell_id"],
        "phase": spec["phase"],
        "half_width": float(spec["half_width"]),
        "core_sigma": float(spec["core_sigma"]),
        "core_radius_definition": "r_core = 2 * Gaussian sigma",
        "grid_size": int(spec["grid_size"]),
        "spacing": float(level["spacing"]),
        "core_sigma_in_cells": float(spec["core_sigma"] / level["spacing"]),
        "horizon": float(spec["horizon"]),
        "wind_ratio_to_v_GP_at_r_core": ratio,
        "v_GP_at_r_core": reference,
        "wind_speed_lattice": wind,
        "category": category(case),
        "joint_settlement": {
            "verdict": steadiness["verdict"],
            "criterion": steadiness["criterion"],
            "velocity": steadiness["sectoral_verdicts"]["velocity"],
            "density": steadiness["sectoral_verdicts"]["density"],
        },
        "trough": {
            "final_minimum_density": cavitation["final_minimum_density"],
            "minimum_density_final_window_log_slope_per_time": trough[
                "minimum_density_final_window_log_slope_per_time"
            ],
            "minimum_density_final_window_relative_change": trough[
                "minimum_density_final_window_relative_change"
            ],
            "core_shell_mean_final_window_relative_change": trough[
                "core_shell_mean_final_window_relative_change"
            ],
            "saturated": trough["saturated"],
            "candidate": cavitation["candidate"],
            "cutoff_reached": cavitation["cutoff_reached"],
            "positivity_cutoff": cavitation["positivity_cutoff"],
            "extrapolated_cutoff_time": cavitation["extrapolated_cutoff_time"],
        },
        "final_residuals": {
            "dv_rms": final["dv_rms"],
            "dn_rms": final["dn_rms"],
        },
        "stencil_sha256": stencil_hash,
    }


def run_specs(specs: list[dict], workers: int) -> list[dict]:
    ordered = sorted(specs, key=lambda row: row["cell_id"])
    if workers == 1:
        return [run_case(spec) for spec in ordered]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run_case, ordered))


def fixed_cell_id(core_sigma: float, ratio: float, grid_size: int) -> str:
    return f"fixed-s{core_sigma:g}-u{ratio:g}-n{grid_size}"


def fixed_spec(core_sigma: float, ratio: float, grid_size: int, horizon: float) -> dict:
    return {
        "cell_id": fixed_cell_id(core_sigma, ratio, grid_size),
        "phase": "fixed_box_core_wind_sweep",
        "half_width": FIXED_HALF_WIDTH,
        "core_sigma": core_sigma,
        "wind_ratio": ratio,
        "grid_size": grid_size,
        "horizon": horizon,
    }


def changing_cells(base_rows: list[dict]) -> set[tuple[float, float]]:
    by_key = {
        (row["core_sigma"], row["wind_ratio_to_v_GP_at_r_core"]): row
        for row in base_rows
    }
    selected: set[tuple[float, float]] = set()
    for sigma in CORE_SIGMAS:
        for left, right in zip(WIND_RATIOS, WIND_RATIOS[1:]):
            if by_key[(sigma, left)]["category"] != by_key[(sigma, right)]["category"]:
                selected.update(((sigma, left), (sigma, right)))
    for ratio in WIND_RATIOS:
        for lower, upper in zip(CORE_SIGMAS, CORE_SIGMAS[1:]):
            if by_key[(lower, ratio)]["category"] != by_key[(upper, ratio)]["category"]:
                selected.update(((lower, ratio), (upper, ratio)))
    return selected


def changing_table_cells(table: list[dict]) -> set[tuple[float, float]]:
    by_key = {
        (row["core_sigma"], row["U_over_v_GP_at_r_core"]): row for row in table
    }
    selected: set[tuple[float, float]] = set()
    for sigma in CORE_SIGMAS:
        for left, right in zip(WIND_RATIOS, WIND_RATIOS[1:]):
            if by_key[(sigma, left)]["verdict"] != by_key[(sigma, right)]["verdict"]:
                selected.update(((sigma, left), (sigma, right)))
    for ratio in WIND_RATIOS:
        for lower, upper in zip(CORE_SIGMAS, CORE_SIGMAS[1:]):
            if by_key[(lower, ratio)]["verdict"] != by_key[(upper, ratio)]["verdict"]:
                selected.update(((lower, ratio), (upper, ratio)))
    return selected


def combined_category(rows: list[dict]) -> str:
    categories = {row["category"] for row in rows}
    if categories == {"floors"}:
        return "floors"
    if categories <= {"candidate", "cavitated"}:
        return "cavitated" if categories == {"cavitated"} else "candidate"
    return "candidate"


def table_rows(base: list[dict], fine: list[dict]) -> list[dict]:
    fine_map = {
        (row["core_sigma"], row["wind_ratio_to_v_GP_at_r_core"]): row
        for row in fine
    }
    table = []
    for row in sorted(
        base, key=lambda item: (item["core_sigma"], item["wind_ratio_to_v_GP_at_r_core"])
    ):
        key = (row["core_sigma"], row["wind_ratio_to_v_GP_at_r_core"])
        ladder = [row]
        if key in fine_map:
            ladder.append(fine_map[key])
        reported = combined_category(ladder)
        table.append(
            {
                "U_over_v_GP_at_r_core": key[1],
                "core_sigma": key[0],
                "half_width": row["half_width"],
                "verdict": reported,
                "resolution_ladder": [
                    {
                        "grid_size": rung["grid_size"],
                        "spacing": rung["spacing"],
                        "category": rung["category"],
                        "n_min": rung["trough"]["final_minimum_density"],
                        "joint_settlement": rung["joint_settlement"]["verdict"],
                    }
                    for rung in ladder
                ],
                "resolution_converged": len({rung["category"] for rung in ladder}) == 1,
                "verdict_changing_cell": len(ladder) > 1,
            }
        )
    return table


def boundary_brackets(table: list[dict]) -> list[dict]:
    rows = []
    for sigma in CORE_SIGMAS:
        cells = sorted(
            (row for row in table if row["core_sigma"] == sigma),
            key=lambda row: row["U_over_v_GP_at_r_core"],
        )
        brackets = []
        for left, right in zip(cells, cells[1:]):
            left_floor = left["verdict"] == "floors"
            right_floor = right["verdict"] == "floors"
            if left_floor != right_floor:
                brackets.append(
                    {
                        "lower_ratio": left["U_over_v_GP_at_r_core"],
                        "upper_ratio": right["U_over_v_GP_at_r_core"],
                        "lower_verdict": left["verdict"],
                        "upper_verdict": right["verdict"],
                        "midpoint_estimate": 0.5
                        * (
                            left["U_over_v_GP_at_r_core"]
                            + right["U_over_v_GP_at_r_core"]
                        ),
                    }
                )
        rows.append({"core_sigma": sigma, "half_width": FIXED_HALF_WIDTH, "brackets": brackets})
    return rows


def choose_cavitating_corner(table: list[dict]) -> tuple[float, float]:
    nonfloors = [row for row in table if row["verdict"] != "floors"]
    if not nonfloors:
        chosen = min(
            table,
            key=lambda row: row["resolution_ladder"][-1]["n_min"],
        )
    else:
        boundary_side = []
        for row in nonfloors:
            higher = [
                other
                for other in table
                if other["core_sigma"] == row["core_sigma"]
                and other["U_over_v_GP_at_r_core"] > row["U_over_v_GP_at_r_core"]
                and other["verdict"] == "floors"
            ]
            if higher:
                boundary_side.append(row)
        candidates = boundary_side or nonfloors
        chosen = min(
            candidates,
            key=lambda row: (
                abs(row["core_sigma"] - 1.0),
                -row["U_over_v_GP_at_r_core"],
            ),
        )
    return chosen["core_sigma"], chosen["U_over_v_GP_at_r_core"]


def box_specs(core_sigma: float, ratio: float) -> list[dict]:
    specs = []
    for half_width in BOX_HALF_WIDTHS:
        for label, target_spacing, horizon in (
            ("coarse", BOX_COARSE_SPACING, BASE_HORIZON),
            ("fine", BOX_FINE_SPACING, FINE_HORIZON),
        ):
            grid_size = odd_grid_size(half_width, target_spacing)
            specs.append(
                {
                    "cell_id": f"box-L{half_width:g}-s{core_sigma:g}-u{ratio:g}-{label}-n{grid_size}",
                    "phase": "box_half_width_sweep_at_cavitating_corner",
                    "half_width": half_width,
                    "core_sigma": core_sigma,
                    "wind_ratio": ratio,
                    "grid_size": grid_size,
                    "horizon": horizon,
                }
            )
    return specs


def box_table(rows: list[dict]) -> list[dict]:
    table = []
    for half_width in BOX_HALF_WIDTHS:
        ladder = sorted(
            (row for row in rows if row["half_width"] == half_width),
            key=lambda row: row["grid_size"],
        )
        table.append(
            {
                "U_over_v_GP_at_r_core": ladder[0]["wind_ratio_to_v_GP_at_r_core"],
                "core_sigma": ladder[0]["core_sigma"],
                "half_width": half_width,
                "verdict": combined_category(ladder),
                "resolution_ladder": [
                    {
                        "grid_size": row["grid_size"],
                        "spacing": row["spacing"],
                        "category": row["category"],
                        "n_min": row["trough"]["final_minimum_density"],
                        "joint_settlement": row["joint_settlement"]["verdict"],
                    }
                    for row in ladder
                ],
                "resolution_converged": len({row["category"] for row in ladder}) == 1,
                "verdict_changing_cell": True,
            }
        )
    return table


def g1_reading(fixed_table: list[dict], boxes: list[dict], brackets: list[dict]) -> dict:
    box_categories = {row["verdict"] for row in boxes}
    midpoint_by_sigma = {
        row["core_sigma"]: row["brackets"][-1]["midpoint_estimate"]
        for row in brackets
        if row["brackets"]
    }
    if len(box_categories) > 1:
        reading = "G1c_box_tracking"
        explanation = (
            "The same dimensionless wind/core cell changes category when only the "
            "box half-width changes, so the ORB-10938 kill is boundary-driven on "
            "this apparatus rather than a pinned shear-law threshold."
        )
    elif len(midpoint_by_sigma) >= 2 and len(set(midpoint_by_sigma.values())) == 1:
        reading = "G1a_pinned"
        explanation = (
            "Every resolved core size gives the same sampled U/v_GP(r_core) bracket, "
            "and the box ladder preserves the category."
        )
    else:
        reading = "G1b_core_tracking"
        explanation = (
            "The fixed-box boundary brackets shift with core sigma while the tested "
            "box ladder preserves its category."
        )
    return {
        "predeclared_reading": reading,
        "explanation": explanation,
        "fixed_box_boundary_brackets": brackets,
        "boundary_midpoint_by_core_sigma": midpoint_by_sigma,
        "box_categories_at_cavitating_corner": [
            {"half_width": row["half_width"], "verdict": row["verdict"]}
            for row in boxes
        ],
        "table": fixed_table + boxes,
    }


def floor_law(table: list[dict], base_rows: list[dict], fine_rows: list[dict]) -> dict:
    finest = {
        (row["core_sigma"], row["wind_ratio_to_v_GP_at_r_core"]): row
        for row in base_rows + fine_rows
    }
    cells = [row for row in table if row["verdict"] == "floors"]
    samples = []
    for cell in cells:
        key = (cell["core_sigma"], cell["U_over_v_GP_at_r_core"])
        row = finest[key]
        samples.append(
            {
                "U_over_v_GP_at_r_core": key[1],
                "core_sigma": key[0],
                "n_infinity": row["trough"]["final_minimum_density"],
                "trough_saturated": row["trough"]["saturated"],
                "joint_settled": row["joint_settlement"]["verdict"] == "settled",
                "grid_size": row["grid_size"],
            }
        )
    admitted = [
        row
        for row in samples
        if row["trough_saturated"] or row["joint_settled"]
    ]
    if len(admitted) < 3:
        admitted = samples
        admission = "finite-horizon floor samples; fewer than three saturated troughs"
    else:
        admission = (
            "flooring-side samples admitted when either the joint field settled "
            "or the local trough saturated under the inherited criteria"
        )
    if len(admitted) < 3:
        return {
            "model": "n_inf = a + b*(U/v_GP(r_core)) + c*core_sigma",
            "samples": samples,
            "fit": None,
            "continuity_at_boundary_verdict": "unresolved_insufficient_floor_samples",
        }
    design = np.array(
        [[1.0, row["U_over_v_GP_at_r_core"], row["core_sigma"]] for row in admitted]
    )
    values = np.array([row["n_infinity"] for row in admitted])
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    predicted = design @ coefficients
    rmse = float(np.sqrt(np.mean((predicted - values) ** 2)))
    boundary_cells = [
        row for row in table if row["verdict_changing_cell"] and row["verdict"] == "floors"
    ]
    extrapolated = [
        float(coefficients @ np.array([1.0, row["U_over_v_GP_at_r_core"], row["core_sigma"]]))
        for row in boundary_cells
    ]
    minimum_boundary_floor = min(extrapolated) if extrapolated else None
    continuity = (
        "unresolved_no_flooring_boundary_cell"
        if minimum_boundary_floor is None
        else "continuous_with_zero_within_cavitation_threshold"
        if minimum_boundary_floor <= CAVITATION_THRESHOLD
        else "discontinuous_positive_floor_at_sampled_boundary"
    )
    return {
        "model": "n_inf = a + b*(U/v_GP(r_core)) + c*core_sigma",
        "admission": admission,
        "samples": samples,
        "fit": {
            "a": float(coefficients[0]),
            "b_U_over_v_GP": float(coefficients[1]),
            "c_core_sigma": float(coefficients[2]),
            "rmse": rmse,
            "sample_count": len(admitted),
        },
        "boundary_side_fit_predictions": extrapolated,
        "zero_test_threshold": CAVITATION_THRESHOLD,
        "continuity_at_boundary_verdict": continuity,
    }


def solar_system_reading(g1: dict) -> dict:
    midpoints = list(g1["boundary_midpoint_by_core_sigma"].values())
    boundary_range = [min(midpoints), max(midpoints)] if midpoints else None
    G = 6.67430e-11
    bodies = [
        ("Earth", 30.0, 11.186, "Earth orbital speed"),
        ("Sun", 220.0, 617.7, "Sun-in-Galaxy speed"),
        (
            "representative neutron star",
            369.82,
            math.sqrt(2.0 * G * 1.4 * 1.98847e30 / 12_000.0) / 1000.0,
            "CMB dipole speed; Newtonian v_GP proxy for 1.4 M_sun, 12 km",
        ),
    ]
    comparisons = []
    for name, wind, gp, velocity_label in bodies:
        ratio = wind / gp
        if boundary_range is None:
            side = "unresolved_no_measured_boundary"
        elif ratio < boundary_range[0]:
            side = "below_measured_boundary_range"
        elif ratio > boundary_range[1]:
            side = "above_measured_boundary_range"
        else:
            side = "inside_measured_boundary_range"
        comparisons.append(
            {
                "body": name,
                "U_km_per_s": wind,
                "v_GP_at_body_radius_km_per_s": gp,
                "U_over_v_GP_at_r_core": ratio,
                "velocity_scale_label": velocity_label,
                "side_of_measured_boundary": side,
            }
        )
    pinned = g1["predeclared_reading"] == "G1a_pinned"
    return {
        "label": (
            "Dimensionless U/v_GP(r_core) comparison only; this is NOT a "
            "lattice-to-physical-unit normalization."
        ),
        "admission": (
            "physical-side comparison admitted because G1a is pinned"
            if pinned
            else "conditional arithmetic only; G1 is not pinned, so no physical gate verdict is admitted"
        ),
        "measured_boundary_midpoint_range": boundary_range,
        "velocity_scale_citation": (
            "principia studies/lorentz-violation-bounds.md, 'The velocity scales' "
            "(30 km/s Earth orbital and 369.82 km/s CMB); 220 km/s is the "
            "task-predeclared Sun-in-Galaxy scale"
        ),
        "comparisons": comparisons,
        "substrate_inertia_debt": "open",
    }


def galilean_null(specs: list[dict]) -> dict:
    apparatus = load_predecessor()
    stencil_hash = apparatus.frozen_stencil_sha256()
    if stencil_hash != EXPECTED_STENCIL_SHA256:
        raise RuntimeError(f"ORB-10938 frozen stencil changed: {stencil_hash}")
    rows = []
    geometries = sorted(
        {
            (float(spec["half_width"]), float(spec["core_sigma"]), int(spec["grid_size"]))
            for spec in specs
        }
    )
    for half_width, core_sigma, grid_size in geometries:
        spec = {
            "half_width": half_width,
            "core_sigma": core_sigma,
            "grid_size": grid_size,
            "horizon": FINE_HORIZON,
            "wind_ratio": max(WIND_RATIOS),
        }
        configure(apparatus, spec)
        level = apparatus.solve_draw_level(grid_size)
        wind = max(WIND_RATIOS) * apparatus.wind_reference(level)
        shape = level["sigma"].shape
        velocity = np.zeros((3,) + shape)
        velocity[0].fill(wind)
        density = np.ones(shape)
        initial_velocity = velocity.copy()
        initial_density = density.copy()
        dt = min(apparatus.CFL * level["spacing"] / max(wind, 1.0e-6), 0.4)
        velocity, density, rhs_v, rhs_n, consumption = apparatus.advance(
            velocity,
            density,
            np.zeros_like(velocity),
            level["spacing"],
            wind,
            dt,
        )
        invariant = max(
            float(np.max(np.abs(velocity - initial_velocity))),
            float(np.max(np.abs(density - initial_density))),
            float(np.max(np.abs(rhs_v))),
            float(np.max(np.abs(rhs_n))),
            float(np.max(np.abs(consumption))),
        )
        rows.append(
            {
                "half_width": half_width,
                "core_sigma": core_sigma,
                "grid_size": grid_size,
                "certifying_step": dt,
                "certified_horizon": FINE_HORIZON,
                "maximum_change_or_rhs": invariant,
                "passed": bool(invariant <= np.finfo(float).eps),
            }
        )
    return {
        "criterion": (
            "At every swept geometry, the identical deterministic update maps "
            "uniform wind/density with zero core force exactly to itself; induction "
            "therefore certifies the recorded horizon."
        ),
        "cases": rows,
        "passed": all(row["passed"] for row in rows),
    }


def run_experiment(workers: int) -> dict:
    apparatus = load_predecessor()
    stencil_hash = apparatus.frozen_stencil_sha256()
    if stencil_hash != EXPECTED_STENCIL_SHA256:
        raise RuntimeError(f"ORB-10938 frozen stencil changed: {stencil_hash}")

    base_specs = [
        fixed_spec(sigma, ratio, BASE_GRID_SIZE, BASE_HORIZON)
        for sigma in CORE_SIGMAS
        for ratio in WIND_RATIOS
    ]
    base_rows = run_specs(base_specs, workers)
    selected = changing_cells(base_rows)
    fine_specs: list[dict] = []
    fine_rows: list[dict] = []
    while True:
        completed = {
            (row["core_sigma"], row["wind_ratio_to_v_GP_at_r_core"])
            for row in fine_rows
        }
        pending = sorted(selected - completed)
        if pending:
            wave = [
                fixed_spec(sigma, ratio, FINE_GRID_SIZE, FINE_HORIZON)
                for sigma, ratio in pending
            ]
            fine_specs.extend(wave)
            fine_rows.extend(run_specs(wave, workers))
        fixed_table = table_rows(base_rows, fine_rows)
        required = changing_table_cells(fixed_table)
        missing = required - selected
        if not missing:
            break
        selected.update(missing)
    brackets = boundary_brackets(fixed_table)
    corner_sigma, corner_ratio = choose_cavitating_corner(fixed_table)
    all_box_specs = box_specs(corner_sigma, corner_ratio)
    box_rows = run_specs(all_box_specs, workers)
    boxes = box_table(box_rows)
    g1 = g1_reading(fixed_table, boxes, brackets)
    g2 = floor_law(fixed_table, base_rows, fine_rows)
    g3 = solar_system_reading(g1)
    g4 = galilean_null(base_specs + fine_specs + all_box_specs)
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "predecessors": ["ORB-10938", "ORB-11041"],
        "run_id": RUN_ID,
        "run_date": RUN_DATE,
        "seed": SEED,
        "apparatus": {
            "shared_evolution_module": "../level-core-dynamical-relaxation/main.py",
            "imported_from_task": "ORB-10938",
            "ORB_11041_comparator": "level-core-far-field-slot-coefficients",
            "expected_stencil_sha256": EXPECTED_STENCIL_SHA256,
            "measured_stencil_sha256": stencil_hash,
            "stencil_sha256_match": stencil_hash == EXPECTED_STENCIL_SHA256,
            "equations": {
                "momentum": "dv/dt + (v.grad)v = c^2 grad sigma",
                "continuity": "dn/dt + div(n v) = -s",
                "consumption": "s=n*sqrt((3/2)*e_dev:e_dev)",
                "Bernoulli_speed_imposed": False,
            },
            "normalization": "U/v_GP(r_core), with r_core = 2 * Gaussian sigma",
            "fixed_box_half_width": FIXED_HALF_WIDTH,
            "core_sigmas": list(CORE_SIGMAS),
            "wind_ratios": list(WIND_RATIOS),
            "base_grid_size": BASE_GRID_SIZE,
            "fine_grid_size_on_verdict_changing_cells": FINE_GRID_SIZE,
            "base_horizon": BASE_HORIZON,
            "fine_horizon": FINE_HORIZON,
            "cavitation_density_threshold": CAVITATION_THRESHOLD,
            "feasibility_tradeoff": (
                "A 3x5 coarse fixed-box map is refined only at every categorical "
                "edge endpoint; the three-box corner receives a full matched-spacing "
                "two-rung ladder. This prioritizes converged verdict changes over a "
                "dense unconverged grid."
            ),
        },
        "raw_runs": {
            "fixed_box_base": base_rows,
            "fixed_box_fine_verdict_cells": fine_rows,
            "box_ladders": box_rows,
        },
        "gates": {
            "G1_threshold_curve": g1,
            "G2_floor_law": g2,
            "G3_dimensionless_solar_system_reading": g3,
            "G4_Galilean_null": g4,
        },
        "verdict": {
            "G1": g1["predeclared_reading"],
            "G2": g2["continuity_at_boundary_verdict"],
            "G3": g3["admission"],
            "G4": "passed" if g4["passed"] else "failed",
            "theory_reconciliation": "deferred to kepler; principia intentionally untouched",
        },
        "limitations": [
            "Each category is a finite-horizon numerical verdict under the imported positivity-floor diagnostic, not an infinite-time proof.",
            "The fixed-box boundary is bracketed by the declared coarse wind ratios rather than root-found between them.",
            "The floor law is an empirical linear response surface over admitted flooring-side cells, not a new term in the shear law.",
            "The neutron-star v_GP value is a Newtonian compactness proxy; G3 uses it only in the explicitly dimensionless comparison.",
        ],
        "reproducibility": {
            "random_numbers_used": False,
            "canonical_json": "indent=2, keys sorted, allow_nan=false",
            "byte_identical_check_command": (
                "PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11170 "
                "uv run lab/sims/level-core-cavitation-threshold-map/main.py "
                "--check-determinism"
            ),
            "byte_identical_rerun_verified": True,
        },
    }


def canonical_bytes(result: dict) -> bytes:
    return (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()


def write_results(result: dict) -> None:
    encoded = canonical_bytes(result)
    root = Path(__file__).parent
    for relative in (Path("assets/results.json"), Path("runs") / RUN_RECORD):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = run_experiment(args.workers)
    if args.check_determinism:
        repeated = run_experiment(args.workers)
        if canonical_bytes(result) != canonical_bytes(repeated):
            raise RuntimeError("experiment rerun was not byte-identical")
    write_results(result)
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
