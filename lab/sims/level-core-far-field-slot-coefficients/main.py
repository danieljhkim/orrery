"""Measure far-field PPN-slot proxies of the settled moving level-core wake.

This fixture reuses the ORB-10938 rolling-rule momentum, consumed-density
continuity, boundary treatment, and frozen ORB-10751 shear stencil directly.
It changes only the geometry and diagnostics: a small core is placed in a
large box, three winds cross their matching radii inside the shell ladder, and
the settled disturbance field is reduced into radial tail verdicts and
dimensionless lattice-unit slot proxies.  No Bernoulli speed is imposed.

Usage:
    uv run lab/sims/level-core-far-field-slot-coefficients/main.py
    uv run lab/sims/level-core-far-field-slot-coefficients/main.py --check-determinism
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SEED = 42
TASK_ID = "ORB-11041"
RUN_ID = "jrun-20260828-0507-3"
RUN_DATE = "2026-08-28"
RUN_RECORD = "2026-08-28-seed-42.json"

GRID_SIZES = (41, 57)
DOMAIN_HALF_WIDTH = 24.0
CORE_SIGMA = 1.0
CORE_REGION_RADIUS = 2.0
PROBE_RADIUS = 5.0
SHELL_RADII = (3.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0)
WIND_RATIOS = (0.65, 0.8, 1.0)
END_TIME = 480.0
DIAGNOSTIC_INTERVAL = 10.0
C_SQUARED = 1.0
EXPECTED_STENCIL_SHA256 = (
    "aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461"
)
TAIL_SLOPE_TOLERANCE = 0.35
TAIL_R2_MINIMUM = 0.90
TAIL_LOCAL_SLOPE_SPREAD_MAXIMUM = 1.0
CONVERGENCE_RELATIVE_TOLERANCE = 0.30


def load_predecessor():
    """Load ORB-10938 as the single implementation of the shared apparatus."""
    path = Path(__file__).parents[1] / "level-core-dynamical-relaxation" / "main.py"
    spec = importlib.util.spec_from_file_location("orb_10938_apparatus", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load predecessor apparatus at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_apparatus(apparatus) -> None:
    """Reconfigure geometry/horizon without changing the evolution operators."""
    apparatus.DOMAIN_HALF_WIDTH = DOMAIN_HALF_WIDTH
    apparatus.CORE_SIGMA = CORE_SIGMA
    apparatus.PROBE_RADIUS = PROBE_RADIUS
    apparatus.MEASUREMENT_RADII = SHELL_RADII
    apparatus.WIND_RATIOS = WIND_RATIOS
    apparatus.CONTROL_HALF_WIDTH = 20.0
    apparatus.CORE_EXCLUSION_RADIUS = CORE_REGION_RADIUS
    apparatus.CORE_SHELL_INNER_RADIUS = CORE_REGION_RADIUS
    apparatus.CORE_SHELL_OUTER_RADIUS = 2.0 * CORE_REGION_RADIUS
    apparatus.SHELL_MU_NODES = 10
    apparatus.SHELL_PHI_NODES = 20
    apparatus.END_TIME = END_TIME
    apparatus.DIAGNOSTIC_INTERVAL = DIAGNOSTIC_INTERVAL
    apparatus.ATTRACTOR_GRID_SIZE = GRID_SIZES[0]
    apparatus.ATTRACTOR_WIND_RATIO = WIND_RATIOS[1]


def vector_calculus(field: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    gradients = [np.gradient(field[index], spacing, edge_order=2) for index in range(3)]
    curl = np.stack(
        (
            gradients[2][1] - gradients[1][2],
            gradients[0][2] - gradients[2][0],
            gradients[1][0] - gradients[0][1],
        )
    )
    squared = np.sum(field**2, axis=0)
    grad_squared = np.stack(np.gradient(squared, spacing, edge_order=2))
    return curl, grad_squared


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * values**2)))


def shell_reduction(apparatus, level: dict, velocity: np.ndarray, wind: float) -> list[dict]:
    # ORB-10938 uses +x inflow.  The reservoir-frame disturbance is therefore
    # u = v - U e_x, the sign-convention equivalent of principia's v + U.
    disturbance = velocity.copy()
    disturbance[0] -= wind
    curl, grad_squared = vector_calculus(disturbance, level["spacing"])
    disturbance_squared = np.sum(disturbance**2, axis=0)
    denominator = C_SQUARED - disturbance_squared
    safe_denominator = np.maximum(denominator, 1.0e-12)
    vorticity_term = curl / safe_denominator
    bernoulli_term = np.moveaxis(
        np.cross(np.moveaxis(grad_squared, 0, -1), np.moveaxis(disturbance, 0, -1)),
        -1,
        0,
    ) / safe_denominator**2
    total_obstruction = vorticity_term + bernoulli_term
    rows = []
    for radius in SHELL_RADII:
        points, directions, weights = apparatus.shell_quadrature(radius)
        sampled_u = apparatus.sample_vector(disturbance, level["axis"], points)
        sampled_u2 = np.sum(sampled_u**2, axis=1)
        sampled_speed = np.sqrt(sampled_u2)
        sampled_phi = apparatus.sample_scalar(level["sigma"], level["axis"], points)
        sampled_curl = apparatus.sample_vector(curl, level["axis"], points)
        sampled_vorticity = apparatus.sample_vector(
            vorticity_term, level["axis"], points
        )
        sampled_bernoulli = apparatus.sample_vector(
            bernoulli_term, level["axis"], points
        )
        sampled_total = apparatus.sample_vector(
            total_obstruction, level["axis"], points
        )
        phi_mean = float(np.sum(weights * sampled_phi))
        local_gp_speed = float(np.sqrt(2.0 * C_SQUARED * phi_mean))

        def vector_summary(values: np.ndarray) -> dict:
            magnitude = np.linalg.norm(values, axis=1)
            axis_coefficients = apparatus.legendre_coefficients(
                values[:, 0], directions, weights
            )
            return {
                "rms_magnitude": weighted_rms(magnitude, weights),
                "wind_axis_legendre": axis_coefficients,
                "wind_axis_l1_absolute": abs(axis_coefficients["l1"]),
            }

        direction = sampled_u / np.maximum(sampled_speed[:, None], 1.0e-30)
        rows.append(
            {
                "radius": radius,
                "mean_phi": phi_mean,
                "local_GP_speed": local_gp_speed,
                "local_wind_to_GP_ratio": wind / local_gp_speed,
                "wind_dominated": bool(wind >= local_gp_speed),
                "minimum_c_squared_minus_u_squared_on_grid": float(
                    np.min(denominator)
                ),
                "disturbance_speed": {
                    "mean": float(np.sum(weights * sampled_speed)),
                    "legendre": apparatus.legendre_coefficients(
                        sampled_speed, directions, weights
                    ),
                    "squared_legendre": apparatus.legendre_coefficients(
                        sampled_u2, directions, weights
                    ),
                },
                "direction_field": {
                    f"u_{component}_legendre": apparatus.legendre_coefficients(
                        direction[:, index], directions, weights
                    )
                    for index, component in enumerate(("x", "y", "z"))
                },
                "raw_vorticity": vector_summary(sampled_curl),
                "g0i_vorticity_piece": vector_summary(sampled_vorticity),
                "g0i_Bernoulli_anisotropy_piece": vector_summary(sampled_bernoulli),
                "g0i_total_obstruction": vector_summary(sampled_total),
            }
        )
    return rows


def power_law_fit(rows: list[dict], value) -> dict:
    selected = [row for row in rows if row["wind_dominated"] and value(row) > 0.0]
    if len(selected) < 3:
        return {
            "verdict": "no_scaling_regime",
            "reason": "fewer than three positive wind-dominated shells",
            "shell_count": len(selected),
        }
    radii = np.array([row["radius"] for row in selected])
    amplitudes = np.array([value(row) for row in selected])
    log_r = np.log(radii)
    log_a = np.log(amplitudes)
    slope, intercept = np.polyfit(log_r, log_a, 1)
    predicted = intercept + slope * log_r
    residual = log_a - predicted
    total = log_a - np.mean(log_a)
    r_squared = float(1.0 - np.sum(residual**2) / max(np.sum(total**2), 1.0e-30))
    local_slopes = np.diff(log_a) / np.diff(log_r)
    local_spread = float(np.max(local_slopes) - np.min(local_slopes))
    stable = r_squared >= TAIL_R2_MINIMUM and local_spread <= TAIL_LOCAL_SLOPE_SPREAD_MAXIMUM
    if not stable:
        verdict = "no_scaling_regime"
    elif abs(float(slope) + 1.0) <= TAIL_SLOPE_TOLERANCE:
        verdict = "slot_matching_tail"
    elif slope < -(1.0 + TAIL_SLOPE_TOLERANCE):
        verdict = "faster_decay_near_zone_confined"
    else:
        verdict = "no_scaling_regime"
    return {
        "verdict": verdict,
        "shell_count": len(selected),
        "radii": [float(value_) for value_ in radii],
        "amplitudes": [float(value_) for value_ in amplitudes],
        "fit_amplitude_at_r_equals_1": float(np.exp(intercept)),
        "power_law_exponent": float(slope),
        "r_squared": r_squared,
        "local_power_law_exponents": [float(value_) for value_ in local_slopes],
        "local_exponent_spread": local_spread,
        "slot_matching_criterion": (
            f"potential-equivalent exponent within {TAIL_SLOPE_TOLERANCE:g} of -1, "
            f"R^2 >= {TAIL_R2_MINIMUM:g}, and local-slope spread <= "
            f"{TAIL_LOCAL_SLOPE_SPREAD_MAXIMUM:g}"
        ),
    }


def radial_fits(rows: list[dict]) -> dict:
    return {
        "speed_w_hat_l1": power_law_fit(
            rows, lambda row: abs(row["disturbance_speed"]["squared_legendre"]["l1"])
        ),
        "speed_w_hat_l2": power_law_fit(
            rows, lambda row: abs(row["disturbance_speed"]["squared_legendre"]["l2"])
        ),
        "speed_w_hat_l3": power_law_fit(
            rows, lambda row: abs(row["disturbance_speed"]["squared_legendre"]["l3"])
        ),
        # Obstruction is one derivative above g0i.  Multiplication by r maps
        # its r^-2 PPN derivative tail to the requested Phi-like r^-1 ladder.
        "g0i_vorticity_piece": power_law_fit(
            rows,
            lambda row: row["radius"]
            * row["g0i_vorticity_piece"]["rms_magnitude"],
        ),
        "g0i_Bernoulli_anisotropy_piece": power_law_fit(
            rows,
            lambda row: row["radius"]
            * row["g0i_Bernoulli_anisotropy_piece"]["rms_magnitude"],
        ),
        "g0i_total_dipole": power_law_fit(
            rows,
            lambda row: row["radius"]
            * row["g0i_total_obstruction"]["wind_axis_l1_absolute"],
        ),
    }


def relative_difference(first: float, second: float) -> float:
    return float(abs(second - first) / max(abs(first), abs(second), 1.0e-30))


def converge_fits(coarse: dict, fine: dict) -> dict:
    answer = {}
    for field in fine:
        coarse_fit = coarse[field]
        fine_fit = fine[field]
        if "power_law_exponent" not in coarse_fit or "power_law_exponent" not in fine_fit:
            converged = False
            exponent_shift = None
            amplitude_shift = None
        else:
            exponent_shift = abs(
                fine_fit["power_law_exponent"] - coarse_fit["power_law_exponent"]
            )
            amplitude_shift = relative_difference(
                coarse_fit["amplitudes"][-1], fine_fit["amplitudes"][-1]
            )
            converged = bool(
                exponent_shift <= TAIL_SLOPE_TOLERANCE
                and amplitude_shift <= CONVERGENCE_RELATIVE_TOLERANCE
                and coarse_fit["verdict"] == fine_fit["verdict"]
            )
        answer[field] = {
            "coarse_verdict": coarse_fit["verdict"],
            "fine_verdict": fine_fit["verdict"],
            "exponent_absolute_shift": exponent_shift,
            "outer_amplitude_relative_shift": amplitude_shift,
            "converged": converged,
            "reported_verdict": (
                fine_fit["verdict"] if converged else "no_scaling_regime"
            ),
        }
    return answer


def shell_slot_proxies(row: dict, wind: float) -> dict:
    phi = max(row["mean_phi"], 1.0e-30)
    return {
        "alpha1_non_gauge_g0i_lattice": (
            row["radius"]
            * row["g0i_total_obstruction"]["wind_axis_legendre"]["l1"]
            / (phi * wind)
        ),
        "alpha2_speed_quadrupole_lattice": (
            row["disturbance_speed"]["squared_legendre"]["l2"]
            / (phi * wind**2)
        ),
        "two_alpha3_minus_alpha1_dipole_lattice": (
            row["disturbance_speed"]["squared_legendre"]["l1"] / (phi * wind)
        ),
    }


def slot_extractions(
    coarse_rows: list[dict], fine_rows: list[dict], convergence: dict, wind: float
) -> dict:
    mapping = {
        "alpha1_non_gauge_g0i_lattice": "g0i_total_dipole",
        "alpha2_speed_quadrupole_lattice": "speed_w_hat_l2",
        "two_alpha3_minus_alpha1_dipole_lattice": "speed_w_hat_l1",
    }
    coarse_by_radius = {row["radius"]: row for row in coarse_rows}
    answer = {}
    for slot, tail_field in mapping.items():
        tail = convergence[tail_field]
        if tail["reported_verdict"] == "faster_decay_near_zone_confined":
            answer[slot] = {
                "verdict": "zero_asymptotic_slot_from_faster_decay",
                "coefficient": 0.0,
                "apparatus_error": None,
            }
            continue
        if tail["reported_verdict"] != "slot_matching_tail":
            answer[slot] = {
                "verdict": "not_extractable_without_a_converged_scaling_regime",
                "coefficient": None,
                "apparatus_error": None,
            }
            continue
        selected = [row for row in fine_rows if row["wind_dominated"]][-3:]
        fine_values = np.array([shell_slot_proxies(row, wind)[slot] for row in selected])
        coarse_values = np.array(
            [shell_slot_proxies(coarse_by_radius[row["radius"]], wind)[slot] for row in selected]
        )
        coefficient = float(np.mean(fine_values))
        radial_scatter = float(np.std(fine_values))
        resolution_shift = float(np.max(np.abs(fine_values - coarse_values)))
        error = max(radial_scatter, resolution_shift)
        answer[slot] = {
            "verdict": (
                "zero_consistent_within_apparatus_error"
                if abs(coefficient) <= error
                else "resolved_nonzero_lattice_coefficient"
            ),
            "coefficient": coefficient,
            "apparatus_error": error,
            "error_components": {
                "far_shell_scatter": radial_scatter,
                "maximum_adjacent_rung_shift": resolution_shift,
            },
            "shell_radii": [row["radius"] for row in selected],
            "shell_values": [float(value_) for value_ in fine_values],
        }
    return answer


def wind_scaling(rows_by_ratio: dict, coarse_by_ratio: dict, field_name: str) -> dict:
    def amplitude(rows: list[dict]) -> float:
        row = rows[-1]
        if field_name == "vorticity":
            return row["radius"] * row["g0i_vorticity_piece"]["rms_magnitude"]
        if field_name == "Bernoulli_anisotropy":
            return row["radius"] * row["g0i_Bernoulli_anisotropy_piece"]["rms_magnitude"]
        return row["radius"] * row["g0i_total_obstruction"]["rms_magnitude"]

    ratios = sorted(rows_by_ratio)
    winds = np.array([rows_by_ratio[ratio]["wind"] for ratio in ratios])
    values = np.array([amplitude(rows_by_ratio[ratio]["rows"]) for ratio in ratios])
    coarse_values = np.array(
        [amplitude(coarse_by_ratio[ratio]["rows"]) for ratio in ratios]
    )
    slope, intercept = np.polyfit(np.log(winds), np.log(values), 1)
    predicted = intercept + slope * np.log(winds)
    centered = np.log(values) - np.mean(np.log(values))
    r_squared = float(
        1.0
        - np.sum((np.log(values) - predicted) ** 2)
        / max(np.sum(centered**2), 1.0e-30)
    )
    rung_shifts = [relative_difference(a, b) for a, b in zip(coarse_values, values)]
    converged = max(rung_shifts) <= CONVERGENCE_RELATIVE_TOLERANCE
    linear = 0.75 <= slope <= 1.25 and r_squared >= TAIL_R2_MINIMUM
    return {
        "field": field_name,
        "wind_speeds": [float(value_) for value_ in winds],
        "outer_potential_equivalent_amplitudes": [float(value_) for value_ in values],
        "power_of_U": float(slope),
        "r_squared": r_squared,
        "adjacent_rung_relative_shifts": rung_shifts,
        "resolution_converged": converged,
        "O_U": bool(converged and linear),
        "verdict": (
            "converged_O_U"
            if converged and linear
            else "converged_not_O_U"
            if converged
            else "unresolved_under_resolution"
        ),
    }


def galilean_null(apparatus, levels: list[dict], winds: list[float]) -> dict:
    rows = []
    for level in levels:
        for wind in winds:
            shape = level["sigma"].shape
            velocity = np.zeros((3,) + shape)
            velocity[0].fill(wind)
            density = np.ones(shape)
            initial_v = velocity.copy()
            initial_n = density.copy()
            dt = min(apparatus.CFL * level["spacing"] / wind, 0.4)
            velocity, density, velocity_rhs, density_rhs, consumption = apparatus.advance(
                velocity,
                density,
                np.zeros_like(velocity),
                level["spacing"],
                wind,
                dt,
            )
            invariant = max(
                float(np.max(np.abs(velocity - initial_v))),
                float(np.max(np.abs(density - initial_n))),
                float(np.max(np.abs(velocity_rhs))),
                float(np.max(np.abs(density_rhs))),
                float(np.max(np.abs(consumption))),
            )
            rows.append(
                {
                    "grid_size": len(level["axis"]),
                    "wind_speed": wind,
                    "certifying_step": dt,
                    "certified_horizon": END_TIME,
                    "maximum_change_or_rhs": invariant,
                    "fixed_point_induction": (
                        "the deterministic update maps the uniform state exactly to "
                        "itself, so repeated steps remain identical through T"
                    ),
                    "passed": bool(invariant <= np.finfo(float).eps),
                }
            )
    return {"cases": rows, "passed": all(row["passed"] for row in rows)}


def run_experiment() -> dict:
    np.random.seed(SEED)
    apparatus = load_predecessor()
    configure_apparatus(apparatus)
    stencil_hash = apparatus.frozen_stencil_sha256()
    if stencil_hash != EXPECTED_STENCIL_SHA256:
        raise RuntimeError(f"frozen stencil changed: {stencil_hash}")

    levels = [apparatus.solve_draw_level(size) for size in GRID_SIZES]
    wind_reference = apparatus.wind_reference(levels[-1])
    winds = [ratio * wind_reference for ratio in WIND_RATIOS]
    dummy_marched = {
        (ratio, radius): {f"l{order}": 0.0 for order in range(4)}
        for ratio in WIND_RATIOS
        for radius in SHELL_RADII
    }
    rungs = []
    reductions = {}
    base_cases = {}
    for level in levels:
        grid_size = len(level["axis"])
        rung_cases = []
        for ratio, wind in zip(WIND_RATIOS, winds):
            case, (velocity, _density) = apparatus.run_case(
                level, wind, ratio, dummy_marched
            )
            rows = shell_reduction(apparatus, level, velocity, wind)
            fits = radial_fits(rows)
            reductions[(grid_size, ratio)] = {"wind": wind, "rows": rows, "fits": fits}
            base_cases[(grid_size, ratio)] = case
            rung_cases.append(
                {
                    "wind_ratio_to_v_GP_at_r5": ratio,
                    "wind_speed": wind,
                    "steadiness": case["steadiness"],
                    "mass_budget": case["mass_budget"],
                    "shells": rows,
                    "radial_fits": fits,
                }
            )
        rungs.append(
            {
                "grid_size": grid_size,
                "spacing": level["spacing"],
                "core_sigma_in_cells": CORE_SIGMA / level["spacing"],
                "draw_strength_recovered": level["draw_strength_recovered"],
                "cases": rung_cases,
            }
        )

    coarse_size, fine_size = GRID_SIZES
    convergence_by_wind = []
    slots_by_wind = []
    for ratio in WIND_RATIOS:
        coarse = reductions[(coarse_size, ratio)]
        fine = reductions[(fine_size, ratio)]
        convergence = converge_fits(coarse["fits"], fine["fits"])
        convergence_by_wind.append(
            {
                "wind_ratio_to_v_GP_at_r5": ratio,
                "wind_speed": fine["wind"],
                "fields": convergence,
            }
        )
        slots_by_wind.append(
            {
                "wind_ratio_to_v_GP_at_r5": ratio,
                "wind_speed": fine["wind"],
                "slots": slot_extractions(
                    coarse["rows"], fine["rows"], convergence, fine["wind"]
                ),
            }
        )

    coarse_by_ratio = {
        ratio: reductions[(coarse_size, ratio)] for ratio in WIND_RATIOS
    }
    fine_by_ratio = {ratio: reductions[(fine_size, ratio)] for ratio in WIND_RATIOS}
    matching = []
    for ratio, wind in zip(WIND_RATIOS, winds):
        rows = reductions[(fine_size, ratio)]["rows"]
        wind_rows = [row for row in rows if row["wind_dominated"]]
        first = wind_rows[0] if wind_rows else None
        matching.append(
            {
                "wind_ratio_to_v_GP_at_r5": ratio,
                "wind_speed": wind,
                "first_wind_dominated_shell": None if first is None else first["radius"],
                "outer_shell_local_wind_to_GP_ratio": rows[-1]["local_wind_to_GP_ratio"],
                "core_region_to_first_wind_dominated_radius": (
                    None if first is None else CORE_REGION_RADIUS / first["radius"]
                ),
                "first_wind_dominated_radius_to_box_half_width": (
                    None if first is None else first["radius"] / DOMAIN_HALF_WIDTH
                ),
            }
        )

    g1_cases = []
    cavitation_kills = []
    for ratio in WIND_RATIOS:
        candidates = []
        for size in GRID_SIZES:
            case = base_cases[(size, ratio)]
            trough = case["steadiness"]["trough_saturation"]
            candidates.append(
                {
                    "grid_size": size,
                    "joint_verdict": case["steadiness"]["verdict"],
                    "trough": trough,
                }
            )
            g1_cases.append(
                {
                    "grid_size": size,
                    "wind_ratio_to_v_GP_at_r5": ratio,
                    "wind_speed": case["wind_speed"],
                    **case["steadiness"],
                }
            )
        branch_kill = bool(
            any(row["trough"]["cavitation"]["cutoff_reached"] for row in candidates)
            or all(row["trough"]["cavitation"]["candidate"] for row in candidates)
        )
        cavitation_kills.append(
            {
                "wind_ratio_to_v_GP_at_r5": ratio,
                "by_rung": candidates,
                "claim_ppn_far_field_trans_critical_killed": branch_kill,
            }
        )

    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "run_date": RUN_DATE,
        "seed": SEED,
        "apparatus": {
            "implementation_decision": (
                "new slug because the large-box/small-core geometry and far-field "
                "question differ from ORB-10938; its evolution module is imported "
                "instead of duplicating more than 50 helper lines"
            ),
            "shared_apparatus_source": "../level-core-dynamical-relaxation/main.py",
            "equations": {
                "momentum": "dv/dt + (v.grad)v = c^2 grad sigma",
                "continuity": "dn/dt + div(n v) = -s",
                "consumption": "s=n*sqrt((3/2)*e_dev:e_dev)",
                "Bernoulli_speed_imposed": False,
            },
            "expected_stencil_sha256": EXPECTED_STENCIL_SHA256,
            "measured_stencil_sha256": stencil_hash,
            "stencil_sha256_match": stencil_hash == EXPECTED_STENCIL_SHA256,
            "domain_half_width": DOMAIN_HALF_WIDTH,
            "core_sigma": CORE_SIGMA,
            "core_region_radius": CORE_REGION_RADIUS,
            "grid_sizes": list(GRID_SIZES),
            "shell_radii": list(SHELL_RADII),
            "wind_ratios_to_v_GP_at_r5": list(WIND_RATIOS),
            "wind_speeds": winds,
            "v_GP_at_r5_on_fine_rung": wind_reference,
            "end_time": END_TIME,
            "scale_separation": matching,
            "feasibility_decision": (
                "The 48-unit box doubles ORB-10938's width while a one-unit core "
                "keeps all three matching transitions inside the shell ladder.  "
                "The 41^3->57^3 ladder prioritizes three winds and a complete "
                "T=480 settlement test; any failed resolution gate is reported as "
                "no scaling regime rather than extrapolated."
            ),
            "obstruction_tail_convention": (
                "curl fields are one derivative above g0i; r times each obstruction "
                "is fitted to the Phi~1/r potential ladder"
            ),
            "slot_normalization_scope": (
                "dimensionless lattice proxies only; no observational or physical-unit alpha claim"
            ),
        },
        "rungs": rungs,
        "gates": {
            "G1_settlement_and_trough_control": {
                "criterion_inherited_from_ORB_10938": True,
                "cases": g1_cases,
                "cavitation_claim_gate": cavitation_kills,
                "passed": bool(
                    all(row["steady"] for row in g1_cases)
                    and not any(
                        row["claim_ppn_far_field_trans_critical_killed"]
                        for row in cavitation_kills
                    )
                ),
            },
            "G2_radial_falloff": {
                "predeclared_worst_verdict": "no_scaling_regime",
                "per_wind_rung_convergence": convergence_by_wind,
            },
            "G3_slot_coefficients_lattice_units": {
                "normalization_warning": (
                    "These are apparatus-defined dimensionless lattice proxies, not physical alpha values."
                ),
                "per_wind": slots_by_wind,
            },
            "G4_per_wind_scaling": {
                "outer_shell_radius": SHELL_RADII[-1],
                "fields": [
                    wind_scaling(fine_by_ratio, coarse_by_ratio, field)
                    for field in ("vorticity", "Bernoulli_anisotropy", "total")
                ],
            },
            "G5_Galilean_null": galilean_null(apparatus, levels, winds),
        },
    }


def canonical_bytes(result: dict) -> bytes:
    return json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False, separators=(",", ": ")
    ).encode()


def write_results(result: dict) -> None:
    encoded = canonical_bytes(result) + b"\n"
    root = Path(__file__).parent
    for relative in (Path("assets/results.json"), Path("runs") / RUN_RECORD):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="execute twice and require byte-identical canonical JSON",
    )
    args = parser.parse_args()
    result = run_experiment()
    if args.check_determinism:
        repeated = run_experiment()
        if canonical_bytes(result) != canonical_bytes(repeated):
            raise RuntimeError("determinism check failed")
    write_results(result)
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "G1_passed": result["gates"]["G1_settlement_and_trough_control"]["passed"],
                "G5_passed": result["gates"]["G5_Galilean_null"]["passed"],
                "results": "assets/results.json",
                "determinism_checked": args.check_determinism,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
