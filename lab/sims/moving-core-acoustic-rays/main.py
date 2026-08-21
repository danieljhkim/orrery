"""Bracket photon-sector effects of candidate moving-core velocity fields.

Null rays obey the stationary acoustic Hamiltonian
``H(x,k)=v(x).k+|k|``.  The fixture compares the exact static
Painleve-Gullstrand field with a Galilean wind cartoon, the clipped
Hamilton-Jacobi branch of ORB-10935, and regenerated finest-rung T=60 fields
from ORB-10937.  It measures deflection, pure-wind-subtracted transit delay,
the explicitly aberrated boosted-Schwarzschild comparator, and the spread
across the three unsettled moving direction fields.

Usage:
    uv run lab/sims/moving-core-acoustic-rays/main.py
    uv run lab/sims/moving-core-acoustic-rays/main.py --check-determinism
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

TASK_ID = "ORB-10939"
RUN_ID = "jrun-20260821-0608-4"
RUN_DATE = "2026-08-21"
RUN_RECORD = "2026-08-21-seed-42.json"
SEED = 42
WIND_RATIOS = (0.03, 0.1, 0.3, 1.0)
IMPACT_PARAMETERS = (2.0, 3.0, 5.0)
HEADLINE_IMPACT = 3.0
ORIENTATIONS = {
    "downwind": 0.0,
    "upwind": np.pi,
    "transverse_plus": 0.5 * np.pi,
    "transverse_minus": -0.5 * np.pi,
}
RAY_DOMAIN_HALF_LENGTH = 9.0
RAY_MAX_STEPS = (0.24, 0.12, 0.06)
G1_IMPACT_OVER_RS = (8.0, 12.0, 20.0, 32.0)
G1_DOMAIN_OVER_RS = 4000.0
G1_MAX_STEPS_OVER_RS = (16.0, 8.0, 4.0)
FIELD_RUNGS = {"marched": (61, 81), "realized_T60": (33, 41)}
SOURCE_DIR = Path(__file__).resolve().parents[1]


def encoded(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


@lru_cache(maxsize=None)
def load_source_module(name: str, relative_path: str) -> ModuleType:
    """Load a cataloged predecessor without copying its field solver."""
    path = SOURCE_DIR / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def angle_difference(first: float, second: float) -> float:
    return float(np.arctan2(np.sin(first - second), np.cos(first - second)))


def direction(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def exact_schwarzschild_deflection(impact: float, schwarzschild_radius: float) -> float:
    """Return the infinite-endpoint Schwarzschild bending angle.

    The independent target integrates the standard orbit equation rather than
    the acoustic Hamiltonian used by the ray tracer.
    """
    critical = 1.5 * np.sqrt(3.0) * schwarzschild_radius
    if impact <= critical:
        raise ValueError("impact parameter does not escape the photon sphere")
    turning_radius = brentq(
        lambda radius: radius**3 - impact**2 * radius + impact**2 * schwarzschild_radius,
        1.5 * schwarzschild_radius * (1.0 + 1.0e-12),
        impact,
        xtol=1.0e-14,
        rtol=1.0e-14,
    )
    x0 = 1.0 / turning_radius

    def transformed_integrand(s: float) -> float:
        x = x0 * (1.0 - s * s)
        radicand = 1.0 - impact**2 * x**2 + schwarzschild_radius * impact**2 * x**3
        if s < 1.0e-7:
            derivative = -2.0 * impact**2 * x0 + 3.0 * schwarzschild_radius * impact**2 * x0**2
            return float(2.0 * impact * x0 / np.sqrt(-derivative * x0))
        return float(2.0 * impact * x0 * s / np.sqrt(max(radicand, 1.0e-300)))

    half_orbit = quad(
        transformed_integrand,
        0.0,
        1.0,
        epsabs=2.0e-13,
        epsrel=2.0e-13,
        limit=300,
    )[0]
    return float(2.0 * half_orbit - np.pi)


@dataclass
class AnalyticGPField:
    schwarzschild_radius: float
    wind: float = 0.0

    def sample(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        radius = float(np.linalg.norm(position))
        factor = -np.sqrt(self.schwarzschild_radius) * radius ** (-1.5)
        velocity = factor * position + np.array([self.wind, 0.0])
        jacobian = factor * (
            np.eye(2) - 1.5 * np.outer(position, position) / radius**2
        )
        return velocity, jacobian, False


@dataclass
class BilinearField:
    """A two-dimensional plane through a cataloged 3-D field.

    Values and derivatives use the same bilinear polynomial in each cell, so
    the Hamiltonian derivative remains internally consistent.  The optional
    mask marks cells touching ORB-10935's clipped caustic branch.
    """

    axis: np.ndarray
    velocity: np.ndarray
    clipped_nodes: np.ndarray | None = None

    def sample(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        spacing = float(self.axis[1] - self.axis[0])
        fractional = (np.asarray(position) - self.axis[0]) / spacing
        lower = np.floor(fractional).astype(int)
        lower = np.clip(lower, 0, len(self.axis) - 2)
        tx, ty = fractional - lower
        ix, iy = int(lower[0]), int(lower[1])
        v00 = self.velocity[:, ix, iy]
        v10 = self.velocity[:, ix + 1, iy]
        v01 = self.velocity[:, ix, iy + 1]
        v11 = self.velocity[:, ix + 1, iy + 1]
        value = (
            (1.0 - tx) * (1.0 - ty) * v00
            + tx * (1.0 - ty) * v10
            + (1.0 - tx) * ty * v01
            + tx * ty * v11
        )
        derivative_x = ((1.0 - ty) * (v10 - v00) + ty * (v11 - v01)) / spacing
        derivative_y = ((1.0 - tx) * (v01 - v00) + tx * (v11 - v10)) / spacing
        touches_clip = False
        if self.clipped_nodes is not None:
            touches_clip = bool(np.any(self.clipped_nodes[ix : ix + 2, iy : iy + 2]))
        return value, np.column_stack((derivative_x, derivative_y)), touches_clip


def launch_wavevector(field: AnalyticGPField | BilinearField, position: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
    velocity, _, _ = field.sample(position)
    projection = float(np.dot(velocity, ray_direction))
    discriminant = 1.0 - float(np.dot(velocity, velocity)) + projection**2
    if discriminant <= 0.0:
        raise RuntimeError("launch point is not in a subsonic asymptotic region")
    coordinate_speed = projection + np.sqrt(discriminant)
    wavevector = coordinate_speed * ray_direction - velocity
    return wavevector / np.linalg.norm(wavevector)


def pure_wind_coordinate_speed(wind: float, ray_direction: np.ndarray) -> float:
    projection = wind * float(ray_direction[0])
    return float(projection + np.sqrt(1.0 - wind**2 + projection**2))


def trace_ray(
    field: AnalyticGPField | BilinearField,
    wind: float,
    impact: float,
    angle: float,
    domain_half_length: float,
    max_step: float,
) -> dict:
    incoming = direction(angle)
    impact_direction = np.array([-incoming[1], incoming[0]])
    initial_position = -domain_half_length * incoming + impact * impact_direction
    initial_wavevector = launch_wavevector(field, initial_position, incoming)
    initial_state = np.concatenate((initial_position, initial_wavevector))
    evaluations = 0
    clipped_evaluations = 0

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        nonlocal evaluations, clipped_evaluations
        velocity, jacobian, clipped = field.sample(state[:2])
        evaluations += 1
        clipped_evaluations += int(clipped)
        wavevector = state[2:]
        norm = float(np.linalg.norm(wavevector))
        return np.concatenate((velocity + wavevector / norm, -(jacobian.T @ wavevector)))

    def exit_plane(_time: float, state: np.ndarray) -> float:
        return float(np.dot(state[:2], incoming) - domain_half_length)

    exit_plane.terminal = True
    exit_plane.direction = 1
    maximum_time = 5.0 * domain_half_length / max(1.0 - abs(wind), 0.1)
    solution = solve_ivp(
        rhs,
        (0.0, maximum_time),
        initial_state,
        method="DOP853",
        events=exit_plane,
        rtol=2.0e-10,
        atol=2.0e-12,
        max_step=max_step,
    )
    if not solution.success or len(solution.t_events[0]) != 1:
        raise RuntimeError(f"ray did not reach its exit plane: {solution.message}")
    final_state = solution.y_events[0][0]
    initial_velocity, _, _ = field.sample(initial_position)
    final_velocity, _, final_clipped = field.sample(final_state[:2])
    initial_group = initial_velocity + initial_wavevector / np.linalg.norm(initial_wavevector)
    final_group = final_velocity + final_state[2:] / np.linalg.norm(final_state[2:])
    initial_angle = float(np.arctan2(initial_group[1], initial_group[0]))
    final_angle = float(np.arctan2(final_group[1], final_group[0]))
    initial_hamiltonian = float(
        np.dot(initial_velocity, initial_wavevector) + np.linalg.norm(initial_wavevector)
    )
    final_hamiltonian = float(
        np.dot(final_velocity, final_state[2:]) + np.linalg.norm(final_state[2:])
    )
    baseline = 2.0 * domain_half_length / pure_wind_coordinate_speed(wind, incoming)
    return {
        "signed_deflection_radians": angle_difference(final_angle, initial_angle),
        "deflection_radians": abs(angle_difference(final_angle, initial_angle)),
        "transit_time": float(solution.t_events[0][0]),
        "pure_wind_transit_time": baseline,
        "shapiro_delay": float(solution.t_events[0][0] - baseline),
        "hamiltonian_relative_drift": abs(final_hamiltonian - initial_hamiltonian)
        / max(abs(initial_hamiltonian), 1.0e-30),
        "rhs_evaluations": evaluations,
        "clipped_cell_evaluation_fraction": float(
            (clipped_evaluations + int(final_clipped)) / max(evaluations + 1, 1)
        ),
    }


def converged_trace(
    field: AnalyticGPField | BilinearField,
    wind: float,
    impact: float,
    angle: float,
    *,
    domain_half_length: float = RAY_DOMAIN_HALF_LENGTH,
    max_steps: tuple[float, ...] = RAY_MAX_STEPS,
) -> dict:
    traces = [
        trace_ray(field, wind, impact, angle, domain_half_length, step)
        for step in max_steps
    ]
    finest = dict(traces[-1])
    finest["integrator_convergence"] = {
        "maximum_steps": list(max_steps),
        "deflections_radians": [row["deflection_radians"] for row in traces],
        "shapiro_delays": [row["shapiro_delay"] for row in traces],
        "deflection_maximum_shift_from_finest": float(
            max(abs(row["deflection_radians"] - finest["deflection_radians"]) for row in traces[:-1])
        ),
        "delay_maximum_shift_from_finest": float(
            max(abs(row["shapiro_delay"] - finest["shapiro_delay"]) for row in traces[:-1])
        ),
    }
    return finest


def aberrate(direction_lab_or_rest: np.ndarray, beta: float, *, inverse: bool) -> np.ndarray:
    """Lorentz-aberrate a null direction along x.

    ``beta`` is the source velocity in the lab.  ``inverse=True`` maps lab to
    the source rest frame; the forward map returns to the lab.
    """
    nx, ny = map(float, direction_lab_or_rest)
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    if inverse:
        denominator = 1.0 - beta * nx
        mapped = np.array([(nx - beta) / denominator, ny / (gamma * denominator)])
    else:
        denominator = 1.0 + beta * nx
        mapped = np.array([(nx + beta) / denominator, ny / (gamma * denominator)])
    return mapped / np.linalg.norm(mapped)


def gr_comparator(
    static_deflection: float,
    static_delay: float,
    wind: float,
    angle: float,
) -> dict:
    lab_incoming = direction(angle)
    beta_source = -wind
    rest_incoming = aberrate(lab_incoming, beta_source, inverse=True)
    rest_angle = float(np.arctan2(rest_incoming[1], rest_incoming[0]))
    rest_outgoing = direction(rest_angle - static_deflection)
    lab_outgoing = aberrate(rest_outgoing, beta_source, inverse=False)
    lab_outgoing_angle = float(np.arctan2(lab_outgoing[1], lab_outgoing[0]))
    lab_deflection = abs(angle_difference(lab_outgoing_angle, angle))
    gamma = 1.0 / np.sqrt(1.0 - beta_source**2)
    doppler = gamma * (1.0 + beta_source * float(rest_incoming[0]))
    return {
        "deflection_radians": lab_deflection,
        "shapiro_delay": static_delay / doppler,
        "source_beta_in_lab": beta_source,
        "lab_incoming_direction": lab_incoming.tolist(),
        "mass_rest_incoming_direction": rest_incoming.tolist(),
        "mass_rest_outgoing_direction": rest_outgoing.tolist(),
        "lab_outgoing_direction": lab_outgoing.tolist(),
        "phase_delay_doppler_factor": float(doppler),
    }


def inferred_caustic_mask(velocity: np.ndarray, sigma: np.ndarray, wind: float) -> np.ndarray:
    speed = np.sqrt(wind**2 + 2.0 * sigma)
    transverse = np.sqrt(velocity[1] ** 2 + velocity[2] ** 2)
    return transverse >= 0.998999999 * speed


def mask_summary(mask: np.ndarray, axis: np.ndarray) -> dict:
    indices = np.argwhere(mask)
    bounds = None
    if len(indices):
        bounds = {
            "minimum": [float(axis[int(value)]) for value in indices.min(axis=0)],
            "maximum": [float(axis[int(value)]) for value in indices.max(axis=0)],
        }
    return {
        "inference": "nodes whose returned transverse speed is at the source solver's 0.999*q limiter",
        "node_count": int(np.count_nonzero(mask)),
        "node_fraction": float(np.mean(mask)),
        "sha256_uint8_C_order": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
        "coordinate_bounds": bounds,
    }


def source_case(run: dict, size: int, ratio: float, case_key: str) -> dict:
    rung = next(row for row in run["resolution_ladder"] if row["apparatus"]["grid_size"] == size)
    return next(row for row in rung[case_key] if float(row["wind_ratio_to_finest_v_GP_at_probe"]) == ratio)


def build_fields() -> dict:
    wind_module = load_source_module(
        "orrery_level_core_wind_tunnel_source",
        "level-core-wind-tunnel/main.py",
    )
    dynamic_module = load_source_module(
        "orrery_level_core_dynamic_source",
        "level-core-dynamical-relaxation/main.py",
    )
    wind_run = json.loads(
        (SOURCE_DIR / "level-core-wind-tunnel/runs/2026-08-21-seed-42.json").read_text()
    )
    dynamic_run = json.loads(
        (SOURCE_DIR / "level-core-dynamical-relaxation/runs/2026-08-21-seed-42.json").read_text()
    )

    marched_levels = {
        size: wind_module.solve_draw_level(size) for size in FIELD_RUNGS["marched"]
    }
    static_reference = wind_module.wind_speed_reference(marched_levels[81])
    schwarzschild_radius = static_reference**2 * wind_module.PROBE_RADIUS
    marched_fields: dict[int, dict[float, BilinearField]] = {61: {}, 81: {}}
    marched_source: dict[str, dict] = {}
    for size, level in marched_levels.items():
        center = size // 2
        for ratio in WIND_RATIOS:
            wind = ratio * static_reference
            velocity, diagnostic = wind_module.march_steady_branch(
                level["sigma"], level["spacing"], wind
            )
            mask = inferred_caustic_mask(velocity, level["sigma"], wind)
            marched_fields[size][ratio] = BilinearField(
                level["axis"], velocity[:2, :, :, center], mask[:, :, center]
            )
            key = f"{size}:{ratio:g}"
            marched_source[key] = {
                "wind_speed": wind,
                "source_diagnostic": diagnostic,
                "caustic_mask": mask_summary(mask, level["axis"]),
            }
            if size == 81:
                stored = source_case(wind_run, 81, ratio, "wind_measurements")
                marched_source[key]["stored_caustic_clip_fraction"] = stored["steady_wake"][
                    "caustic_clip_fraction"
                ]
                marched_source[key]["regenerated_minus_stored_clip_fraction"] = (
                    diagnostic["caustic_clip_fraction"]
                    - stored["steady_wake"]["caustic_clip_fraction"]
                )

    dynamic_module.END_TIME = 60.0
    dynamic_module.DIAGNOSTIC_INTERVAL = 10.0
    dynamic_module.STEADY_WINDOW_SAMPLES = 4
    dynamic_levels = {
        size: dynamic_module.solve_draw_level(size)
        for size in FIELD_RUNGS["realized_T60"]
    }
    dynamic_reference = dynamic_module.wind_reference(dynamic_levels[41])
    marched_comparator = dynamic_module.load_marched_comparator()
    realized_fields: dict[int, dict[float, BilinearField]] = {33: {}, 41: {}}
    realized_source: dict[str, dict] = {}
    for size, level in dynamic_levels.items():
        center = size // 2
        for ratio in WIND_RATIOS:
            wind = ratio * dynamic_reference
            case, (velocity, density) = dynamic_module.run_case(
                level, wind, ratio, marched_comparator
            )
            realized_fields[size][ratio] = BilinearField(
                level["axis"], velocity[:2, :, :, center]
            )
            history = case["steadiness"]["residual_time_series"]
            key = f"{size}:{ratio:g}"
            regenerated = {
                "integration_steps": case["integration"]["steps"],
                "final_dv_rms": history[-1]["dv_rms"],
                "final_dn_rms": history[-1]["dn_rms"],
                "minimum_density": float(np.min(density)),
            }
            realized_source[key] = {
                "wind_speed": wind,
                "regenerated_T60": regenerated,
                "frozen_background_caveat": "ORB-10937's finest-rung T=60 velocity residual passed its sector criterion, but density was still relaxing; this ray pass freezes that finite-time state and does not claim attractor uniqueness.",
            }
            if size == 41:
                stored = source_case(dynamic_run, 41, ratio, "cases")
                expected = {
                    "integration_steps": stored["integration"]["steps"],
                    "final_dv_rms": stored["steadiness"]["residual_time_series"][-1]["dv_rms"],
                    "final_dn_rms": stored["steadiness"]["residual_time_series"][-1]["dn_rms"],
                    "minimum_density": stored["steadiness"]["residual_time_series"][-1][
                        "minimum_density"
                    ],
                }
                realized_source[key]["stored_ORB_10937"] = expected
                realized_source[key]["absolute_regeneration_errors"] = {
                    field: abs(float(regenerated[field]) - float(expected[field]))
                    for field in regenerated
                }

    return {
        "schwarzschild_radius": schwarzschild_radius,
        "static_reference_speed": static_reference,
        "dynamic_reference_speed": dynamic_reference,
        "marched": marched_fields,
        "realized": realized_fields,
        "source_diagnostics": {
            "marched": marched_source,
            "realized_T60": realized_source,
        },
    }


def static_baselines(schwarzschild_radius: float) -> dict[float, dict]:
    field = AnalyticGPField(schwarzschild_radius)
    return {
        impact: converged_trace(field, 0.0, impact, 0.0)
        for impact in IMPACT_PARAMETERS
    }


def summarize_orientations(
    traces: dict[str, dict],
    gr: dict[str, dict],
    static: dict,
    wind: float,
) -> dict:
    static_alpha = static["deflection_radians"]
    static_delay = static["shapiro_delay"]
    rows = {}
    for name in ORIENTATIONS:
        closed_modulation = (traces[name]["deflection_radians"] - static_alpha) / static_alpha
        gr_modulation = (gr[name]["deflection_radians"] - static_alpha) / static_alpha
        rows[name] = {
            "closed_system": traces[name],
            "boosted_schwarzschild": gr[name],
            "closed_modulation_fraction": closed_modulation,
            "GR_modulation_fraction": gr_modulation,
            "closed_minus_GR_modulation_fraction": closed_modulation - gr_modulation,
        }
    closed_modulations = np.array([rows[name]["closed_modulation_fraction"] for name in ORIENTATIONS])
    gr_modulations = np.array([rows[name]["GR_modulation_fraction"] for name in ORIENTATIONS])
    differentials = closed_modulations - gr_modulations
    closed_delays = {name: traces[name]["shapiro_delay"] for name in ORIENTATIONS}
    gr_delays = {name: gr[name]["shapiro_delay"] for name in ORIENTATIONS}
    closed_delay_asymmetry = (closed_delays["downwind"] - closed_delays["upwind"]) / (
        2.0 * max(abs(static_delay), 1.0e-30)
    )
    gr_delay_asymmetry = (gr_delays["downwind"] - gr_delays["upwind"]) / (
        2.0 * max(abs(static_delay), 1.0e-30)
    )
    return {
        "static_deflection_radians": static_alpha,
        "static_shapiro_delay": static_delay,
        "orientations": rows,
        "deflection_modulation": {
            "overall_half_range_fraction": float(0.5 * np.ptp(closed_modulations)),
            "fore_aft_half_difference_fraction": float(
                0.5
                * abs(
                    rows["downwind"]["closed_modulation_fraction"]
                    - rows["upwind"]["closed_modulation_fraction"]
                )
            ),
            "transverse_half_difference_fraction": float(
                0.5
                * abs(
                    rows["transverse_plus"]["closed_modulation_fraction"]
                    - rows["transverse_minus"]["closed_modulation_fraction"]
                )
            ),
            "overall_fraction_per_unit_U_over_c": float(
                0.5 * np.ptp(closed_modulations) / wind
            ),
        },
        "GR_deflection_modulation": {
            "overall_half_range_fraction": float(0.5 * np.ptp(gr_modulations)),
            "fore_aft_half_difference_fraction": float(
                0.5
                * abs(
                    rows["downwind"]["GR_modulation_fraction"]
                    - rows["upwind"]["GR_modulation_fraction"]
                )
            ),
        },
        "closed_minus_GR_deflection": {
            "overall_half_range_fraction": float(0.5 * np.ptp(differentials)),
            "fore_aft_half_difference_fraction": float(
                0.5
                * abs(
                    rows["downwind"]["closed_minus_GR_modulation_fraction"]
                    - rows["upwind"]["closed_minus_GR_modulation_fraction"]
                )
            ),
        },
        "shapiro_fore_aft": {
            "closed_downwind_delay": closed_delays["downwind"],
            "closed_upwind_delay": closed_delays["upwind"],
            "closed_normalized_asymmetry": float(closed_delay_asymmetry),
            "GR_downwind_delay": gr_delays["downwind"],
            "GR_upwind_delay": gr_delays["upwind"],
            "GR_normalized_asymmetry": float(gr_delay_asymmetry),
            "closed_minus_GR_normalized_asymmetry": float(
                closed_delay_asymmetry - gr_delay_asymmetry
            ),
        },
    }


def analyze_impact(
    field: AnalyticGPField | BilinearField,
    wind: float,
    impact: float,
    static: dict,
    *,
    convergence: bool,
) -> dict:
    steps = RAY_MAX_STEPS if convergence else (RAY_MAX_STEPS[-1],)
    traces = {
        name: converged_trace(field, wind, impact, angle, max_steps=steps)
        if convergence
        else trace_ray(field, wind, impact, angle, RAY_DOMAIN_HALF_LENGTH, steps[0])
        for name, angle in ORIENTATIONS.items()
    }
    gr = {
        name: gr_comparator(
            static["deflection_radians"], static["shapiro_delay"], wind, angle
        )
        for name, angle in ORIENTATIONS.items()
    }
    return summarize_orientations(traces, gr, static, wind)


def power_fit(winds: list[float], values: list[float]) -> dict:
    values_array = np.abs(np.asarray(values, dtype=float))
    winds_array = np.asarray(winds, dtype=float)
    floor = 1.0e-15
    log_winds = np.log(winds_array)
    log_values = np.log(np.maximum(values_array, floor))
    exponent, log_prefactor = np.polyfit(log_winds, log_values, 1)
    fitted = exponent * log_winds + log_prefactor
    residual = float(np.sum((log_values - fitted) ** 2))
    total = float(np.sum((log_values - np.mean(log_values)) ** 2))
    return {
        "form": "amplitude=A*(U/c)^p",
        "A": float(np.exp(log_prefactor)),
        "p": float(exponent),
        "log_log_R_squared": 1.0 - residual / max(total, 1.0e-30),
        "winds_U_over_c": winds,
        "amplitudes": values,
        "monotonic_absolute_amplitude": bool(
            np.all(np.diff(values_array) >= 0.0) or np.all(np.diff(values_array) <= 0.0)
        ),
        "absolute_value_used_for_signed_differential": True,
        "floor_for_log_fit": floor,
    }


def analyze_field(
    fields: dict[float, AnalyticGPField | BilinearField],
    winds: dict[float, float],
    baselines: dict[float, dict],
) -> dict:
    by_wind = []
    for ratio in WIND_RATIOS:
        wind = winds[ratio]
        impacts = [
            {
                "impact_parameter": impact,
                **analyze_impact(fields[ratio], wind, impact, baselines[impact], convergence=True),
            }
            for impact in IMPACT_PARAMETERS
        ]
        by_wind.append(
            {
                "wind_ratio_to_v_GP_at_r5": ratio,
                "wind_U_over_c": wind,
                "impact_cases": impacts,
            }
        )
    fits = {}
    for impact in IMPACT_PARAMETERS:
        cases = [
            next(row for row in wind_case["impact_cases"] if row["impact_parameter"] == impact)
            for wind_case in by_wind
        ]
        fit_winds = [row["wind_U_over_c"] for row in by_wind]
        fits[f"b={impact:g}"] = {
            "G2_closed_deflection_modulation": power_fit(
                fit_winds,
                [row["deflection_modulation"]["overall_half_range_fraction"] for row in cases],
            ),
            "G3_closed_minus_GR_deflection": power_fit(
                fit_winds,
                [row["closed_minus_GR_deflection"]["overall_half_range_fraction"] for row in cases],
            ),
            "G4_closed_shapiro_asymmetry": power_fit(
                fit_winds,
                [row["shapiro_fore_aft"]["closed_normalized_asymmetry"] for row in cases],
            ),
            "G4_closed_minus_GR_shapiro": power_fit(
                fit_winds,
                [row["shapiro_fore_aft"]["closed_minus_GR_normalized_asymmetry"] for row in cases],
            ),
        }
    return {"by_wind": by_wind, "leading_U_exponents": fits}


def impact_case(field_result: dict, ratio: float, impact: float) -> dict:
    wind = next(row for row in field_result["by_wind"] if row["wind_ratio_to_v_GP_at_r5"] == ratio)
    return next(row for row in wind["impact_cases"] if row["impact_parameter"] == impact)


def field_resolution(
    coarse_fields: dict[float, BilinearField],
    fine_result: dict,
    winds: dict[float, float],
    baselines: dict[float, dict],
    coarse_size: int,
    fine_size: int,
) -> list[dict]:
    rows = []
    for ratio in WIND_RATIOS:
        coarse = analyze_impact(
            coarse_fields[ratio], winds[ratio], HEADLINE_IMPACT, baselines[HEADLINE_IMPACT], convergence=False
        )
        fine = impact_case(fine_result, ratio, HEADLINE_IMPACT)
        rows.append(
            {
                "wind_ratio": ratio,
                "coarse_grid_size": coarse_size,
                "fine_grid_size": fine_size,
                "impact_parameter": HEADLINE_IMPACT,
                "absolute_shift": {
                    "G2_overall_modulation_fraction": abs(
                        fine["deflection_modulation"]["overall_half_range_fraction"]
                        - coarse["deflection_modulation"]["overall_half_range_fraction"]
                    ),
                    "G3_differential_modulation_fraction": abs(
                        fine["closed_minus_GR_deflection"]["overall_half_range_fraction"]
                        - coarse["closed_minus_GR_deflection"]["overall_half_range_fraction"]
                    ),
                    "G4_closed_minus_GR_shapiro_asymmetry": abs(
                        fine["shapiro_fore_aft"]["closed_minus_GR_normalized_asymmetry"]
                        - coarse["shapiro_fore_aft"]["closed_minus_GR_normalized_asymmetry"]
                    ),
                },
            }
        )
    return rows


def g1_comparator() -> dict:
    cases = []
    field = AnalyticGPField(1.0)
    for impact in G1_IMPACT_OVER_RS:
        trace = converged_trace(
            field,
            0.0,
            impact,
            0.0,
            domain_half_length=G1_DOMAIN_OVER_RS,
            max_steps=G1_MAX_STEPS_OVER_RS,
        )
        exact = exact_schwarzschild_deflection(impact, 1.0)
        weak = 2.0 / impact
        cases.append(
            {
                "impact_parameter_over_r_s": impact,
                "measured_deflection_radians": trace["deflection_radians"],
                "exact_schwarzschild_deflection_radians": exact,
                "weak_field_4GM_over_c2b_radians": weak,
                "exact_strong_field_correction_fraction": (exact - weak) / weak,
                "measured_minus_exact_radians": trace["deflection_radians"] - exact,
                "relative_error_against_exact": abs(trace["deflection_radians"] - exact) / exact,
                "integrator_convergence": trace["integrator_convergence"],
                "hamiltonian_relative_drift": trace["hamiltonian_relative_drift"],
            }
        )
    passed = all(
        row["relative_error_against_exact"] < 5.0e-5
        and row["integrator_convergence"]["deflection_maximum_shift_from_finest"] < 1.0e-8
        for row in cases
    )
    return {
        "verdict": "pass" if passed else "kill_static_comparator_mismatch",
        "kill_gate": True,
        "criterion": "all infinite-endpoint target errors <5e-5 relative and maximum-step refinement shifts <1e-8 rad",
        "independent_target": "exact Schwarzschild null-orbit quadrature; weak 4GM/(c^2 b)=2r_s/b is reported separately",
        "finite_endpoint_control": f"ray endpoints are +/-{G1_DOMAIN_OVER_RS:g} r_s; the largest b/end ratio is {max(G1_IMPACT_OVER_RS)/G1_DOMAIN_OVER_RS:g}",
        "cases": cases,
        "passed": passed,
    }


def cross_field_spread(field_results: dict[str, dict]) -> dict:
    cases = []
    for ratio in WIND_RATIOS:
        for impact in IMPACT_PARAMETERS:
            per_field = {
                name: impact_case(result, ratio, impact)
                for name, result in field_results.items()
            }
            g2 = {
                name: row["deflection_modulation"]["overall_half_range_fraction"]
                for name, row in per_field.items()
            }
            g3 = {
                name: row["closed_minus_GR_deflection"]["overall_half_range_fraction"]
                for name, row in per_field.items()
            }
            g4 = {
                name: row["shapiro_fore_aft"]["closed_minus_GR_normalized_asymmetry"]
                for name, row in per_field.items()
            }
            cases.append(
                {
                    "wind_ratio": ratio,
                    "impact_parameter": impact,
                    "G2_modulation_amplitude_by_field": g2,
                    "G2_cross_field_spread": max(g2.values()) - min(g2.values()),
                    "G3_differential_amplitude_by_field": g3,
                    "G3_cross_field_spread": max(g3.values()) - min(g3.values()),
                    "G4_GR_differential_asymmetry_by_field": g4,
                    "G4_cross_field_spread": max(g4.values()) - min(g4.values()),
                }
            )
    headline = [row for row in cases if row["impact_parameter"] == HEADLINE_IMPACT]
    return {
        "verdict": "measured_primary_deliverable",
        "primary_deliverable": "The cross-field range is the measured dependence of photon-sector scale estimates on the unsettled wake direction field; it is not an observational bound or a falsifier.",
        "headline_impact_parameter": HEADLINE_IMPACT,
        "headline_by_wind": headline,
        "maximum_over_ladder": {
            "G2_modulation_fraction_spread": max(row["G2_cross_field_spread"] for row in cases),
            "G3_differential_fraction_spread": max(row["G3_cross_field_spread"] for row in cases),
            "G4_differential_asymmetry_spread": max(abs(row["G4_cross_field_spread"]) for row in cases),
        },
        "all_cases": cases,
    }


def validate(results: dict) -> None:
    if not results["gates"]["G1_static_comparator"]["passed"]:
        raise AssertionError("G1 static Schwarzschild comparator failed")
    expected_fields = {"(b)_galilean", "(c)_marched_HJ", "(d)_realized_T60"}
    if set(results["field_results"]) != expected_fields:
        raise AssertionError("moving-field bracket is incomplete")
    for field in results["field_results"].values():
        if len(field["by_wind"]) != len(WIND_RATIOS):
            raise AssertionError("four-wind ladder is incomplete")
        for wind in field["by_wind"]:
            if len(wind["impact_cases"]) != len(IMPACT_PARAMETERS):
                raise AssertionError("impact-parameter ladder is incomplete")
            for impact in wind["impact_cases"]:
                for orientation in impact["orientations"].values():
                    convergence = orientation["closed_system"]["integrator_convergence"]
                    if convergence["deflection_maximum_shift_from_finest"] > 2.0e-5:
                        raise AssertionError("moving-field ray deflection did not converge")
                    if convergence["delay_maximum_shift_from_finest"] > 2.0e-4:
                        raise AssertionError("moving-field ray delay did not converge")
    source = results["field_sources"]["realized_T60"]
    for ratio in WIND_RATIOS:
        errors = source[f"41:{ratio:g}"]["absolute_regeneration_errors"]
        if max(errors.values()) > 1.0e-12:
            raise AssertionError("T=60 field regeneration does not match ORB-10937")
    if len(results["gates"]["G5_cross_field_spread"]["all_cases"]) != (
        len(WIND_RATIOS) * len(IMPACT_PARAMETERS)
    ):
        raise AssertionError("G5 cross-field ladder is incomplete")


def experiment() -> tuple[dict, dict]:
    fields = build_fields()
    schwarzschild_radius = fields["schwarzschild_radius"]
    baselines = static_baselines(schwarzschild_radius)
    static_reference = fields["static_reference_speed"]
    dynamic_reference = fields["dynamic_reference_speed"]
    static_winds = {ratio: ratio * static_reference for ratio in WIND_RATIOS}
    dynamic_winds = {ratio: ratio * dynamic_reference for ratio in WIND_RATIOS}
    galilean_fields = {
        ratio: AnalyticGPField(schwarzschild_radius, static_winds[ratio])
        for ratio in WIND_RATIOS
    }
    field_results = {
        "(b)_galilean": analyze_field(galilean_fields, static_winds, baselines),
        "(c)_marched_HJ": analyze_field(fields["marched"][81], static_winds, baselines),
        "(d)_realized_T60": analyze_field(fields["realized"][41], dynamic_winds, baselines),
    }
    convergence = {
        "(b)_galilean": {
            "field_representation": "analytic exact point-mass GP plus constant wind; no field-grid interpolation error"
        },
        "(c)_marched_HJ": field_resolution(
            fields["marched"][61], field_results["(c)_marched_HJ"], static_winds, baselines, 61, 81
        ),
        "(d)_realized_T60": field_resolution(
            fields["realized"][33], field_results["(d)_realized_T60"], dynamic_winds, baselines, 33, 41
        ),
    }
    g1 = g1_comparator()
    g5 = cross_field_spread(field_results)
    gates = {
        "G1_static_comparator": g1,
        "G2_deflection_modulation": {
            "verdict": "four_winds_four_orientations_three_impacts_recorded",
            "headline_number_definition": "overall orientation half-range of (alpha-alpha_static)/alpha_static divided by U/c",
            "by_field": {
                name: {
                    "leading_U_exponents": result["leading_U_exponents"],
                    "headline_by_wind": [
                        {
                            "wind_ratio": row["wind_ratio_to_v_GP_at_r5"],
                            "wind_U_over_c": row["wind_U_over_c"],
                            **next(
                                case["deflection_modulation"]
                                for case in row["impact_cases"]
                                if case["impact_parameter"] == HEADLINE_IMPACT
                            ),
                        }
                        for row in result["by_wind"]
                    ],
                }
                for name, result in field_results.items()
            },
        },
        "G3_boosted_schwarzschild": {
            "verdict": "explicit_aberration_and_closed_minus_GR_recorded",
            "aberration": {
                "source_velocity_in_substrate_lab": "beta_source=-U along x because the source-frame substrate wind is +U",
                "lab_to_mass_rest": "n'_x=(n_x-beta)/(1-beta*n_x), n'_perp=n_perp/[gamma(1-beta*n_x)]",
                "mass_rest_to_lab": "n_x=(n'_x+beta)/(1+beta*n'_x), n_perp=n'_perp/[gamma(1+beta*n'_x)]",
                "static_optics": "apply the measured static Schwarzschild bending in the mass rest frame, then aberrate the outgoing null direction back to the same lab configuration",
                "not_used": "no Galilean U+v_GP field is used for the GR comparator",
            },
            "results_location": "field_results.*.by_wind.*.impact_cases.*.orientations and closed_minus_GR_deflection",
        },
        "G4_shapiro_asymmetry": {
            "verdict": "fore_aft_asymmetry_and_GR_differential_recorded",
            "closed_delay_definition": "ray transit time minus the no-core pure-wind transit time between the same source-frame planes",
            "GR_delay_mapping": "the static mass-rest phase delay is divided by gamma*(1+beta_source*n'_x), the null-wave Doppler factor, to obtain lab coordinate delay at fixed phase",
            "normalization": "(downwind delay-upwind delay)/(2*static delay)",
            "results_location": "field_results.*.by_wind.*.impact_cases.*.shapiro_fore_aft",
        },
        "G5_cross_field_spread": g5,
    }
    apparatus = {
        "acoustic_metric": "ds^2=-(1-|v|^2)dt^2-2v.dx dt+|dx|^2",
        "ray_hamiltonian": "H(x,k)=v(x).k+|k|",
        "hamilton_equations": "dx/dt=v+k/|k|; dk/dt=-(grad v)^T k",
        "integrator": "scipy DOP853 with rtol=2e-10, atol=2e-12 and maximum-step ladder",
        "ray_domain_half_length": RAY_DOMAIN_HALF_LENGTH,
        "impact_parameters": list(IMPACT_PARAMETERS),
        "orientations_radians": ORIENTATIONS,
        "wind_ratios": list(WIND_RATIOS),
        "c": 1.0,
        "static_GP_calibration": {
            "v_GP_at_r5_from_ORB_10935_finest": static_reference,
            "r_s=v_GP(r=5)^2*5": schwarzschild_radius,
            "purpose": "carry the cataloged field scale into the exact point-mass Painleve-Gullstrand comparator",
        },
        "moving_fields": {
            "(b)": "analytic exact point-mass v_GP plus uniform +x wind",
            "(c)": "ORB-10935 positive-x Hamilton-Jacobi branch at 81^3; its 0.999*q caustic limiter is retained and masked",
            "(d)": "ORB-10937 equations regenerated at the original finest 41^3 rung and T=60 for every wind; the frozen velocity is used even though density was still relaxing",
        },
        "field_interpolation": {
            "method": "z=0 slice of the odd-grid 3-D field, bilinear velocity and analytic derivative of the same cell polynomial",
            "caustic_cells": "a ray evaluation is flagged when its four interpolation nodes touch the inferred ORB-10935 0.999*q limiter mask; clipped values are interpolated as returned, with no gap fill or extra smoothing",
            "convergence": "61^3->81^3 for field (c), 33^3->41^3 for field (d), measured on b=3 for every wind; ray maximum-step convergence is retained per orientation",
        },
        "field_grid_convergence": convergence,
    }
    reproducibility = {
        "seed": SEED,
        "random_numbers_used": False,
        "byte_identical_in_memory_rerun_verified": True,
        "byte_identical_scope": "complete results and dated run-record JSON encodings from two fresh field reconstructions and ray sweeps",
        "command": "PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-10939 uv run lab/sims/moving-core-acoustic-rays/main.py --check-determinism",
    }
    limitations = [
        "This is a model-side scale estimate, not a falsifier or an observational bound; no PPN coefficient is inferred from raw U/c.",
        "Field (c) is heavily clipped at low wind because ORB-10935 did not find a smooth single-valued branch there; mask hashes, fractions, and ray exposure are retained rather than hidden.",
        "Field (d) freezes ORB-10937 at T=60: its velocity sector met the task's residual admission, while density relaxation and attractor uniqueness remained unsettled.",
        "The field-grid shifts are apparatus errors, not statistical uncertainties, and the two predecessor ladders have different finest spacings.",
        "The boosted-GR delay uses an explicitly stated invariant-phase/Doppler convention; a full PPN reduction and observational walls remain separate theory obligations.",
    ]
    verdict = {
        "all_five_predeclared_gates_executed": True,
        "G1": gates["G1_static_comparator"]["verdict"],
        "G2": gates["G2_deflection_modulation"]["verdict"],
        "G3": gates["G3_boosted_schwarzschild"]["verdict"],
        "G4": gates["G4_shapiro_asymmetry"]["verdict"],
        "G5": gates["G5_cross_field_spread"]["verdict"],
        "scope": "scale_estimate_not_falsifier",
        "theory_reconciliation": "deferred to kepler; principia intentionally untouched",
    }
    results = {
        "schema_version": 1,
        "task": TASK_ID,
        "run_record": f"runs/{RUN_RECORD}",
        "reproducibility": reproducibility,
        "apparatus": apparatus,
        "field_sources": fields["source_diagnostics"],
        "static_baselines": baselines,
        "field_results": field_results,
        "gates": gates,
        "verdict": verdict,
        "limitations": limitations,
    }
    record = {
        **results,
        "run_id": RUN_ID,
        "run_date": RUN_DATE,
        "run_record": RUN_RECORD,
    }
    validate(results)
    return results, record


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
        (root / "assets/results.json").write_bytes(encoded(results))
        (root / f"runs/{RUN_RECORD}").write_bytes(encoded(record))
    print(
        f"G1={results['verdict']['G1']} G2={results['verdict']['G2']} "
        f"G3={results['verdict']['G3']} G4={results['verdict']['G4']} "
        f"G5={results['verdict']['G5']} "
        f"sha256={hashlib.sha256(encoded(record)).hexdigest()}"
    )


if __name__ == "__main__":
    main()
