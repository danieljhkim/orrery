"""Measure the common-mode budget carried by a dynamical relative-mode packet.

This deterministic 2-D finite-volume experiment evolves two separately
conserved substance densities and their two-component fluxes.  The Hamiltonian
density is

    H = (|j1|^2 + |j2|^2)/2
        + c_plus^2 n_plus^2/4 + c_minus^2 n_minus^2/4
        + lambda n_plus n_minus^2/4,

where n_plus = n1+n2-2*n0 and n_minus = n1-n2.  Continuity is updated as a
periodic discrete divergence, while the fluxes respond to the gradient of the
Hamiltonian chemical potentials.  The relative stiffness and its symmetric
coupling to the common mode are declared model hypotheses, not emergent facts.

The apparatus directly initializes a localized right-moving relative packet,
runs a four-amplitude energy ladder, and records conservation, convergence,
packet coherence, global budgets, co-moving common content, initialization-
region refill, and the far-monopole null.

Run with:
    uv run lab/sims/two-substance-dynamical-packet/main.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DENSITY = 1.0
DOMAIN_LENGTH = 96.0
BASE_SIZE = 128
BASE_DT = 0.06
DURATION = 30.0
SAMPLE_INTERVAL = 0.5
PACKET_X0 = -24.0
PACKET_SIGMA_X = 6.0
PACKET_SIGMA_Y = 12.0
COMMON_SPEED = 0.65
RELATIVE_SPEED = 1.0
MODE_COUPLING = 0.80
AMPLITUDES = (0.02, 0.04, 0.06, 0.08)
PROFILE_AMPLITUDE = AMPLITUDES[-1]
CONVERGENCE_AMPLITUDE = 0.01
CONVERGENCE_DURATION = 24.0


@dataclass
class State:
    n1: np.ndarray
    n2: np.ndarray
    j1x: np.ndarray
    j1y: np.ndarray
    j2x: np.ndarray
    j2y: np.ndarray


def coordinates(size: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(size, dtype=float) - size // 2) * dx
    return np.meshgrid(axis, axis, indexing="ij")


def derivative(values: np.ndarray, axis: int, dx: float) -> np.ndarray:
    """Periodic centered derivative; its lattice sum is identically telescoping."""
    return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (
        2.0 * dx
    )


def divergence(x_values: np.ndarray, y_values: np.ndarray, dx: float) -> np.ndarray:
    return derivative(x_values, 0, dx) + derivative(y_values, 1, dx)


def packet_profile(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Localized central excess with side lobes and exactly zero discrete sum."""
    x = xx - PACKET_X0
    gaussian = np.exp(
        -0.5 * (x / PACKET_SIGMA_X) ** 2
        - 0.5 * (yy / PACKET_SIGMA_Y) ** 2
    )
    profile = (1.0 - (x / PACKET_SIGMA_X) ** 2) * gaussian
    profile -= np.mean(profile)
    return profile


def initialize(size: int, dx: float, amplitude: float) -> State:
    xx, yy = coordinates(size, dx)
    relative = amplitude * packet_profile(xx, yy)
    # n_plus and J_plus are exactly zero at initialization.  J_minus=c*n_minus
    # is the right-moving relation for the linear 1-D relative wave equation.
    n1 = BASE_DENSITY + 0.5 * relative
    n2 = BASE_DENSITY - 0.5 * relative
    j_minus_x = RELATIVE_SPEED * relative
    zeros = np.zeros_like(relative)
    return State(
        n1=n1,
        n2=n2,
        j1x=0.5 * j_minus_x,
        j1y=zeros.copy(),
        j2x=-0.5 * j_minus_x,
        j2y=zeros.copy(),
    )


def modes(state: State) -> tuple[np.ndarray, np.ndarray]:
    common = state.n1 + state.n2 - 2.0 * BASE_DENSITY
    relative = state.n1 - state.n2
    return common, relative


def chemical_potentials(state: State) -> tuple[np.ndarray, np.ndarray]:
    common, relative = modes(state)
    common_linear = 0.5 * COMMON_SPEED**2 * common
    relative_linear = 0.5 * RELATIVE_SPEED**2 * relative
    common_from_relative = 0.25 * MODE_COUPLING * relative**2
    relative_from_common = 0.5 * MODE_COUPLING * common * relative
    mu1 = common_linear + relative_linear + common_from_relative + relative_from_common
    mu2 = common_linear - relative_linear + common_from_relative - relative_from_common
    return mu1, mu2


def kick_flux(
    state: State, mu1: np.ndarray, mu2: np.ndarray, amount: float, dx: float
) -> None:
    state.j1x -= amount * derivative(mu1, 0, dx)
    state.j1y -= amount * derivative(mu1, 1, dx)
    state.j2x -= amount * derivative(mu2, 0, dx)
    state.j2y -= amount * derivative(mu2, 1, dx)


def step(state: State, dt: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """One symmetric kick-drift-kick step; return fluxes used by continuity."""
    mu1, mu2 = chemical_potentials(state)
    kick_flux(state, mu1, mu2, 0.5 * dt, dx)
    continuity_j1 = (state.j1x.copy(), state.j1y.copy())
    continuity_j2 = (state.j2x.copy(), state.j2y.copy())
    div1 = divergence(*continuity_j1, dx)
    div2 = divergence(*continuity_j2, dx)
    state.n1 -= dt * div1
    state.n2 -= dt * div2
    mu1, mu2 = chemical_potentials(state)
    kick_flux(state, mu1, mu2, 0.5 * dt, dx)
    return div1, div2


def energy_density(state: State) -> np.ndarray:
    common, relative = modes(state)
    kinetic = 0.5 * (
        state.j1x**2
        + state.j1y**2
        + state.j2x**2
        + state.j2y**2
    )
    potential = (
        0.25 * COMMON_SPEED**2 * common**2
        + 0.25 * RELATIVE_SPEED**2 * relative**2
        + 0.25 * MODE_COUPLING * common * relative**2
    )
    return kinetic + potential


def relative_energy_density(state: State) -> np.ndarray:
    _, relative = modes(state)
    jx = state.j1x - state.j2x
    jy = state.j1y - state.j2y
    return 0.25 * (jx**2 + jy**2 + RELATIVE_SPEED**2 * relative**2)


def centroid_x(state: State, xx: np.ndarray, dx: float) -> float:
    weight = relative_energy_density(state)
    return float(np.sum(xx * weight) * dx**2 / (np.sum(weight) * dx**2))


def common_content(
    state: State, xx: np.ndarray, yy: np.ndarray, center_x: float, dx: float
) -> dict[str, float]:
    common, _ = modes(state)
    mask = (
        (np.abs(xx - center_x) <= 3.0 * PACKET_SIGMA_X)
        & (np.abs(yy) <= 3.0 * PACKET_SIGMA_Y)
    )
    selected = common[mask]
    return {
        "absolute_integral": float(np.sum(np.abs(selected)) * dx**2),
        "positive_integral": float(np.sum(np.clip(selected, 0.0, None)) * dx**2),
        "negative_integral": float(np.sum(np.clip(selected, None, 0.0)) * dx**2),
        "maximum": float(np.max(selected)),
        "minimum": float(np.min(selected)),
    }


def coherence_metrics(
    initial_relative: np.ndarray,
    state: State,
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    dx: float,
) -> dict[str, float]:
    _, relative = modes(state)
    energy = relative_energy_density(state)
    window = (
        (np.abs(xx - center_x) <= 3.0 * PACKET_SIGMA_X)
        & (np.abs(yy) <= 3.0 * PACKET_SIGMA_Y)
    )
    energy_fraction = float(np.sum(energy[window]) / np.sum(energy))

    displacement_cells = int(round((center_x - PACKET_X0) / dx))
    shifted = np.roll(initial_relative, displacement_cells, axis=0)
    numerator = float(np.sum(relative[window] * shifted[window]))
    denominator = float(
        np.sqrt(np.sum(relative[window] ** 2) * np.sum(shifted[window] ** 2))
    )
    correlation = numerator / max(denominator, np.finfo(float).tiny)
    return {
        "relative_energy_fraction_in_comoving_window": energy_fraction,
        "shape_correlation_with_translated_initial_packet": correlation,
    }


def profile_record(
    state: State, xx: np.ndarray, yy: np.ndarray, center_x: float, dx: float
) -> dict[str, list[float]]:
    common, relative = modes(state)
    y_mask = np.abs(yy[0, :]) <= 3.0 * PACKET_SIGMA_Y
    offsets = xx[:, 0] - center_x
    order = np.argsort(offsets)
    common_line = np.sum(common[:, y_mask], axis=1) * dx
    relative_line = np.sum(relative[:, y_mask], axis=1) * dx
    return {
        "x_offset_cells": [float(value) for value in offsets[order]],
        "common_mode_y_integral": [float(value) for value in common_line[order]],
        "relative_mode_y_integral": [float(value) for value in relative_line[order]],
    }


def fit_speed(times: list[float], centroids: list[float]) -> float:
    values_t = np.asarray(times)
    values_x = np.asarray(centroids)
    fit = (values_t >= 0.2 * values_t[-1]) & (values_t <= 0.8 * values_t[-1])
    return float(np.polyfit(values_t[fit], values_x[fit], 1)[0])


def run_packet(
    amplitude: float,
    *,
    size: int = BASE_SIZE,
    dx: float = DOMAIN_LENGTH / BASE_SIZE,
    dt: float = BASE_DT,
    duration: float = DURATION,
    keep_profile: bool = False,
) -> tuple[dict, dict[str, np.ndarray] | None]:
    state = initialize(size, dx, amplitude)
    xx, yy = coordinates(size, dx)
    initial_relative = modes(state)[1].copy()
    initial_n1_total = float(np.sum(state.n1) * dx**2)
    initial_n2_total = float(np.sum(state.n2) * dx**2)
    initial_energy = float(np.sum(energy_density(state)) * dx**2)
    emission_region = (
        (np.abs(xx - PACKET_X0) <= PACKET_SIGMA_X)
        & (np.abs(yy) <= 2.0 * PACKET_SIGMA_Y)
    )
    initial_region_n1 = float(
        np.sum((state.n1[emission_region] - BASE_DENSITY)) * dx**2
    )
    initial_region_n2 = float(
        np.sum((state.n2[emission_region] - BASE_DENSITY)) * dx**2
    )
    initial_region_energy = float(np.sum(energy_density(state)[emission_region]) * dx**2)

    sample_every = max(1, int(round(SAMPLE_INTERVAL / dt)))
    steps = int(round(duration / dt))
    times: list[float] = []
    centroids: list[float] = []
    n1_errors: list[float] = []
    n2_errors: list[float] = []
    common_integrals: list[float] = []
    relative_integrals: list[float] = []
    energies: list[float] = []
    region_n1: list[float] = []
    region_n2: list[float] = []
    region_energy: list[float] = []
    inflow_n1: list[float] = []
    inflow_n2: list[float] = []
    cumulative_inflow_n1 = 0.0
    cumulative_inflow_n2 = 0.0

    def sample(time: float) -> None:
        common, relative = modes(state)
        times.append(float(time))
        centroids.append(centroid_x(state, xx, dx))
        n1_errors.append(float(np.sum(state.n1) * dx**2 - initial_n1_total))
        n2_errors.append(float(np.sum(state.n2) * dx**2 - initial_n2_total))
        common_integrals.append(float(np.sum(common) * dx**2))
        relative_integrals.append(float(np.sum(relative) * dx**2))
        energies.append(float(np.sum(energy_density(state)) * dx**2))
        region_n1.append(
            float(np.sum((state.n1[emission_region] - BASE_DENSITY)) * dx**2)
        )
        region_n2.append(
            float(np.sum((state.n2[emission_region] - BASE_DENSITY)) * dx**2)
        )
        region_energy.append(float(np.sum(energy_density(state)[emission_region]) * dx**2))
        inflow_n1.append(cumulative_inflow_n1)
        inflow_n2.append(cumulative_inflow_n2)

    sample(0.0)
    for index in range(1, steps + 1):
        div1, div2 = step(state, dt, dx)
        cumulative_inflow_n1 += float(-dt * np.sum(div1[emission_region]) * dx**2)
        cumulative_inflow_n2 += float(-dt * np.sum(div2[emission_region]) * dx**2)
        if index % sample_every == 0 or index == steps:
            sample(index * dt)

    center = centroids[-1]
    content = common_content(state, xx, yy, center, dx)
    coherence = coherence_metrics(initial_relative, state, xx, yy, center, dx)
    conservation_floor = float(
        32.0 * np.finfo(float).eps * size**2 * BASE_DENSITY * dx**2
    )
    common_signal_floor = float(
        32.0
        * np.finfo(float).eps
        * size**2
        * max(float(np.max(np.abs(modes(state)[0]))), np.finfo(float).eps)
        * dx**2
    )
    final_region_n1 = region_n1[-1]
    final_region_n2 = region_n2[-1]
    final_region_energy = region_energy[-1]
    result = {
        "amplitude": amplitude,
        "initial_packet_energy": initial_energy,
        "measured_speed": fit_speed(times, centroids),
        "final_centroid_x": center,
        "coherence": coherence,
        "common_content": content,
        "conservation": {
            "instrument_floor": conservation_floor,
            "maximum_absolute_n1_total_drift": float(np.max(np.abs(n1_errors))),
            "maximum_absolute_n2_total_drift": float(np.max(np.abs(n2_errors))),
            "maximum_absolute_common_integral": float(
                np.max(np.abs(common_integrals))
            ),
            "maximum_absolute_relative_integral": float(
                np.max(np.abs(relative_integrals))
            ),
            "common_signal_roundoff_floor": common_signal_floor,
        },
        "energy": {
            "initial": initial_energy,
            "final": energies[-1],
            "maximum_relative_drift": float(
                np.max(np.abs(np.asarray(energies) - initial_energy)) / initial_energy
            ),
        },
        "initialization_region": {
            "definition": (
                "fixed rectangle |x-x0| <= sigma_x and |y| <= 2 sigma_y; "
                "inflow is the time-integrated conservative face-flux divergence"
            ),
            "initial_n1_excess": initial_region_n1,
            "final_n1_excess": final_region_n1,
            "integrated_n1_inflow": inflow_n1[-1],
            "n1_budget_residual": final_region_n1
            - initial_region_n1
            - inflow_n1[-1],
            "initial_n2_excess": initial_region_n2,
            "final_n2_excess": final_region_n2,
            "integrated_n2_inflow": inflow_n2[-1],
            "n2_budget_residual": final_region_n2
            - initial_region_n2
            - inflow_n2[-1],
            "initial_energy": initial_region_energy,
            "final_energy": final_region_energy,
            "fraction_of_initial_energy_departed": 1.0
            - final_region_energy / initial_region_energy,
            "final_common_mode_integral": float(
                np.sum(modes(state)[0][emission_region]) * dx**2
            ),
        },
        "time_series": {
            "time": times,
            "n1_total_drift": n1_errors,
            "n2_total_drift": n2_errors,
            "common_mode_integral": common_integrals,
            "relative_mode_integral": relative_integrals,
            "total_energy": energies,
            "packet_centroid_x": centroids,
            "initialization_region_n1_excess": region_n1,
            "initialization_region_n2_excess": region_n2,
            "initialization_region_energy": region_energy,
            "integrated_n1_inflow": inflow_n1,
            "integrated_n2_inflow": inflow_n2,
        },
    }
    profiles = None
    if keep_profile:
        common, relative = modes(state)
        profiles = {
            "x": xx,
            "y": yy,
            "common": common,
            "relative": relative,
            "j1x": state.j1x,
            "j1y": state.j1y,
            "j2x": state.j2x,
            "j2y": state.j2y,
        }
        result["co_moving_profile"] = profile_record(state, xx, yy, center, dx)
    return result, profiles


def convergence_study() -> dict:
    grid_rows = []
    for size in (128, 192, 256):
        dx = DOMAIN_LENGTH / size
        row, _ = run_packet(
            CONVERGENCE_AMPLITUDE,
            size=size,
            dx=dx,
            dt=0.06,
            duration=CONVERGENCE_DURATION,
        )
        grid_rows.append(
            {"size": size, "dx": dx, "dt": 0.06, "speed": row["measured_speed"]}
        )

    time_rows = []
    for dt in (0.09, 0.06, 0.03):
        row, _ = run_packet(
            CONVERGENCE_AMPLITUDE,
            size=128,
            dx=0.75,
            dt=dt,
            duration=CONVERGENCE_DURATION,
        )
        time_rows.append(
            {"size": 128, "dx": 0.75, "dt": dt, "speed": row["measured_speed"]}
        )

    grid_relative_change = abs(grid_rows[-1]["speed"] - grid_rows[-2]["speed"]) / abs(
        grid_rows[-1]["speed"]
    )
    time_relative_change = abs(time_rows[-1]["speed"] - time_rows[-2]["speed"]) / abs(
        time_rows[-1]["speed"]
    )
    return {
        "small_amplitude": CONVERGENCE_AMPLITUDE,
        "declared_linear_relative_speed": RELATIVE_SPEED,
        "grid_refinement": grid_rows,
        "time_step_refinement": time_rows,
        "finest_grid_relative_change": float(grid_relative_change),
        "finest_time_step_relative_change": float(time_relative_change),
        "finest_measured_speed": time_rows[-1]["speed"],
    }


def analyze() -> tuple[dict, dict[str, np.ndarray]]:
    ladder = []
    representative_profiles = None
    for amplitude in AMPLITUDES:
        row, profiles = run_packet(amplitude, keep_profile=amplitude == PROFILE_AMPLITUDE)
        ladder.append(row)
        if profiles is not None:
            representative_profiles = profiles
    assert representative_profiles is not None

    energies = np.asarray([row["initial_packet_energy"] for row in ladder])
    contents = np.asarray([row["common_content"]["absolute_integral"] for row in ladder])
    exponent, log_prefactor = np.polyfit(np.log(energies), np.log(contents), 1)
    prefactor = float(np.exp(log_prefactor))
    representative = ladder[-1]
    all_coherent = all(
        row["coherence"]["relative_energy_fraction_in_comoving_window"] > 0.70
        and row["coherence"]["shape_correlation_with_translated_initial_packet"] > 0.55
        for row in ladder
    )
    conservation_floor = representative["conservation"]["instrument_floor"]
    maximum_common_monopole = max(
        row["conservation"]["maximum_absolute_common_integral"] for row in ladder
    )
    common_hook = (
        all_coherent
        and 0.8 <= exponent <= 1.2
        and contents[0] > 100.0 * conservation_floor
    )

    result = {
        "schema_version": 1,
        "task": "ORB-10940",
        "run_date": "2026-08-21",
        "question": (
            "Does a propagating relative-mode packet dynamically carry co-moving "
            "common-mode structure, and does its magnitude scale with packet energy?"
        ),
        "apparatus": {
            "dimensions": 2,
            "three_dimensional_scope": (
                "not tested; this 2-D fixture establishes only the model's radiation-sector "
                "mode structure and cannot determine 3-D propagation or gravity laws"
            ),
            "lattice": [BASE_SIZE, BASE_SIZE],
            "domain_length": DOMAIN_LENGTH,
            "boundary": "periodic square",
            "time_step": BASE_DT,
            "duration": DURATION,
            "base_density_per_substance": BASE_DENSITY,
            "common_mode_speed": COMMON_SPEED,
            "relative_mode_speed": RELATIVE_SPEED,
            "relative_common_coupling": MODE_COUPLING,
            "seed": None,
            "determinism": "no random numbers; fixed operation order and sorted JSON keys",
        },
        "predeclared_dynamics": {
            "hamiltonian_density": (
                "(|j1|^2+|j2|^2)/2 + c_plus^2*n_plus^2/4 + "
                "c_minus^2*n_minus^2/4 + lambda*n_plus*n_minus^2/4"
            ),
            "continuity": "partial_t n_s = -div(j_s), independently for s=1,2",
            "flux_update": "partial_t j_s = -grad(partial H / partial n_s)",
            "discrete_update": (
                "periodic centered differences with symmetric kick-drift-kick; the "
                "drift is a telescoping discrete divergence and separately conserves n1,n2"
            ),
            "model_hypothesis": (
                "in the irrotational sector j_s=grad(theta_s), so J_minus is the "
                "gradient of the relative phase theta1-theta2; treating that relative "
                "phase/current channel and its c_minus stiffness as physical inherits "
                "ORB-10163's relative-phase hypothesis. The symmetric "
                "lambda*n_plus*n_minus^2 coupling is additionally declared here and is "
                "not derived from a microscopic vacuum model"
            ),
            "packet_initialization": (
                "direct mode (i): n1=n0+n_minus/2, n2=n0-n_minus/2, n_plus=0, "
                "J_minus_x=c_minus*n_minus, J_plus=0; a localized derivative-Gaussian "
                "has zero discrete integral"
            ),
            "co_moving_region": (
                "|x-x_centroid| <= 3 sigma_x and |y| <= 3 sigma_y at the final time"
            ),
        },
        "G1_apparatus_calibration": {
            "convergence": convergence_study(),
            "packet_verdict": "propagates" if all_coherent else "disperses",
            "packet_detail": (
                "a translating packet remains resolved with bounded transverse spreading"
                if all_coherent
                else "no ladder member retained the predeclared energy fraction and shape correlation"
            ),
            "per_substance_conservation_floor": conservation_floor,
            "maximum_n1_drift": max(
                row["conservation"]["maximum_absolute_n1_total_drift"] for row in ladder
            ),
            "maximum_n2_drift": max(
                row["conservation"]["maximum_absolute_n2_total_drift"] for row in ladder
            ),
        },
        "G2_global_budget_integrals": {
            "interpretation": (
                "calibration only: both global mode integrals must remain zero because "
                "each substance is independently conserved"
            ),
            "instrument_floor": conservation_floor,
            "maximum_absolute_common_integral": maximum_common_monopole,
            "maximum_absolute_relative_integral": max(
                row["conservation"]["maximum_absolute_relative_integral"] for row in ladder
            ),
            "representative_time_series": representative["time_series"],
        },
        "G3_primary_common_mode_result": {
            "energy_ladder": [
                {
                    "amplitude": row["amplitude"],
                    "packet_energy": row["initial_packet_energy"],
                    "co_moving_absolute_common_integral": row["common_content"][
                        "absolute_integral"
                    ],
                    "common_integral_per_energy": row["common_content"][
                        "absolute_integral"
                    ]
                    / row["initial_packet_energy"],
                    "profile_maximum": row["common_content"]["maximum"],
                    "profile_minimum": row["common_content"]["minimum"],
                }
                for row in ladder
            ],
            "power_law": {
                "definition": "co-moving integral |n_plus| = prefactor * E^exponent",
                "exponent": float(exponent),
                "prefactor": prefactor,
            },
            "representative_profile": representative["co_moving_profile"],
            "representative_sign_structure": {
                "positive_pocket_integral": representative["common_content"][
                    "positive_integral"
                ],
                "negative_halo_integral": representative["common_content"][
                    "negative_integral"
                ],
                "profile_maximum": representative["common_content"]["maximum"],
                "profile_minimum": representative["common_content"]["minimum"],
            },
            "verdict": (
                "pass: a co-moving common-mode pocket plus compensating structure is "
                "resolved and scales proportionally to packet energy; this declared "
                "dynamics has an E/c^2-type source hook (normalization not established)"
                if common_hook
                else "kill: common-mode content is absent or does not scale with packet energy"
            ),
            "pass": bool(common_hook),
        },
        "G4_initialization_region_relaxation": representative[
            "initialization_region"
        ],
        "G5_far_monopole_null": {
            "definition": (
                "global periodic-lattice integral of n_plus, the monopole seen outside "
                "the complete packet-plus-compensation system"
            ),
            "instrument_floor": conservation_floor,
            "maximum_absolute_monopole": maximum_common_monopole,
            "verdict": (
                "null at instrument floor"
                if maximum_common_monopole <= conservation_floor
                else "apparatus error: monopole exceeds instrument floor"
            ),
        },
        "ladder_runs": ladder,
        "conclusions": {
            "primary_deliverable": (
                f"The co-moving common-mode magnitude scales as E^{exponent:.6f} "
                f"with prefactor {prefactor:.6g} in lattice units."
            ),
            "scope": (
                "The proportional hook is a consequence of the predeclared nonlinear "
                "lambda coupling. It is not a derivation of gravity, an observational "
                "fit, or a result about the already-refuted conserved far monopole."
            ),
        },
    }
    validate_result(result)
    return result, representative_profiles


def validate_result(result: dict) -> None:
    g1 = result["G1_apparatus_calibration"]
    convergence = g1["convergence"]
    g2 = result["G2_global_budget_integrals"]
    g3 = result["G3_primary_common_mode_result"]
    g4 = result["G4_initialization_region_relaxation"]
    g5 = result["G5_far_monopole_null"]
    budget_floor = g1["per_substance_conservation_floor"]
    checks = {
        "n1 conserved at instrument floor": g1["maximum_n1_drift"] <= budget_floor,
        "n2 conserved at instrument floor": g1["maximum_n2_drift"] <= budget_floor,
        "grid-converged small-amplitude speed": convergence[
            "finest_grid_relative_change"
        ]
        < 0.01,
        "time-step-converged small-amplitude speed": convergence[
            "finest_time_step_relative_change"
        ]
        < 0.01,
        "packet propagates": g1["packet_verdict"] == "propagates",
        "global common budget at floor": g2["maximum_absolute_common_integral"]
        <= budget_floor,
        "global relative budget at floor": g2["maximum_absolute_relative_integral"]
        <= budget_floor,
        "common content scales with energy": g3["pass"],
        "initialization region substantially relaxes": g4[
            "fraction_of_initial_energy_departed"
        ]
        > 0.80,
        "n1 region flux closes": abs(g4["n1_budget_residual"]) <= budget_floor,
        "n2 region flux closes": abs(g4["n2_budget_residual"]) <= budget_floor,
        "far monopole null": g5["maximum_absolute_monopole"] <= budget_floor,
    }
    result["validation"] = {"checks": checks, "all_passed": all(checks.values())}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("validation failed: " + "; ".join(failed))


def make_plot(result: dict, profiles: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    common = profiles["common"]
    relative = profiles["relative"]
    extent = [
        float(profiles["x"].min()),
        float(profiles["x"].max()),
        float(profiles["y"].min()),
        float(profiles["y"].max()),
    ]
    common_limit = float(np.max(np.abs(common)))
    relative_limit = float(np.max(np.abs(relative)))
    common_image = axes[0, 0].imshow(
        common.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-common_limit,
        vmax=common_limit,
    )
    axes[0, 0].set(title="Final common mode n+", xlabel="x", ylabel="y")
    fig.colorbar(common_image, ax=axes[0, 0])
    relative_image = axes[0, 1].imshow(
        relative.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-relative_limit,
        vmax=relative_limit,
    )
    axes[0, 1].set(title="Final relative mode n-", xlabel="x", ylabel="y")
    fig.colorbar(relative_image, ax=axes[0, 1])

    ladder = result["G3_primary_common_mode_result"]["energy_ladder"]
    energy = np.asarray([row["packet_energy"] for row in ladder])
    content = np.asarray([row["co_moving_absolute_common_integral"] for row in ladder])
    exponent = result["G3_primary_common_mode_result"]["power_law"]["exponent"]
    prefactor = result["G3_primary_common_mode_result"]["power_law"]["prefactor"]
    axes[1, 0].loglog(energy, content, "o", color="#ffcc66", label="ladder")
    axes[1, 0].loglog(
        energy,
        prefactor * energy**exponent,
        "--",
        color="#75c9ff",
        label=f"fit exponent {exponent:.4f}",
    )
    axes[1, 0].set(
        title="Primary result: co-moving common content",
        xlabel="initial packet energy E",
        ylabel="integral |n+| in packet window",
    )
    axes[1, 0].legend()

    series = result["G2_global_budget_integrals"]["representative_time_series"]
    time = np.asarray(series["time"])
    energy_series = np.asarray(series["initialization_region_energy"])
    initial_energy = energy_series[0]
    axes[1, 1].plot(
        time,
        energy_series / initial_energy,
        color="#78d69b",
        label="region energy / initial",
    )
    axes[1, 1].plot(
        time,
        np.asarray(series["packet_centroid_x"]),
        color="#caa6ff",
        label="packet centroid x",
    )
    axes[1, 1].set(
        title="Packet departure and source-region relaxation",
        xlabel="time",
        ylabel="normalized energy / x position",
    )
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("Two-substance dynamical packet budget")
    fig.tight_layout()
    fig.savefig(output, dpi=180, metadata={"Software": "orrery"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("assets")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result, profiles = analyze()
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    make_plot(result, profiles, args.output_dir / "packet-budget.png")
    g1 = result["G1_apparatus_calibration"]
    g3 = result["G3_primary_common_mode_result"]
    g5 = result["G5_far_monopole_null"]
    print(
        f"packet: {g1['packet_verdict']}; speed "
        f"{g1['convergence']['finest_measured_speed']:.8f}"
    )
    print(
        "co-moving common content: "
        f"E^{g3['power_law']['exponent']:.8f}; {g3['verdict']}"
    )
    print(
        f"far monopole: {g5['maximum_absolute_monopole']:.3e} "
        f"(floor {g5['instrument_floor']:.3e})"
    )


if __name__ == "__main__":
    main()
