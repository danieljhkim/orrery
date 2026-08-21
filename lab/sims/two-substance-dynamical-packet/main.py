"""Distinguish a packet-bound common mode from a coupling-driven trailing wake.

This deterministic 2-D finite-volume experiment evolves two separately
conserved substance densities and their fluxes under the original Hamiltonian

    H = (|j1|^2 + |j2|^2)/2
        + c_plus^2 n_plus^2/4 + c_minus^2 n_minus^2/4
        + lambda n_plus n_minus^2/4.

It runs lambda=0 and three nonzero couplings over a four-amplitude ladder,
tracks time-resolved signed and absolute mode profiles, and refines the primary
common-content exponent, prefactor, and profile lag in space and time.  The
apparatus also audits the cubic Hamiltonian and explicitly gates defect
emission on whether the existing model contains an emission operator.

Run with:
    uv run lab/sims/two-substance-dynamical-packet/main.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
COUPLINGS = (0.0, 0.2, 0.5, 0.8)
PRIMARY_COUPLING = COUPLINGS[-1]
AMPLITUDES = (0.02, 0.04, 0.06, 0.08)
PROFILE_AMPLITUDE = AMPLITUDES[-1]
CONVERGENCE_DURATION = DURATION
GRID_RUNGS = (128, 192, 256)
TIMESTEP_RUNGS = (0.09, 0.06, 0.03)

# These thresholds are declared before classification and are deliberately
# result-neutral: a failed gate is reported as a wake, mixed result, or
# nonconvergence rather than causing the executable itself to fail.
LATE_TIME_FRACTION = 0.40
ATTACHED_LAG_WIDTHS = 0.75
ATTACHED_PROFILE_CORRELATION = 0.50
ATTACHED_SAMPLE_FRACTION = 0.80
ATTACHED_SPEED_FRACTION = 0.10
RESOLVED_SIGNAL_FLOORS = 100.0
EXPONENT_CONVERGENCE_ABS = 0.02
PREFACTOR_CONVERGENCE_REL = 0.05
LAG_CONVERGENCE_ABS_WIDTHS = 0.10
PROFILE_SNAPSHOT_FRACTIONS = (0.0, 0.40, 0.70, 1.0)


class State:
    def __init__(
        self,
        n1: np.ndarray,
        n2: np.ndarray,
        j1x: np.ndarray,
        j1y: np.ndarray,
        j2x: np.ndarray,
        j2y: np.ndarray,
    ) -> None:
        self.n1 = n1
        self.n2 = n2
        self.j1x = j1x
        self.j1y = j1y
        self.j2x = j2x
        self.j2y = j2y


def coordinates(size: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(size, dtype=float) - size // 2) * dx
    return np.meshgrid(axis, axis, indexing="ij")


def derivative(values: np.ndarray, axis: int, dx: float) -> np.ndarray:
    """Periodic centered derivative; its lattice sum telescopes."""
    return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (
        2.0 * dx
    )


def divergence(x_values: np.ndarray, y_values: np.ndarray, dx: float) -> np.ndarray:
    return derivative(x_values, 0, dx) + derivative(y_values, 1, dx)


def packet_profile(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Localized central excess with side lobes and zero discrete sum."""
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
    # n_plus and J_plus start at zero.  J_minus=c_minus*n_minus is the
    # right-moving relation for the linear relative wave.
    n1 = BASE_DENSITY + 0.5 * relative
    n2 = BASE_DENSITY - 0.5 * relative
    j_minus_x = RELATIVE_SPEED * relative
    zeros = np.zeros_like(relative)
    return State(
        n1,
        n2,
        0.5 * j_minus_x,
        zeros.copy(),
        -0.5 * j_minus_x,
        zeros.copy(),
    )


def modes(state: State) -> tuple[np.ndarray, np.ndarray]:
    common = state.n1 + state.n2 - 2.0 * BASE_DENSITY
    relative = state.n1 - state.n2
    return common, relative


def chemical_potentials(state: State, coupling: float) -> tuple[np.ndarray, np.ndarray]:
    common, relative = modes(state)
    common_linear = 0.5 * COMMON_SPEED**2 * common
    relative_linear = 0.5 * RELATIVE_SPEED**2 * relative
    common_from_relative = 0.25 * coupling * relative**2
    relative_from_common = 0.5 * coupling * common * relative
    mu1 = common_linear + relative_linear + common_from_relative + relative_from_common
    mu2 = common_linear - relative_linear + common_from_relative - relative_from_common
    return mu1, mu2


def kick_flux(
    state: State,
    mu1: np.ndarray,
    mu2: np.ndarray,
    amount: float,
    dx: float,
) -> None:
    state.j1x -= amount * derivative(mu1, 0, dx)
    state.j1y -= amount * derivative(mu1, 1, dx)
    state.j2x -= amount * derivative(mu2, 0, dx)
    state.j2y -= amount * derivative(mu2, 1, dx)


def step(
    state: State, dt: float, dx: float, coupling: float
) -> tuple[np.ndarray, np.ndarray]:
    """One symmetric kick-drift-kick step; return continuity divergences."""
    mu1, mu2 = chemical_potentials(state, coupling)
    kick_flux(state, mu1, mu2, 0.5 * dt, dx)
    div1 = divergence(state.j1x, state.j1y, dx)
    div2 = divergence(state.j2x, state.j2y, dx)
    state.n1 -= dt * div1
    state.n2 -= dt * div2
    mu1, mu2 = chemical_potentials(state, coupling)
    kick_flux(state, mu1, mu2, 0.5 * dt, dx)
    return div1, div2


def energy_density(state: State, coupling: float) -> np.ndarray:
    common, relative = modes(state)
    kinetic = 0.5 * (
        state.j1x**2 + state.j1y**2 + state.j2x**2 + state.j2y**2
    )
    potential = (
        0.25 * COMMON_SPEED**2 * common**2
        + 0.25 * RELATIVE_SPEED**2 * relative**2
        + 0.25 * coupling * common * relative**2
    )
    return kinetic + potential


def relative_energy_density(state: State) -> np.ndarray:
    _, relative = modes(state)
    jx = state.j1x - state.j2x
    jy = state.j1y - state.j2y
    return 0.25 * (jx**2 + jy**2 + RELATIVE_SPEED**2 * relative**2)


def weighted_moments(
    line: np.ndarray, x_axis: np.ndarray, dx: float, signal_floor: float
) -> dict[str, float | None]:
    positive = np.clip(line, 0.0, None)
    negative_weight = -np.clip(line, None, 0.0)
    absolute = np.abs(line)

    def centroid(weight: np.ndarray) -> float | None:
        content = float(np.sum(weight) * dx)
        if content <= signal_floor:
            return None
        return float(np.sum(x_axis * weight) * dx / content)

    absolute_content = float(np.sum(absolute) * dx)
    absolute_centroid = centroid(absolute)
    width = None
    if absolute_centroid is not None:
        width = float(
            np.sqrt(
                np.sum((x_axis - absolute_centroid) ** 2 * absolute)
                * dx
                / absolute_content
            )
        )
    return {
        "signed_content": float(np.sum(line) * dx),
        "absolute_content": absolute_content,
        "positive_content": float(np.sum(positive) * dx),
        "negative_content": float(-np.sum(negative_weight) * dx),
        "absolute_centroid_x": absolute_centroid,
        "positive_centroid_x": centroid(positive),
        "negative_centroid_x": centroid(negative_weight),
        "absolute_width": width,
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denominator <= np.finfo(float).tiny:
        return None
    return float(np.sum(left * right) / denominator)


def line_profiles(
    state: State,
    yy: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    common, relative = modes(state)
    y_mask = np.abs(yy[0, :]) <= 3.0 * PACKET_SIGMA_Y
    return (
        np.sum(common[:, y_mask], axis=1) * dx,
        np.sum(relative[:, y_mask], axis=1) * dx,
    )


def profile_diagnostics(
    state: State,
    x_axis: np.ndarray,
    yy: np.ndarray,
    dx: float,
    initial_relative_line: np.ndarray,
    signal_floor: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    common_line, relative_line = line_profiles(state, yy, dx)
    common = weighted_moments(common_line, x_axis, dx, signal_floor)
    relative = weighted_moments(relative_line, x_axis, dx, signal_floor)
    relative_center = relative["absolute_centroid_x"]
    relative_width = relative["absolute_width"]
    common_center = common["absolute_centroid_x"]

    lag = None
    normalized_lag = None
    translated_correlation = None
    if relative_center is not None and relative_width is not None:
        displacement_cells = int(round((relative_center - PACKET_X0) / dx))
        shifted = np.roll(initial_relative_line, displacement_cells)
        translated_correlation = cosine_similarity(relative_line, shifted)
        if common_center is not None:
            lag = float(common_center - relative_center)
            normalized_lag = float(lag / max(relative_width, np.finfo(float).tiny))

    return (
        {
            "common": common,
            "relative": relative,
            "common_relative_absolute_correlation": cosine_similarity(
                np.abs(common_line), np.abs(relative_line)
            ),
            "common_relative_signed_correlation": cosine_similarity(
                common_line, relative_line
            ),
            "relative_translated_initial_correlation": translated_correlation,
            "common_minus_relative_centroid_lag": lag,
            "lag_normalized_by_relative_width": normalized_lag,
        },
        common_line,
        relative_line,
    )


def append_profile_sample(series: dict[str, Any], sample: dict[str, Any]) -> None:
    for mode_name in ("common", "relative"):
        for key, value in sample[mode_name].items():
            series[mode_name][key].append(value)
    for key in (
        "common_relative_absolute_correlation",
        "common_relative_signed_correlation",
        "relative_translated_initial_correlation",
        "common_minus_relative_centroid_lag",
        "lag_normalized_by_relative_width",
    ):
        series[key].append(sample[key])


def fit_speed(times: list[float], centroids: list[float | None], start: float) -> float | None:
    pairs = [
        (time, center)
        for time, center in zip(times, centroids, strict=True)
        if time >= start and center is not None
    ]
    if len(pairs) < 3:
        return None
    values_t = np.asarray([pair[0] for pair in pairs])
    values_x = np.asarray([pair[1] for pair in pairs])
    return float(np.polyfit(values_t, values_x, 1)[0])


def attachment_verdict(
    profile_series: dict[str, Any], duration: float, signal_floor: float
) -> dict[str, Any]:
    times = profile_series["time"]
    late = [index for index, time in enumerate(times) if time >= LATE_TIME_FRACTION * duration]
    common_content = profile_series["common"]["absolute_content"]
    normalized_lag = profile_series["lag_normalized_by_relative_width"]
    correlations = profile_series["common_relative_absolute_correlation"]
    resolved = [
        index
        for index in late
        if common_content[index] > RESOLVED_SIGNAL_FLOORS * signal_floor
        and normalized_lag[index] is not None
        and correlations[index] is not None
    ]
    resolved_fraction = len(resolved) / max(len(late), 1)
    relative_speed = fit_speed(
        times,
        profile_series["relative"]["absolute_centroid_x"],
        LATE_TIME_FRACTION * duration,
    )
    common_speed = fit_speed(
        times,
        profile_series["common"]["absolute_centroid_x"],
        LATE_TIME_FRACTION * duration,
    )

    if not resolved:
        verdict = "null_control" if max(common_content) <= RESOLVED_SIGNAL_FLOORS * signal_floor else "unresolved"
        attached_fraction = 0.0
        trailing_fraction = 0.0
        median_lag = None
        median_correlation = None
    else:
        attached = [
            index
            for index in resolved
            if abs(normalized_lag[index]) <= ATTACHED_LAG_WIDTHS
            and correlations[index] >= ATTACHED_PROFILE_CORRELATION
        ]
        trailing = [index for index in resolved if normalized_lag[index] < -ATTACHED_LAG_WIDTHS]
        attached_fraction = len(attached) / len(resolved)
        trailing_fraction = len(trailing) / len(resolved)
        median_lag = float(np.median([normalized_lag[index] for index in resolved]))
        median_correlation = float(np.median([correlations[index] for index in resolved]))
        speed_tolerance = ATTACHED_SPEED_FRACTION * max(
            abs(relative_speed) if relative_speed is not None else 0.0,
            np.finfo(float).tiny,
        )
        speeds_match = (
            common_speed is not None
            and relative_speed is not None
            and abs(common_speed - relative_speed) <= speed_tolerance
        )
        common_is_slower = (
            common_speed is not None
            and relative_speed is not None
            and common_speed < relative_speed - speed_tolerance
        )
        if (
            resolved_fraction >= ATTACHED_SAMPLE_FRACTION
            and attached_fraction >= ATTACHED_SAMPLE_FRACTION
            and speeds_match
        ):
            verdict = "packet_attached"
        elif (
            resolved_fraction >= ATTACHED_SAMPLE_FRACTION
            and median_lag < -ATTACHED_LAG_WIDTHS
            and median_correlation >= ATTACHED_PROFILE_CORRELATION
            and common_is_slower
        ):
            verdict = "trailing_wake"
        else:
            verdict = "mixed_or_detached"

    return {
        "predeclared_evaluation": (
            "For samples at t >= 0.40 duration, signal is resolved above 100 numerical "
            "floors. Packet-attached requires >=80% of resolved samples with |lag|/width "
            "<=0.75, common/relative absolute-profile correlation >=0.50, and fitted "
            "profile speeds within 10%. A trailing wake requires >=80% resolved late "
            "samples, median lag/width < -0.75, median correlation >=0.50, and a common "
            "profile speed more than 10% below the relative packet speed."
        ),
        "late_sample_count": len(late),
        "resolved_late_sample_count": len(resolved),
        "resolved_late_sample_fraction": resolved_fraction,
        "attached_resolved_sample_fraction": attached_fraction,
        "trailing_resolved_sample_fraction": trailing_fraction,
        "median_late_lag_normalized_by_packet_width": median_lag,
        "median_late_absolute_profile_correlation": median_correlation,
        "relative_profile_speed": relative_speed,
        "common_profile_speed": common_speed,
        "verdict": verdict,
    }


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
        "signed_integral": float(np.sum(selected) * dx**2),
        "positive_integral": float(np.sum(np.clip(selected, 0.0, None)) * dx**2),
        "negative_integral": float(np.sum(np.clip(selected, None, 0.0)) * dx**2),
        "maximum": float(np.max(selected)),
        "minimum": float(np.min(selected)),
    }


def snapshot_record(
    time: float,
    x_axis: np.ndarray,
    common_line: np.ndarray,
    relative_line: np.ndarray,
) -> dict[str, Any]:
    return {
        "time": time,
        "x": x_axis.tolist(),
        "common_signed": common_line.tolist(),
        "common_positive": np.clip(common_line, 0.0, None).tolist(),
        "common_negative": np.clip(common_line, None, 0.0).tolist(),
        "common_absolute": np.abs(common_line).tolist(),
        "relative_signed": relative_line.tolist(),
        "relative_positive": np.clip(relative_line, 0.0, None).tolist(),
        "relative_negative": np.clip(relative_line, None, 0.0).tolist(),
        "relative_absolute": np.abs(relative_line).tolist(),
    }


def empty_profile_series() -> dict[str, Any]:
    moment_keys = (
        "signed_content",
        "absolute_content",
        "positive_content",
        "negative_content",
        "absolute_centroid_x",
        "positive_centroid_x",
        "negative_centroid_x",
        "absolute_width",
    )
    return {
        "time": [],
        "common": {key: [] for key in moment_keys},
        "relative": {key: [] for key in moment_keys},
        "common_relative_absolute_correlation": [],
        "common_relative_signed_correlation": [],
        "relative_translated_initial_correlation": [],
        "common_minus_relative_centroid_lag": [],
        "lag_normalized_by_relative_width": [],
    }


def run_packet(
    amplitude: float,
    coupling: float,
    *,
    size: int = BASE_SIZE,
    dx: float | None = None,
    dt: float = BASE_DT,
    duration: float = DURATION,
    keep_profiles: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    if dx is None:
        dx = DOMAIN_LENGTH / size
    state = initialize(size, dx, amplitude)
    xx, yy = coordinates(size, dx)
    x_axis = xx[:, 0]
    _, initial_relative_line = line_profiles(state, yy, dx)
    initial_n1_total = float(np.sum(state.n1) * dx**2)
    initial_n2_total = float(np.sum(state.n2) * dx**2)
    initial_energy = float(np.sum(energy_density(state, coupling)) * dx**2)
    emission_region = (
        (np.abs(xx - PACKET_X0) <= PACKET_SIGMA_X)
        & (np.abs(yy) <= 2.0 * PACKET_SIGMA_Y)
    )
    initial_region_n1 = float(np.sum((state.n1[emission_region] - BASE_DENSITY)) * dx**2)
    initial_region_n2 = float(np.sum((state.n2[emission_region] - BASE_DENSITY)) * dx**2)
    initial_region_energy = float(
        np.sum(energy_density(state, coupling)[emission_region]) * dx**2
    )
    conservation_floor = float(
        32.0 * np.finfo(float).eps * size**2 * BASE_DENSITY * dx**2
    )
    profile_signal_floor = float(
        128.0 * np.finfo(float).eps * size**2 * BASE_DENSITY * dx**2
    )

    sample_every = max(1, int(round(SAMPLE_INTERVAL / dt)))
    steps = int(round(duration / dt))
    regular_samples = set(range(0, steps + 1, sample_every)) | {steps}
    snapshot_steps = {
        steps
        if fraction == 1.0
        else min(steps, int(round(fraction * steps / sample_every)) * sample_every)
        for fraction in PROFILE_SNAPSHOT_FRACTIONS
    }
    # Raw snapshots are taken on the regular diagnostic cadence so enabling
    # field capture cannot change the samples used by attachment statistics.
    sample_steps = regular_samples

    times: list[float] = []
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
    profile_series = empty_profile_series()
    snapshots: list[dict[str, Any]] = []
    cumulative_inflow_n1 = 0.0
    cumulative_inflow_n2 = 0.0
    sampled_min_n1 = float("inf")
    sampled_min_n2 = float("inf")
    sampled_min_energy_density = float("inf")
    sampled_max_energy_density = -float("inf")
    sampled_max_abs_common = 0.0
    sampled_max_abs_relative = 0.0

    def sample(index: int) -> None:
        nonlocal sampled_min_n1, sampled_min_n2, sampled_min_energy_density
        nonlocal sampled_max_energy_density, sampled_max_abs_common
        nonlocal sampled_max_abs_relative
        time = float(index * dt)
        common, relative = modes(state)
        density = energy_density(state, coupling)
        diagnostics, common_line, relative_line = profile_diagnostics(
            state,
            x_axis,
            yy,
            dx,
            initial_relative_line,
            profile_signal_floor,
        )
        times.append(time)
        profile_series["time"].append(time)
        append_profile_sample(profile_series, diagnostics)
        n1_errors.append(float(np.sum(state.n1) * dx**2 - initial_n1_total))
        n2_errors.append(float(np.sum(state.n2) * dx**2 - initial_n2_total))
        common_integrals.append(float(np.sum(common) * dx**2))
        relative_integrals.append(float(np.sum(relative) * dx**2))
        energies.append(float(np.sum(density) * dx**2))
        region_n1.append(float(np.sum((state.n1[emission_region] - BASE_DENSITY)) * dx**2))
        region_n2.append(float(np.sum((state.n2[emission_region] - BASE_DENSITY)) * dx**2))
        region_energy.append(float(np.sum(density[emission_region]) * dx**2))
        inflow_n1.append(cumulative_inflow_n1)
        inflow_n2.append(cumulative_inflow_n2)
        sampled_min_n1 = min(sampled_min_n1, float(np.min(state.n1)))
        sampled_min_n2 = min(sampled_min_n2, float(np.min(state.n2)))
        sampled_min_energy_density = min(sampled_min_energy_density, float(np.min(density)))
        sampled_max_energy_density = max(sampled_max_energy_density, float(np.max(density)))
        sampled_max_abs_common = max(sampled_max_abs_common, float(np.max(np.abs(common))))
        sampled_max_abs_relative = max(sampled_max_abs_relative, float(np.max(np.abs(relative))))
        if keep_profiles and index in snapshot_steps:
            snapshots.append(snapshot_record(time, x_axis, common_line, relative_line))

    sample(0)
    for index in range(1, steps + 1):
        div1, div2 = step(state, dt, dx, coupling)
        cumulative_inflow_n1 += float(-dt * np.sum(div1[emission_region]) * dx**2)
        cumulative_inflow_n2 += float(-dt * np.sum(div2[emission_region]) * dx**2)
        if index in sample_steps:
            sample(index)

    relative_centers = profile_series["relative"]["absolute_centroid_x"]
    final_center = relative_centers[-1]
    if final_center is None:
        raise RuntimeError("relative packet centroid became unresolved")
    content = common_content(state, xx, yy, final_center, dx)
    attachment = attachment_verdict(profile_series, duration, profile_signal_floor)
    initial_energy_safe = max(abs(initial_energy), np.finfo(float).tiny)
    max_energy_density_scale = max(
        abs(sampled_min_energy_density), abs(sampled_max_energy_density), np.finfo(float).eps
    )
    energy_floor = float(
        64.0 * np.finfo(float).eps * size**2 * max_energy_density_scale * dx**2
    )
    final_region_n1 = region_n1[-1]
    final_region_n2 = region_n2[-1]
    final_region_energy = region_energy[-1]
    result: dict[str, Any] = {
        "amplitude": amplitude,
        "coupling_lambda": coupling,
        "lattice": {"size": size, "dx": dx, "dt": dt, "duration": duration},
        "initial_packet_energy": initial_energy,
        "measured_relative_profile_speed": attachment["relative_profile_speed"],
        "final_relative_absolute_centroid_x": final_center,
        "common_content": content,
        "attachment_diagnostic": attachment,
        "profile_time_series": profile_series,
        "conservation": {
            "instrument_floor": conservation_floor,
            "maximum_absolute_n1_total_drift": float(np.max(np.abs(n1_errors))),
            "maximum_absolute_n2_total_drift": float(np.max(np.abs(n2_errors))),
            "maximum_absolute_common_integral": float(np.max(np.abs(common_integrals))),
            "maximum_absolute_relative_integral": float(np.max(np.abs(relative_integrals))),
            "common_signal_roundoff_floor": profile_signal_floor,
        },
        "energy": {
            "initial": initial_energy,
            "final": energies[-1],
            "roundoff_floor": energy_floor,
            "maximum_absolute_drift": float(
                np.max(np.abs(np.asarray(energies) - initial_energy))
            ),
            "maximum_relative_drift": float(
                np.max(np.abs(np.asarray(energies) - initial_energy)) / initial_energy_safe
            ),
        },
        "sampled_stability_domain": {
            "minimum_n1": sampled_min_n1,
            "minimum_n2": sampled_min_n2,
            "maximum_absolute_n_plus": sampled_max_abs_common,
            "maximum_absolute_n_minus": sampled_max_abs_relative,
            "minimum_hamiltonian_density": sampled_min_energy_density,
            "maximum_hamiltonian_density": sampled_max_energy_density,
            "all_sampled_values_finite": bool(
                all(
                    np.isfinite(value)
                    for value in (
                        sampled_min_n1,
                        sampled_min_n2,
                        sampled_min_energy_density,
                        sampled_max_energy_density,
                    )
                )
            ),
        },
        "initialization_region": {
            "definition": (
                "fixed rectangle |x-x0| <= sigma_x and |y| <= 2 sigma_y; "
                "inflow is the time-integrated conservative divergence"
            ),
            "initial_n1_excess": initial_region_n1,
            "final_n1_excess": final_region_n1,
            "integrated_n1_inflow": inflow_n1[-1],
            "n1_budget_residual": final_region_n1 - initial_region_n1 - inflow_n1[-1],
            "initial_n2_excess": initial_region_n2,
            "final_n2_excess": final_region_n2,
            "integrated_n2_inflow": inflow_n2[-1],
            "n2_budget_residual": final_region_n2 - initial_region_n2 - inflow_n2[-1],
            "initial_energy": initial_region_energy,
            "final_energy": final_region_energy,
            "fraction_of_initial_energy_departed": 1.0
            - final_region_energy / initial_region_energy,
            "final_common_mode_integral": float(np.sum(modes(state)[0][emission_region]) * dx**2),
        },
        "budget_time_series": {
            "time": times,
            "n1_total_drift": n1_errors,
            "n2_total_drift": n2_errors,
            "common_mode_integral": common_integrals,
            "relative_mode_integral": relative_integrals,
            "total_energy": energies,
            "initialization_region_n1_excess": region_n1,
            "initialization_region_n2_excess": region_n2,
            "initialization_region_energy": region_energy,
            "integrated_n1_inflow": inflow_n1,
            "integrated_n2_inflow": inflow_n2,
        },
    }
    if keep_profiles:
        result["time_resolved_profile_snapshots"] = snapshots
    final_fields = None
    if keep_profiles:
        common, relative = modes(state)
        final_fields = {"x": xx, "y": yy, "common": common, "relative": relative}
    return result, final_fields


def fit_common_scaling(ladder: list[dict[str, Any]]) -> dict[str, Any]:
    energies = np.asarray([row["initial_packet_energy"] for row in ladder])
    contents = np.asarray([row["common_content"]["absolute_integral"] for row in ladder])
    floors = np.asarray([row["conservation"]["common_signal_roundoff_floor"] for row in ladder])
    resolved = contents > RESOLVED_SIGNAL_FLOORS * floors
    if int(np.count_nonzero(resolved)) < 3:
        return {
            "definition": "co-moving integral |n_plus| = prefactor * E^exponent",
            "resolved_points": int(np.count_nonzero(resolved)),
            "exponent": None,
            "prefactor": None,
            "verdict": "unresolved_at_numerical_floor",
        }
    exponent, log_prefactor = np.polyfit(
        np.log(energies[resolved]), np.log(contents[resolved]), 1
    )
    return {
        "definition": "co-moving integral |n_plus| = prefactor * E^exponent",
        "resolved_points": int(np.count_nonzero(resolved)),
        "exponent": float(exponent),
        "prefactor": float(np.exp(log_prefactor)),
        "verdict": "resolved_fit",
    }


def run_ladder(
    coupling: float,
    *,
    size: int = BASE_SIZE,
    dx: float | None = None,
    dt: float = BASE_DT,
    duration: float = DURATION,
    keep_primary_profiles: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None]:
    rows = []
    representative_fields = None
    for amplitude in AMPLITUDES:
        row, fields = run_packet(
            amplitude,
            coupling,
            size=size,
            dx=dx,
            dt=dt,
            duration=duration,
            keep_profiles=keep_primary_profiles and amplitude == PROFILE_AMPLITUDE,
        )
        rows.append(row)
        if fields is not None:
            representative_fields = fields
    return rows, representative_fields


def compact_ladder(coupling: float, ladder: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "coupling_lambda": coupling,
        "power_law": fit_common_scaling(ladder),
        "energy_ladder": [
            {
                "amplitude": row["amplitude"],
                "packet_energy": row["initial_packet_energy"],
                "co_moving_absolute_common_integral": row["common_content"]["absolute_integral"],
                "co_moving_signed_common_integral": row["common_content"]["signed_integral"],
                "positive_common_integral": row["common_content"]["positive_integral"],
                "negative_common_integral": row["common_content"]["negative_integral"],
                "common_signal_roundoff_floor": row["conservation"]["common_signal_roundoff_floor"],
                "attachment_verdict": row["attachment_diagnostic"]["verdict"],
                "median_lag_normalized_by_packet_width": row["attachment_diagnostic"][
                    "median_late_lag_normalized_by_packet_width"
                ],
            }
            for row in ladder
        ],
        "runs": ladder,
    }


def rung_summary(
    ladder: list[dict[str, Any]], *, size: int, dx: float, dt: float
) -> dict[str, Any]:
    power = fit_common_scaling(ladder)
    representative = ladder[-1]
    return {
        "size": size,
        "dx": dx,
        "dt": dt,
        "exponent": power["exponent"],
        "prefactor": power["prefactor"],
        "representative_lag_normalized_by_packet_width": representative[
            "attachment_diagnostic"
        ]["median_late_lag_normalized_by_packet_width"],
        "representative_attachment_verdict": representative["attachment_diagnostic"]["verdict"],
        "representative_relative_profile_speed": representative[
            "attachment_diagnostic"
        ]["relative_profile_speed"],
        "maximum_n1_total_drift": max(
            row["conservation"]["maximum_absolute_n1_total_drift"] for row in ladder
        ),
        "maximum_n2_total_drift": max(
            row["conservation"]["maximum_absolute_n2_total_drift"] for row in ladder
        ),
        "maximum_relative_energy_drift": max(
            row["energy"]["maximum_relative_drift"] for row in ladder
        ),
    }


def convergence_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous, finest = rows[-2], rows[-1]
    exponent_shift = abs(finest["exponent"] - previous["exponent"])
    prefactor_change = abs(finest["prefactor"] - previous["prefactor"]) / max(
        abs(finest["prefactor"]), np.finfo(float).tiny
    )
    lag_shift = abs(
        finest["representative_lag_normalized_by_packet_width"]
        - previous["representative_lag_normalized_by_packet_width"]
    )
    checks = {
        "exponent_absolute_shift_within_threshold": exponent_shift
        <= EXPONENT_CONVERGENCE_ABS,
        "prefactor_relative_change_within_threshold": prefactor_change
        <= PREFACTOR_CONVERGENCE_REL,
        "normalized_lag_absolute_shift_within_threshold": lag_shift
        <= LAG_CONVERGENCE_ABS_WIDTHS,
        "attachment_classification_unchanged": finest[
            "representative_attachment_verdict"
        ]
        == previous["representative_attachment_verdict"],
    }
    return {
        "predeclared_thresholds": {
            "exponent_absolute_shift_maximum": EXPONENT_CONVERGENCE_ABS,
            "prefactor_relative_change_maximum": PREFACTOR_CONVERGENCE_REL,
            "normalized_lag_absolute_shift_maximum": LAG_CONVERGENCE_ABS_WIDTHS,
            "attachment_classification_must_match": True,
        },
        "finest_successive_exponent_absolute_shift": exponent_shift,
        "finest_successive_prefactor_relative_change": prefactor_change,
        "finest_successive_normalized_lag_absolute_shift": lag_shift,
        "checks": checks,
        "verdict": "converged" if all(checks.values()) else "nonconverged",
    }


def convergence_study() -> dict[str, Any]:
    grid_rows = []
    for size in GRID_RUNGS:
        dx = DOMAIN_LENGTH / size
        ladder, _ = run_ladder(
            PRIMARY_COUPLING,
            size=size,
            dx=dx,
            dt=BASE_DT,
            duration=CONVERGENCE_DURATION,
        )
        grid_rows.append(rung_summary(ladder, size=size, dx=dx, dt=BASE_DT))

    time_rows = []
    for dt in TIMESTEP_RUNGS:
        ladder, _ = run_ladder(
            PRIMARY_COUPLING,
            size=BASE_SIZE,
            dx=DOMAIN_LENGTH / BASE_SIZE,
            dt=dt,
            duration=CONVERGENCE_DURATION,
        )
        time_rows.append(
            rung_summary(
                ladder,
                size=BASE_SIZE,
                dx=DOMAIN_LENGTH / BASE_SIZE,
                dt=dt,
            )
        )
    return {
        "coupling_lambda": PRIMARY_COUPLING,
        "amplitudes": list(AMPLITUDES),
        "duration": CONVERGENCE_DURATION,
        "grid_refinement": {
            "rungs": grid_rows,
            "comparison": convergence_comparison(grid_rows),
        },
        "time_step_refinement": {
            "rungs": time_rows,
            "comparison": convergence_comparison(time_rows),
        },
        "scope": (
            "Every rung refits the four-amplitude primary G3 exponent and prefactor "
            "and reclassifies the representative attachment lag; speed is secondary."
        ),
    }


def hamiltonian_audit(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sampled = [row["sampled_stability_domain"] for row in all_rows]
    minimum_n1 = min(row["minimum_n1"] for row in sampled)
    minimum_n2 = min(row["minimum_n2"] for row in sampled)
    minimum_density = min(minimum_n1, minimum_n2)
    minimum_hamiltonian = min(row["minimum_hamiltonian_density"] for row in sampled)
    maximum_hamiltonian = max(row["maximum_hamiltonian_density"] for row in sampled)
    return {
        "original_hamiltonian": (
            "H=(|j1|^2+|j2|^2)/2+c_plus^2*n_plus^2/4+"
            "c_minus^2*n_minus^2/4+lambda*n_plus*n_minus^2/4"
        ),
        "analytic_unconstrained_boundedness": {
            "lambda_zero": "nonnegative quadratic and bounded below by zero",
            "nonzero_lambda": "unbounded below",
            "witness": (
                "At fixed n_minus=m, minimize over n_plus: n_plus=-lambda*m^2/"
                "(2*c_plus^2), giving V_min=c_minus^2*m^2/4-"
                "lambda^2*m^4/(16*c_plus^2), which tends to -infinity."
            ),
            "verdict": "the original cubic Hamiltonian is globally unstable for every nonzero lambda",
        },
        "density_positive_domain_note": (
            "Physical n1,n2>=0 imply n_plus>=|n_minus|-2*n0 and remove the "
            "unconstrained quartic witness for the sampled positive-lambda arm, but the "
            "finite-difference evolution has no positivity-preserving limiter, so this "
            "restriction is monitored rather than assumed."
        ),
        "sampled_domain": {
            "coupling_lambda_minimum": min(COUPLINGS),
            "coupling_lambda_maximum": max(COUPLINGS),
            "amplitude_minimum": min(AMPLITUDES),
            "amplitude_maximum": max(AMPLITUDES),
            "lattice_size": BASE_SIZE,
            "time_step": BASE_DT,
            "duration": DURATION,
            "minimum_n1": minimum_n1,
            "minimum_n2": minimum_n2,
            "densities_remained_strictly_positive": minimum_density > 0.0,
            "minimum_hamiltonian_density": minimum_hamiltonian,
            "maximum_hamiltonian_density": maximum_hamiltonian,
            "hamiltonian_density_nonnegative_at_samples": minimum_hamiltonian >= 0.0,
            "all_values_finite": all(row["all_sampled_values_finite"] for row in sampled),
        },
        "stabilizing_variant": {
            "tested": False,
            "status": "not added; any quartic stabilizer is a separate model hypothesis",
        },
        "verdict": (
            "sampled trajectories remain positive and finite, but this does not cure "
            "the original nonzero-lambda Hamiltonian's global unboundedness"
        ),
    }


def defect_emission_gate() -> dict[str, Any]:
    return {
        "tested_model_inventory": [
            "this fixture evolves n1,n2,j1,j2 from a directly initialized relative packet",
            "the companion defect fixture pins analytic winding/core profiles",
            "its moving wake uses an externally translated relaxation target",
        ],
        "endogenous_emission_supported": False,
        "verdict": "unsupported_missing_emission_operator",
        "exact_missing_operator": (
            "No declared equation couples an autonomous defect/core degree of freedom "
            "and its energy to an outgoing J_minus (or n_minus boundary flux). Such a "
            "defect-to-relative-mode emission term, together with a conservative "
            "defect-plus-field energy/refill budget, is required before emission can be run."
        ),
        "ad_hoc_source_added": False,
        "reason_no_emission_arm_was_run": (
            "Initializing a packet or reusing the companion's externally imposed "
            "target-relaxation term would prescribe emission rather than test it."
        ),
    }


def analyze() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    coupling_results = []
    all_rows = []
    representative_fields = None
    for coupling in COUPLINGS:
        ladder, fields = run_ladder(
            coupling,
            keep_primary_profiles=coupling == PRIMARY_COUPLING,
        )
        coupling_results.append(compact_ladder(coupling, ladder))
        all_rows.extend(ladder)
        if fields is not None:
            representative_fields = fields
    if representative_fields is None:
        raise RuntimeError("representative field capture failed")

    null_rows = coupling_results[0]["runs"]
    predicted_null_floor = max(
        row["conservation"]["common_signal_roundoff_floor"] for row in null_rows
    )
    measured_null_signal = max(
        row["common_content"]["absolute_integral"] for row in null_rows
    )
    null_control_floor = max(predicted_null_floor, measured_null_signal)
    primary = coupling_results[-1]
    primary_run = primary["runs"][-1]
    maximum_common_monopole = max(
        row["conservation"]["maximum_absolute_common_integral"] for row in all_rows
    )
    maximum_relative_monopole = max(
        row["conservation"]["maximum_absolute_relative_integral"] for row in all_rows
    )
    conservation_floor = max(
        row["conservation"]["instrument_floor"] for row in all_rows
    )

    lambda_scaling = []
    nonzero = coupling_results[1:]
    for amplitude_index, amplitude in enumerate(AMPLITUDES):
        lambdas = np.asarray([entry["coupling_lambda"] for entry in nonzero])
        signals = np.asarray(
            [entry["energy_ladder"][amplitude_index]["co_moving_absolute_common_integral"] for entry in nonzero]
        )
        exponent, log_prefactor = np.polyfit(np.log(lambdas), np.log(signals), 1)
        lambda_scaling.append(
            {
                "amplitude": amplitude,
                "definition": "co-moving integral |n_plus| = prefactor * lambda^exponent",
                "exponent": float(exponent),
                "prefactor": float(np.exp(log_prefactor)),
            }
        )

    result: dict[str, Any] = {
        "schema_version": 2,
        "task": "ORB-10941",
        "run_date": "2026-08-21",
        "question": (
            "Is the common response packet-attached or a coupling-driven trailing wake, "
            "is the primary scaling converged, and can the existing defect emit it?"
        ),
        "apparatus": {
            "dimensions": 2,
            "lattice": [BASE_SIZE, BASE_SIZE],
            "domain_length": DOMAIN_LENGTH,
            "boundary": "periodic square",
            "time_step": BASE_DT,
            "duration": DURATION,
            "base_density_per_substance": BASE_DENSITY,
            "common_mode_speed": COMMON_SPEED,
            "relative_mode_speed": RELATIVE_SPEED,
            "coupling_ladder": list(COUPLINGS),
            "amplitude_ladder": list(AMPLITUDES),
            "seed": None,
            "determinism": "no random numbers; fixed operation order and sorted JSON keys",
        },
        "predeclared_dynamics_and_thresholds": {
            "hamiltonian_density": (
                "(|j1|^2+|j2|^2)/2+c_plus^2*n_plus^2/4+"
                "c_minus^2*n_minus^2/4+lambda*n_plus*n_minus^2/4"
            ),
            "continuity": "partial_t n_s=-div(j_s), independently for s=1,2",
            "flux_update": "partial_t j_s=-grad(partial H/partial n_s)",
            "packet_initialization": (
                "direct localized zero-integral n_minus packet with n_plus=J_plus=0; "
                "this is not defect emission"
            ),
            "attachment": attachment_verdict(
                primary_run["profile_time_series"], DURATION, primary_run["conservation"]["common_signal_roundoff_floor"]
            )["predeclared_evaluation"],
            "convergence": {
                "exponent_absolute_shift_maximum": EXPONENT_CONVERGENCE_ABS,
                "prefactor_relative_change_maximum": PREFACTOR_CONVERGENCE_REL,
                "normalized_lag_absolute_shift_maximum": LAG_CONVERGENCE_ABS_WIDTHS,
                "classification_must_match": True,
            },
        },
        "G1_coupling_and_energy_controls": {
            "null_control": {
                "coupling_lambda": 0.0,
                "predicted_roundoff_floor": predicted_null_floor,
                "maximum_measured_absolute_common_integral": measured_null_signal,
                "declared_null_control_floor": null_control_floor,
                "verdict": (
                    "null_at_numerical_floor"
                    if measured_null_signal <= predicted_null_floor
                    else "unexpected_common_signal"
                ),
            },
            "coupling_ladders": coupling_results,
            "common_signal_scaling_with_nonzero_lambda": lambda_scaling,
            "interpretation": (
                "The common response is measured as a function of both energy and the "
                "inserted coupling; lambda=0 is the causal control."
            ),
        },
        "G2_time_resolved_attachment_or_wake": {
            "representative_coupling_lambda": PRIMARY_COUPLING,
            "representative_amplitude": PROFILE_AMPLITUDE,
            "profile_time_series": primary_run["profile_time_series"],
            "profile_snapshots": primary_run["time_resolved_profile_snapshots"],
            "classification": primary_run["attachment_diagnostic"],
        },
        "G3_primary_observable_convergence": convergence_study(),
        "G4_defect_emission_gate": defect_emission_gate(),
        "G5_hamiltonian_stability_audit": hamiltonian_audit(all_rows),
        "G6_global_budgets": {
            "interpretation": (
                "Global n_plus is the signed common monopole fixed by separate substance "
                "conservation. Local |n_plus| is not net gravitational mass."
            ),
            "per_substance_and_monopole_instrument_floor": conservation_floor,
            "maximum_absolute_n1_total_drift": max(
                row["conservation"]["maximum_absolute_n1_total_drift"] for row in all_rows
            ),
            "maximum_absolute_n2_total_drift": max(
                row["conservation"]["maximum_absolute_n2_total_drift"] for row in all_rows
            ),
            "maximum_absolute_common_monopole": maximum_common_monopole,
            "maximum_absolute_relative_total": maximum_relative_monopole,
            "maximum_relative_energy_drift": max(
                row["energy"]["maximum_relative_drift"] for row in all_rows
            ),
            "maximum_absolute_energy_drift": max(
                row["energy"]["maximum_absolute_drift"] for row in all_rows
            ),
            "maximum_energy_roundoff_floor": max(
                row["energy"]["roundoff_floor"] for row in all_rows
            ),
            "representative_budget_time_series": primary_run["budget_time_series"],
            "representative_initialization_region": primary_run["initialization_region"],
        },
        "conclusions": {
            "attachment_or_wake": primary_run["attachment_diagnostic"]["verdict"],
            "primary_grid_convergence": None,
            "primary_timestep_convergence": None,
            "emission": "unsupported_missing_emission_operator",
            "stability": "globally_unbounded_for_nonzero_lambda",
            "interpretation_boundary": (
                "No gravity-source law, E/c^2 normalization, observational fit, or claim "
                "that local absolute common density is net gravitational mass is made."
            ),
        },
    }
    grid_verdict = result["G3_primary_observable_convergence"]["grid_refinement"]["comparison"]["verdict"]
    time_verdict = result["G3_primary_observable_convergence"]["time_step_refinement"]["comparison"]["verdict"]
    result["conclusions"]["primary_grid_convergence"] = grid_verdict
    result["conclusions"]["primary_timestep_convergence"] = time_verdict
    validate_result(result)
    return result, representative_fields


def validate_result(result: dict[str, Any]) -> None:
    controls = result["G1_coupling_and_energy_controls"]
    profiles = result["G2_time_resolved_attachment_or_wake"]
    convergence = result["G3_primary_observable_convergence"]
    emission = result["G4_defect_emission_gate"]
    stability = result["G5_hamiltonian_stability_audit"]
    budgets = result["G6_global_budgets"]
    floor = budgets["per_substance_and_monopole_instrument_floor"]
    coupling_entries = controls["coupling_ladders"]
    checks = {
        "zero plus three nonzero couplings run": len(coupling_entries) >= 4
        and coupling_entries[0]["coupling_lambda"] == 0.0
        and sum(entry["coupling_lambda"] != 0.0 for entry in coupling_entries) >= 3,
        "full amplitude ladder at every coupling": all(
            len(entry["energy_ladder"]) == len(AMPLITUDES) for entry in coupling_entries
        ),
        "null control at predicted floor": controls["null_control"]["verdict"]
        == "null_at_numerical_floor",
        "time-resolved profile diagnostics recorded": len(profiles["profile_time_series"]["time"]) >= 3
        and len(profiles["profile_snapshots"]) == len(PROFILE_SNAPSHOT_FRACTIONS),
        "three grid rungs recorded": len(convergence["grid_refinement"]["rungs"]) >= 3,
        "three timestep rungs recorded": len(convergence["time_step_refinement"]["rungs"]) >= 3,
        "each refinement rung fits exponent prefactor and lag": all(
            row["exponent"] is not None
            and row["prefactor"] is not None
            and row["representative_lag_normalized_by_packet_width"] is not None
            for study in ("grid_refinement", "time_step_refinement")
            for row in convergence[study]["rungs"]
        ),
        "defect gate names missing operator": not emission["endogenous_emission_supported"]
        and bool(emission["exact_missing_operator"])
        and not emission["ad_hoc_source_added"],
        "Hamiltonian unboundedness explicitly audited": stability[
            "analytic_unconstrained_boundedness"
        ]["nonzero_lambda"]
        == "unbounded below",
        "sampled densities remain positive": stability["sampled_domain"][
            "densities_remained_strictly_positive"
        ],
        "sampled values finite": stability["sampled_domain"]["all_values_finite"],
        "n1 conserved at instrument floor": budgets["maximum_absolute_n1_total_drift"]
        <= floor,
        "n2 conserved at instrument floor": budgets["maximum_absolute_n2_total_drift"]
        <= floor,
        "global common monopole at floor": budgets["maximum_absolute_common_monopole"]
        <= floor,
        "global relative total at floor": budgets["maximum_absolute_relative_total"]
        <= floor,
        "energy drift remains resolved": budgets["maximum_relative_energy_drift"] < 1.0e-4,
    }
    result["validation"] = {"checks": checks, "all_passed": all(checks.values())}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("validation failed: " + "; ".join(failed))


def make_plot(result: dict[str, Any], fields: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    extent = [
        float(fields["x"].min()),
        float(fields["x"].max()),
        float(fields["y"].min()),
        float(fields["y"].max()),
    ]
    for axis, field_name, title in (
        (axes[0, 0], "common", "Final common mode n+"),
        (axes[0, 1], "relative", "Final relative mode n-"),
    ):
        values = fields[field_name]
        limit = float(np.max(np.abs(values)))
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(image, ax=axis)

    controls = result["G1_coupling_and_energy_controls"]["coupling_ladders"]
    for entry in controls:
        energy = np.asarray([row["packet_energy"] for row in entry["energy_ladder"]])
        signal = np.asarray(
            [row["co_moving_absolute_common_integral"] for row in entry["energy_ladder"]]
        )
        axes[1, 0].loglog(
            energy,
            np.maximum(signal, entry["runs"][0]["conservation"]["common_signal_roundoff_floor"]),
            "o-",
            label=f"lambda={entry['coupling_lambda']:.1f}",
        )
    axes[1, 0].set(
        title="Common content: coupling and energy controls",
        xlabel="initial packet energy",
        ylabel="co-moving integral |n+|",
    )
    axes[1, 0].legend()

    series = result["G2_time_resolved_attachment_or_wake"]["profile_time_series"]
    axes[1, 1].plot(
        series["time"],
        series["lag_normalized_by_relative_width"],
        color="#caa6ff",
        label="common-relative lag / width",
    )
    axes[1, 1].axhline(-ATTACHED_LAG_WIDTHS, color="#ff8a80", ls="--", label="wake threshold")
    axes[1, 1].axhline(ATTACHED_LAG_WIDTHS, color="#78d69b", ls=":")
    axes[1, 1].set(
        title=(
            "Profile classification: "
            + result["G2_time_resolved_attachment_or_wake"]["classification"]["verdict"]
        ),
        xlabel="time",
        ylabel="normalized centroid lag",
    )
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("Two-substance packet: coupling controls and wake test")
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
    result, fields = analyze()
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    make_plot(result, fields, args.output_dir / "packet-budget.png")
    classification = result["G2_time_resolved_attachment_or_wake"]["classification"]
    convergence = result["G3_primary_observable_convergence"]
    print(f"attachment diagnostic: {classification['verdict']}")
    print(
        "primary convergence: grid "
        f"{convergence['grid_refinement']['comparison']['verdict']}; timestep "
        f"{convergence['time_step_refinement']['comparison']['verdict']}"
    )
    print(
        "emission: "
        f"{result['G4_defect_emission_gate']['verdict']}; Hamiltonian: "
        f"{result['G5_hamiltonian_stability_audit']['analytic_unconstrained_boundedness']['nonzero_lambda']}"
    )


if __name__ == "__main__":
    main()
