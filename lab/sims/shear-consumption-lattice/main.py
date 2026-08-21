"""Test local von-Mises-strain destruction on a dynamical substrate lattice.

The primary apparatus is a spherical finite-volume lattice.  Substrate moves
between neighbouring shells down density gradients; the face velocity is
therefore measured from the transported flux, never prescribed as a radial
profile.  A fixed core flux anchors the flow.  Every shell destroys substrate
at the frozen local rate

    s = n * sqrt((3/2) * e_dev:e_dev),

which is ``n * abs(-du/dr + u/r)`` for radial inward speed ``u``.  Semi-implicit
steps evolve the density from either rest or a seeded perturbation to a steady
state.  A manufactured 3-D Hubble flow tests the zero-shear limit.  A 3-D
Cartesian companion lattice supplies the resolution-converged two-core
superposition measurement.

Run with:
    uv run lab/sims/shear-consumption-lattice/main.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn
from scipy.optimize import curve_fit
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import factorized


SEED = 42
INNER_RADIUS = 1.0
OUTER_RADIUS = 32.0
DIFFUSIVITY = 2.0
RESERVOIR_DENSITY = 1.0
PRIMARY_SHELLS = 256
PRIMARY_TIME_STEP = 0.1
PRIMARY_CORE_FLUX = 0.0001
FIT_RADIUS_RANGE = (3.0, 20.0)
STEADY_TOLERANCE = 1.0e-9
MAX_RADIAL_STEPS = 20_000
RUN_RECORD = "2026-08-21-seed-42.json"
CARTESIAN_HALF_WIDTH = 12.0
CARTESIAN_TIME_STEP = 2.0
CARTESIAN_TOLERANCE = 1.0e-8
CORE_SIGMA = 0.75
TWO_CORE_GRID_SIZES = (25, 41, 65)
TWO_CORE_TARGET_POSITION = (-4.0, 0.0, 0.0)
TWO_CORE_NEIGHBOUR_POSITION = (4.0, 0.0, 0.0)
TWO_CORE_TARGET_RATE = 0.1
TWO_CORE_NEIGHBOUR_RATES = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
SEPARATION_CONTROL_VALUES = (6.0, 8.0, 10.0)
SEPARATION_CONTROL_RATE = 1.6


@dataclass(frozen=True)
class PowerFit:
    exponent: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    intercept: float
    samples: int
    radial_decades: float


def power_fit(radius: np.ndarray, values: np.ndarray) -> PowerFit:
    x = np.log(radius)
    y = np.log(values)
    design = np.column_stack((np.ones_like(x), x))
    intercept, exponent = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([intercept, exponent])
    variance = float(np.sum(residual**2) / (len(x) - 2))
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(covariance[1, 1]))
    return PowerFit(
        exponent=float(exponent),
        standard_error=standard_error,
        ci95_low=float(exponent - 1.96 * standard_error),
        ci95_high=float(exponent + 1.96 * standard_error),
        intercept=float(intercept),
        samples=len(radius),
        radial_decades=float(np.log10(radius.max() / radius.min())),
    )


def radial_geometry(shells: int) -> tuple[np.ndarray, ...]:
    edges = np.linspace(INNER_RADIUS, OUTER_RADIUS, shells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    areas = 4.0 * np.pi * edges**2
    return edges, centers, volumes, areas


def radial_diffusion_operator(
    edges: np.ndarray, volumes: np.ndarray, areas: np.ndarray
) -> tuple[csc_matrix, np.ndarray, np.ndarray]:
    """Return density diffusion with fixed inner flux and outer reservoir."""
    shells = len(volumes)
    dr = edges[1] - edges[0]
    conductance = np.zeros(shells + 1)
    conductance[1:shells] = DIFFUSIVITY * areas[1:shells] / dr
    conductance[-1] = DIFFUSIVITY * areas[-1] / (0.5 * dr)
    lower = conductance[1:shells] / volumes[1:]
    upper = conductance[1:shells] / volumes[:-1]
    diagonal = -(conductance[:-1] + conductance[1:]) / volumes
    operator = diags((lower, diagonal, upper), (-1, 0, 1), format="csc")
    outer_boundary = np.zeros(shells)
    outer_boundary[-1] = conductance[-1] * RESERVOIR_DENSITY / volumes[-1]
    return operator, outer_boundary, conductance


def radial_observables(
    density: np.ndarray,
    edges: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
    conductance: np.ndarray,
    core_flux: float,
) -> dict[str, np.ndarray]:
    shells = len(density)
    extended = np.concatenate((density, [RESERVOIR_DENSITY]))
    inward_flux = np.empty(shells + 1)
    inward_flux[0] = core_flux
    inward_flux[1:] = conductance[1:] * (extended[1:] - extended[:-1])
    face_density = np.empty(shells + 1)
    face_density[0] = density[0]
    face_density[1:] = 0.5 * (extended[1:] + extended[:-1])
    face_speed = inward_flux / (areas * face_density)
    cell_speed = 0.5 * (face_speed[:-1] + face_speed[1:])
    dr = edges[1] - edges[0]
    radial_shear = -(face_speed[1:] - face_speed[:-1]) / dr + cell_speed / centers
    consumption_density = density * np.abs(radial_shear)
    return {
        "inward_flux": inward_flux,
        "face_speed": face_speed,
        "cell_speed": cell_speed,
        "radial_shear": radial_shear,
        "consumption_density": consumption_density,
    }


def radial_initial_state(
    centers: np.ndarray, core_flux: float, initial_condition: str, seed: int
) -> np.ndarray:
    if initial_condition == "rest":
        return np.full_like(centers, RESERVOIR_DENSITY)
    if initial_condition != "perturbed":
        raise ValueError(f"unknown initial condition: {initial_condition}")
    random = np.random.default_rng(seed)
    # Keep the perturbation bounded even for the strong-core amplitude branch;
    # it is deliberately a distinct state, not an attempted steady profile.
    envelope = 0.012 * centers**-0.5
    perturbation = 1.0 + 0.35 * random.normal(size=len(centers))
    return np.clip(RESERVOIR_DENSITY - envelope * perturbation, 0.8, 1.0)


def run_radial(
    shells: int,
    time_step: float,
    core_flux: float,
    initial_condition: str,
    seed: int = SEED,
) -> tuple[dict, dict[str, np.ndarray]]:
    edges, centers, volumes, areas = radial_geometry(shells)
    operator, boundary, conductance = radial_diffusion_operator(edges, volumes, areas)
    boundary = boundary.copy()
    boundary[0] -= core_flux / volumes[0]
    advance = factorized(
        csc_matrix(diags(np.ones(shells)) - time_step * operator)
    )
    density = radial_initial_state(centers, core_flux, initial_condition, seed)
    relative_change = np.inf

    for step in range(1, MAX_RADIAL_STEPS + 1):
        observed = radial_observables(
            density, edges, centers, areas, conductance, core_flux
        )
        next_density = advance(
            density + time_step * (boundary - observed["consumption_density"])
        )
        deficit_scale = max(
            float(np.max(RESERVOIR_DENSITY - next_density)), 1.0e-15
        )
        relative_change = float(np.max(np.abs(next_density - density)) / deficit_scale)
        density = next_density
        if relative_change < STEADY_TOLERANCE:
            break
    else:
        raise RuntimeError("radial lattice did not reach the steady tolerance")

    observed = radial_observables(
        density, edges, centers, areas, conductance, core_flux
    )
    fit_mask = (
        (centers >= FIT_RADIUS_RANGE[0])
        & (centers <= FIT_RADIUS_RANGE[1])
        & (observed["cell_speed"] > 0.0)
    )
    fitted = power_fit(centers[fit_mask], observed["cell_speed"][fit_mask])
    template = centers[fit_mask] ** -0.5
    amplitude = float(
        np.dot(template, observed["cell_speed"][fit_mask])
        / np.dot(template, template)
    )
    predicted = amplitude * template
    relative_rmse = float(
        np.sqrt(
            np.mean(
                ((observed["cell_speed"][fit_mask] - predicted) / predicted) ** 2
            )
        )
    )
    balance = (
        operator @ density + boundary - observed["consumption_density"]
    ) * volumes
    distributed_consumption = float(
        np.sum(observed["consumption_density"] * volumes)
    )
    result = {
        "shells": shells,
        "time_step": time_step,
        "core_flux": core_flux,
        "initial_condition": initial_condition,
        "seed": seed,
        "steady_state": {
            "reached": True,
            "steps": step,
            "elapsed_time": step * time_step,
            "last_step_change_over_max_deficit": relative_change,
            "max_shell_balance_over_total_consumption": float(
                np.max(np.abs(balance)) / (core_flux + distributed_consumption)
            ),
            "minimum_density": float(np.min(density)),
            "maximum_density": float(np.max(density)),
            "outer_reservoir_flux": float(observed["inward_flux"][-1]),
            "core_flux": core_flux,
            "distributed_shear_consumption": distributed_consumption,
        },
        "flow_fit": asdict(fitted),
        "fixed_half_power_amplitude": amplitude,
        "fixed_half_power_relative_rmse": relative_rmse,
    }
    profiles = {"radius": centers, "density": density, **observed}
    return result, profiles


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


def silence_gate(size: int = 17, spacing: float = 0.5, hubble_rate: float = 0.02) -> dict:
    coordinates = (np.arange(size) - 0.5 * (size - 1)) * spacing
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    density = np.ones_like(x)
    velocity = hubble_rate * np.stack((x, y, z))
    consumption = strain_consumption_3d(density, velocity, spacing)
    interior = consumption[2:-2, 2:-2, 2:-2]
    isotropic_scale = density[2:-2, 2:-2, 2:-2] * abs(hubble_rate)
    return {
        "lattice_size": size,
        "spacing": spacing,
        "hubble_rate": hubble_rate,
        "maximum_consumption_density": float(np.max(interior)),
        "rms_consumption_density": float(np.sqrt(np.mean(interior**2))),
        "maximum_over_n_abs_H": float(np.max(interior / isotropic_scale)),
        "consistent_with_zero": bool(np.max(interior) < 1.0e-12),
    }


def cartesian_setup(size: int, half_width: float, time_step: float) -> dict:
    spacing = 2.0 * half_width / (size + 1)
    coordinates = np.linspace(-half_width + spacing, half_width - spacing, size)
    xyz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    mode = np.arange(1, size + 1)
    eigenvalue = 2.0 * (np.cos(np.pi * mode / (size + 1)) - 1.0) / spacing**2
    laplacian_eigenvalue = (
        eigenvalue[:, None, None]
        + eigenvalue[None, :, None]
        + eigenvalue[None, None, :]
    )
    denominator = 1.0 - time_step * DIFFUSIVITY * laplacian_eigenvalue
    return {
        "spacing": spacing,
        "coordinates": coordinates,
        "xyz": xyz,
        "laplacian_eigenvalue": laplacian_eigenvalue,
        "denominator": denominator,
    }


def gaussian_core_source(
    xyz: tuple[np.ndarray, ...],
    position: tuple[float, float, float],
    rate: float,
    spacing: float,
    sigma: float = CORE_SIGMA,
) -> np.ndarray:
    radius_squared = sum((axis - center) ** 2 for axis, center in zip(xyz, position))
    # The source has a fixed physical width.  Tying sigma to the cell spacing,
    # as the exploratory 25^3 probe did, changes the model under refinement.
    weights = np.exp(-radius_squared / (2.0 * sigma**2))
    return rate * weights / (np.sum(weights) * spacing**3)


def cartesian_fields(deficit: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    density = 1.0 - deficit
    if not np.all(np.isfinite(density)) or float(np.min(density)) <= 0.0:
        raise RuntimeError("Cartesian lattice reached non-positive density")
    deficit_gradient = np.gradient(deficit, spacing, edge_order=2)
    velocity = np.stack(
        [DIFFUSIVITY * component / density for component in deficit_gradient]
    )
    consumption = strain_consumption_3d(density, velocity, spacing)
    return consumption, velocity


def run_cartesian(
    sources: list[tuple[tuple[float, float, float], float]],
    size: int = 25,
    half_width: float = 12.0,
    time_step: float = CARTESIAN_TIME_STEP,
    tolerance: float = CARTESIAN_TOLERANCE,
    max_steps: int = 1000,
    core_sigma: float = CORE_SIGMA,
) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    apparatus = cartesian_setup(size, half_width, time_step)
    spacing = apparatus["spacing"]
    forcing = sum(
        (
            gaussian_core_source(
                apparatus["xyz"], position, rate, spacing, core_sigma
            )
            for position, rate in sources
        ),
        np.zeros((size, size, size)),
    )
    transformed_forcing = dstn(forcing, type=1, norm="ortho")
    deficit = idstn(
        transformed_forcing
        / (-DIFFUSIVITY * apparatus["laplacian_eigenvalue"]),
        type=1,
        norm="ortho",
    )
    relative_change = np.inf
    for step in range(1, max_steps + 1):
        consumption, _ = cartesian_fields(deficit, spacing)
        right_hand_side = deficit + time_step * (forcing + consumption)
        next_deficit = idstn(
            dstn(right_hand_side, type=1, norm="ortho")
            / apparatus["denominator"],
            type=1,
            norm="ortho",
        )
        relative_change = float(
            np.max(np.abs(next_deficit - deficit))
            / max(float(np.max(next_deficit)), 1.0e-15)
        )
        deficit = next_deficit
        if relative_change < tolerance:
            break
    else:
        raise RuntimeError("Cartesian lattice did not reach the steady tolerance")
    consumption, velocity = cartesian_fields(deficit, spacing)
    record = {
        "steps": step,
        "last_step_change_over_max_deficit": relative_change,
        "maximum_deficit": float(np.max(deficit)),
        "integrated_shear_consumption": float(np.sum(consumption) * spacing**3),
    }
    return record, deficit, velocity, apparatus


def trilinear_sample(
    field: np.ndarray,
    coordinates: np.ndarray,
    position: tuple[float, float, float],
) -> float:
    """Sample a scalar cell-centred field at a fixed physical position."""
    lower = []
    fractions = []
    for component in position:
        upper_index = int(np.searchsorted(coordinates, component))
        upper_index = min(max(upper_index, 1), len(coordinates) - 1)
        lower_index = upper_index - 1
        lower.append(lower_index)
        fractions.append(
            float(
                (component - coordinates[lower_index])
                / (coordinates[upper_index] - coordinates[lower_index])
            )
        )
    sampled = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = (
                    (fractions[0] if dx else 1.0 - fractions[0])
                    * (fractions[1] if dy else 1.0 - fractions[1])
                    * (fractions[2] if dz else 1.0 - fractions[2])
                )
                sampled += weight * field[
                    lower[0] + dx, lower[1] + dy, lower[2] + dz
                ]
    return float(sampled)


def projected_source_amplitude(
    paired_velocity: np.ndarray,
    background_velocity: np.ndarray,
    isolated_velocity: np.ndarray,
    xyz: tuple[np.ndarray, ...],
    source_position: tuple[float, float, float],
) -> float:
    radius = np.sqrt(
        sum((axis - center) ** 2 for axis, center in zip(xyz, source_position))
    )
    mask = (radius >= 2.0) & (radius <= 5.0)
    incremental = paired_velocity[:, mask] - background_velocity[:, mask]
    reference = isolated_velocity[:, mask]
    return float(np.sum(incremental * reference) / np.sum(reference * reference))


def fit_modulation_families(sweep: list[dict]) -> dict:
    """Fit the three predeclared families and report log-space RMSE."""
    depletion = np.array([row["ambient_depletion_at_target"] for row in sweep])
    amplitude = np.array([row["measured_incremental_amplitude"] for row in sweep])
    if np.any(depletion <= 0.0) or np.any(depletion >= 1.0) or np.any(amplitude <= 0.0):
        raise RuntimeError("modulation fit received values outside its log domain")
    log_headroom = np.log1p(-depletion)
    log_amplitude = np.log(amplitude)
    headroom_alpha = float(
        np.dot(log_headroom, log_amplitude) / np.dot(log_headroom, log_headroom)
    )

    def rational_modulation(
        d: np.ndarray, coefficient: float, exponent: float
    ) -> np.ndarray:
        return 1.0 / (1.0 + coefficient * d**exponent)

    parameters, _ = curve_fit(
        rational_modulation,
        depletion,
        amplitude,
        p0=(6.0, 0.65),
        bounds=(0.0, np.inf),
        maxfev=20_000,
    )
    rational_prediction = rational_modulation(depletion, *parameters)
    orb_prediction = (1.0 - depletion) ** 1.071
    return {
        "best_two_parameter_form": {
            "form": "A(D) = 1 / (1 + c D^beta)",
            "c": float(parameters[0]),
            "beta": float(parameters[1]),
            "log_rmse": float(
                np.sqrt(np.mean((log_amplitude - np.log(rational_prediction)) ** 2))
            ),
        },
        "best_headroom_power": {
            "form": "A(D) = (1-D)^alpha, constrained to A(0)=1",
            "alpha": headroom_alpha,
            "log_rmse": float(
                np.sqrt(
                    np.mean((log_amplitude - headroom_alpha * log_headroom) ** 2)
                )
            ),
        },
        "ORB_10157_screening_family": {
            "form": "A(D) = (1-D)^1.071",
            "alpha": 1.071,
            "log_rmse": float(
                np.sqrt(np.mean((log_amplitude - np.log(orb_prediction)) ** 2))
            ),
        },
    }


def run_two_core_sweep(
    size: int,
    neighbour_rates: tuple[float, ...] = TWO_CORE_NEIGHBOUR_RATES,
    target_position: tuple[float, float, float] = TWO_CORE_TARGET_POSITION,
    neighbour_position: tuple[float, float, float] = TWO_CORE_NEIGHBOUR_POSITION,
    time_step: float = CARTESIAN_TIME_STEP,
) -> dict:
    """Measure target-core modulation for one physical configuration."""
    isolated_record, _, isolated_velocity, apparatus = run_cartesian(
        [(target_position, TWO_CORE_TARGET_RATE)],
        size=size,
        half_width=CARTESIAN_HALF_WIDTH,
        time_step=time_step,
    )
    sweep = []
    for rate in neighbour_rates:
        background_record, background_deficit, background_velocity, _ = run_cartesian(
            [(neighbour_position, rate)],
            size=size,
            half_width=CARTESIAN_HALF_WIDTH,
            time_step=time_step,
        )
        paired_record, _, paired_velocity, _ = run_cartesian(
            [(target_position, TWO_CORE_TARGET_RATE), (neighbour_position, rate)],
            size=size,
            half_width=CARTESIAN_HALF_WIDTH,
            time_step=time_step,
        )
        depletion = trilinear_sample(
            background_deficit, apparatus["coordinates"], target_position
        )
        amplitude = projected_source_amplitude(
            paired_velocity,
            background_velocity,
            isolated_velocity,
            apparatus["xyz"],
            target_position,
        )
        sweep.append(
            {
                "neighbour_core_rate": rate,
                "ambient_depletion_at_target": depletion,
                "measured_incremental_amplitude": amplitude,
                "ORB_10157_screening_amplitude": float(
                    (1.0 - depletion) ** 1.071
                ),
                "background_steady_state": background_record,
                "paired_steady_state": paired_record,
            }
        )
    result = {
        "apparatus": {
            "lattice": f"{size}^3 Cartesian cells with a full Dirichlet reservoir",
            "half_width": CARTESIAN_HALF_WIDTH,
            "spacing": float(apparatus["spacing"]),
            "time_step": time_step,
            "steady_tolerance": CARTESIAN_TOLERANCE,
            "core_sigma": CORE_SIGMA,
            "target_position": list(target_position),
            "neighbour_position": list(neighbour_position),
            "core_separation": float(
                np.linalg.norm(np.subtract(neighbour_position, target_position))
            ),
            "target_core_rate": TWO_CORE_TARGET_RATE,
            "neighbour_core_rates": list(neighbour_rates),
            "amplitude_measurement": "paired velocity subtraction projected on the isolated target over radii 2-5",
            "depletion_measurement": "trilinear sample of the background deficit at the fixed physical target position",
        },
        "isolated_steady_state": isolated_record,
        "sweep": sweep,
        "depletion_range": [
            min(row["ambient_depletion_at_target"] for row in sweep),
            max(row["ambient_depletion_at_target"] for row in sweep),
        ],
    }
    if len(sweep) >= 2:
        result["unmatched_raw_modulation_fits"] = fit_modulation_families(sweep)
    return result


def apply_matched_depletion_grid(ladder: list[dict], samples: int = 8) -> list[float]:
    """Interpolate every rung onto one common physical depletion grid."""
    lower = max(rung["depletion_range"][0] for rung in ladder)
    upper = min(rung["depletion_range"][1] for rung in ladder)
    matched_depletion = np.geomspace(lower, upper, samples)
    for rung in ladder:
        raw_depletion = np.array(
            [row["ambient_depletion_at_target"] for row in rung["sweep"]]
        )
        raw_amplitude = np.array(
            [row["measured_incremental_amplitude"] for row in rung["sweep"]]
        )
        matched_amplitude = np.exp(
            np.interp(
                np.log(matched_depletion),
                np.log(raw_depletion),
                np.log(raw_amplitude),
            )
        )
        matched_sweep = [
            {
                "ambient_depletion_at_target": float(depletion),
                "measured_incremental_amplitude": float(amplitude),
            }
            for depletion, amplitude in zip(matched_depletion, matched_amplitude)
        ]
        rung["matched_fit_samples"] = matched_sweep
        rung["modulation_fits"] = fit_modulation_families(matched_sweep)
    return [float(value) for value in matched_depletion]


def parameter_convergence(ladder: list[dict]) -> dict:
    """Report successive shifts and a second-order continuum extrapolation."""
    spacings = np.array([rung["apparatus"]["spacing"] for rung in ladder])
    parameters = {
        name: np.array(
            [
                rung["modulation_fits"]["best_two_parameter_form"][name]
                for rung in ladder
            ]
        )
        for name in ("c", "beta")
    }
    shifts = {
        name: [float(value) for value in np.diff(values)]
        for name, values in parameters.items()
    }
    extrapolated = {}
    errors = {}
    fine_relative_shifts = {}
    for name, values in parameters.items():
        slope, intercept = np.polyfit(spacings**2, values, 1)
        del slope
        extrapolated[name] = float(intercept)
        errors[name] = float(
            max(abs(intercept - values[-1]), abs(values[-1] - values[-2]))
        )
        fine_relative_shifts[name] = float(
            abs(values[-1] - values[-2]) / max(abs(values[-1]), 1.0e-15)
        )
    first_shift_norm = float(
        np.hypot(
            shifts["c"][0] / parameters["c"][-1],
            shifts["beta"][0] / parameters["beta"][-1],
        )
    )
    second_shift_norm = float(
        np.hypot(
            shifts["c"][1] / parameters["c"][-1],
            shifts["beta"][1] / parameters["beta"][-1],
        )
    )
    shifts_shrink = {
        name: bool(abs(values[1]) < abs(values[0]))
        for name, values in shifts.items()
    }
    converged = bool(
        all(shifts_shrink.values())
        and second_shift_norm < first_shift_norm
        and max(fine_relative_shifts.values()) < 0.25
    )
    finest_fits = ladder[-1]["modulation_fits"]
    drifting_to_orb = bool(
        finest_fits["ORB_10157_screening_family"]["log_rmse"]
        <= finest_fits["best_two_parameter_form"]["log_rmse"]
    )
    if converged:
        verdict = "resolution_converged_distinct_shear_family"
    elif drifting_to_orb:
        verdict = "ORB_10751_family_was_a_discretization_artifact"
    else:
        verdict = "resolution_inconclusive_not_drifting_to_ORB_10157"
    return {
        "criterion": "the successive shifts of both c and beta and their normalized joint shift must shrink, with each finest-rung relative parameter shift below 0.25",
        "grid_sizes": [
            int(rung["apparatus"]["lattice"].split("^")[0]) for rung in ladder
        ],
        "spacings": [float(value) for value in spacings],
        "parameter_values": {
            name: [float(value) for value in values]
            for name, values in parameters.items()
        },
        "successive_rung_shifts": shifts,
        "normalized_shift_norms": [first_shift_norm, second_shift_norm],
        "fine_relative_parameter_shifts": fine_relative_shifts,
        "per_parameter_shifts_shrink": shifts_shrink,
        "normalized_shift_shrinks": bool(second_shift_norm < first_shift_norm),
        "converged": converged,
        "continuum_extrapolation": {
            "model": "p(h) = p(0) + k h^2 least-squares fit over all three rungs",
            "parameters": extrapolated,
            "conservative_errors": errors,
        },
        "drifting_to_ORB_10157": drifting_to_orb,
        "verdict": verdict,
    }


def separation_control(size: int, finest: dict, finest_fit: dict) -> dict:
    coefficient = finest_fit["c"]
    exponent = finest_fit["beta"]
    rows = []
    for separation in SEPARATION_CONTROL_VALUES:
        target_position = (-0.5 * separation, 0.0, 0.0)
        neighbour_position = (0.5 * separation, 0.0, 0.0)
        if separation == 8.0:
            measurement = finest
            row = next(
                candidate
                for candidate in finest["sweep"]
                if candidate["neighbour_core_rate"] == SEPARATION_CONTROL_RATE
            )
        else:
            measurement = run_two_core_sweep(
                size,
                (SEPARATION_CONTROL_RATE,),
                target_position,
                neighbour_position,
            )
            row = measurement["sweep"][0]
        depletion = row["ambient_depletion_at_target"]
        amplitude = row["measured_incremental_amplitude"]
        prediction = 1.0 / (1.0 + coefficient * depletion**exponent)
        rows.append(
            {
                "core_separation": separation,
                "target_core_rate": TWO_CORE_TARGET_RATE,
                "neighbour_core_rate": SEPARATION_CONTROL_RATE,
                "ambient_depletion_at_target": depletion,
                "measured_incremental_amplitude": amplitude,
                "finest_resolution_family_prediction": prediction,
                "log_residual": float(np.log(amplitude / prediction)),
                "isolated_steady_state": measurement["isolated_steady_state"],
                "background_steady_state": row["background_steady_state"],
                "paired_steady_state": row["paired_steady_state"],
            }
        )
    residuals = np.array([row["log_residual"] for row in rows])
    rms = float(np.sqrt(np.mean(residuals**2)))
    return {
        "criterion": "fixed-mass separation points follow the finest-rung A(D) family with log-RMSE below 0.08; otherwise the residuals quantify geometry dependence",
        "grid_size": size,
        "fixed_core_rates": [TWO_CORE_TARGET_RATE, SEPARATION_CONTROL_RATE],
        "measurements": rows,
        "log_rmse_against_A_of_D": rms,
        "maximum_absolute_log_residual": float(np.max(np.abs(residuals))),
        "consistent_with_A_of_D": bool(rms < 0.08),
    }


def two_core_adjudication() -> dict:
    ladder = [run_two_core_sweep(size) for size in TWO_CORE_GRID_SIZES]
    matched_depletion_grid = apply_matched_depletion_grid(ladder)
    convergence = parameter_convergence(ladder)
    finest = ladder[-1]
    finest_fit = finest["modulation_fits"]["best_two_parameter_form"]
    separation = separation_control(TWO_CORE_GRID_SIZES[-1], finest, finest_fit)
    half_step = run_two_core_sweep(
        TWO_CORE_GRID_SIZES[-1],
        (SEPARATION_CONTROL_RATE,),
        time_step=0.5 * CARTESIAN_TIME_STEP,
    )
    baseline_row = next(
        row
        for row in finest["sweep"]
        if row["neighbour_core_rate"] == SEPARATION_CONTROL_RATE
    )
    half_row = half_step["sweep"][0]
    step_halving = {
        "grid_size": TWO_CORE_GRID_SIZES[-1],
        "neighbour_core_rate": SEPARATION_CONTROL_RATE,
        "time_steps": [CARTESIAN_TIME_STEP, 0.5 * CARTESIAN_TIME_STEP],
        "depletions": [
            baseline_row["ambient_depletion_at_target"],
            half_row["ambient_depletion_at_target"],
        ],
        "amplitudes": [
            baseline_row["measured_incremental_amplitude"],
            half_row["measured_incremental_amplitude"],
        ],
        "relative_depletion_shift": float(
            abs(
                half_row["ambient_depletion_at_target"]
                - baseline_row["ambient_depletion_at_target"]
            )
            / baseline_row["ambient_depletion_at_target"]
        ),
        "relative_amplitude_shift": float(
            abs(
                half_row["measured_incremental_amplitude"]
                - baseline_row["measured_incremental_amplitude"]
            )
            / baseline_row["measured_incremental_amplitude"]
        ),
    }
    step_halving["passed"] = bool(
        max(
            step_halving["relative_depletion_shift"],
            step_halving["relative_amplitude_shift"],
        )
        < 0.01
    )
    return {
        "role": "predeclared resolution-convergence gate",
        "matched_physical_configuration": {
            "half_width": CARTESIAN_HALF_WIDTH,
            "core_separation_to_domain_width": 8.0 / (2.0 * CARTESIAN_HALF_WIDTH),
            "core_sigma": CORE_SIGMA,
            "target_core_rate": TWO_CORE_TARGET_RATE,
            "neighbour_core_rates": list(TWO_CORE_NEIGHBOUR_RATES),
            "note": "all physical lengths, source rates, and rules are identical across rungs; only cell spacing changes",
        },
        "resolution_ladder": ladder,
        "matched_fit_protocol": {
            "depletion_grid": matched_depletion_grid,
            "interpolation": "piecewise linear in log(D)-log(A) between directly measured sweep points",
            "range": [matched_depletion_grid[0], matched_depletion_grid[-1]],
        },
        "convergence": convergence,
        "range_check": {
            "previous_ceiling": 0.083,
            "measured_ranges_by_grid": [rung["depletion_range"] for rung in ladder],
            "finest_measured_ceiling": finest["depletion_range"][1],
            "extended_beyond_previous_ceiling": bool(
                finest["depletion_range"][1] > 0.083
            ),
            "strongest_cataloged_neighbour_rate": TWO_CORE_NEIGHBOUR_RATES[-1],
        },
        "separation_control": separation,
        "top_rung_step_halving": step_halving,
        "level_core_scope_note": "This task adjudicates the inherited flux-type core apparatus. The level-core closure raised in ORB-10932 is a different boundary model and is not tested here.",
    }


def experiment() -> tuple[dict, dict]:
    primary, _ = run_radial(
        PRIMARY_SHELLS, PRIMARY_TIME_STEP, PRIMARY_CORE_FLUX, "rest"
    )
    perturbed_primary, _ = run_radial(
        PRIMARY_SHELLS, PRIMARY_TIME_STEP, PRIMARY_CORE_FLUX, "perturbed"
    )
    resolution = [
        run_radial(shells, PRIMARY_TIME_STEP, PRIMARY_CORE_FLUX, "rest")[0]
        for shells in (128, 256, 512)
    ]
    time_steps = [
        run_radial(PRIMARY_SHELLS, dt, PRIMARY_CORE_FLUX, "rest")[0]
        for dt in (0.2, 0.1, 0.05)
    ]
    amplitude_sweep = []
    for core_flux in (0.003, 0.01, 0.03, 0.1, 0.3):
        rest, _ = run_radial(
            PRIMARY_SHELLS, PRIMARY_TIME_STEP, core_flux, "rest"
        )
        perturbed, _ = run_radial(
            PRIMARY_SHELLS, PRIMARY_TIME_STEP, core_flux, "perturbed", SEED + 1
        )
        amplitude_sweep.append(
            {
                "core_flux": core_flux,
                "amplitude": rest["fixed_half_power_amplitude"],
                "perturbed_amplitude": perturbed["fixed_half_power_amplitude"],
                "initial_condition_relative_difference": abs(
                    perturbed["fixed_half_power_amplitude"]
                    - rest["fixed_half_power_amplitude"]
                )
                / rest["fixed_half_power_amplitude"],
                "flow_exponent": rest["flow_fit"]["exponent"],
                "steady_state": rest["steady_state"],
            }
        )
    amplitudes = np.array([row["amplitude"] for row in amplitude_sweep])
    exponent_checks = [row["flow_fit"]["exponent"] for row in resolution + time_steps]
    numerical_exponent_half_range = 0.5 * (max(exponent_checks) - min(exponent_checks))
    initial_exponent_difference = abs(
        primary["flow_fit"]["exponent"] - perturbed_primary["flow_fit"]["exponent"]
    )
    combined_exponent_error = float(
        np.sqrt(
            primary["flow_fit"]["standard_error"] ** 2
            + numerical_exponent_half_range**2
            + initial_exponent_difference**2
        )
    )
    attractor = {
        "passed": bool(
            abs(primary["flow_fit"]["exponent"] + 0.5)
            <= 2.0 * combined_exponent_error
        ),
        "target_exponent": -0.5,
        "primary": primary,
        "second_initial_condition": perturbed_primary,
        "initial_condition_exponent_difference": initial_exponent_difference,
        "resolution_checks": resolution,
        "time_step_checks": time_steps,
        "numerical_exponent_half_range": numerical_exponent_half_range,
        "combined_exponent_error": combined_exponent_error,
        "criterion": "target -1/2 lies within two combined regression, seed, resolution, and timestep errors",
    }
    amplitude_gate = {
        "passed": bool(
            np.all(np.diff(amplitudes) > 0.0)
            and max(row["initial_condition_relative_difference"] for row in amplitude_sweep)
            < 1.0e-6
        ),
        "sweep": amplitude_sweep,
        "monotonic_increasing": bool(np.all(np.diff(amplitudes) > 0.0)),
        "maximum_initial_condition_relative_difference": float(
            max(row["initial_condition_relative_difference"] for row in amplitude_sweep)
        ),
        "criterion": "strictly increasing exterior half-power amplitude with <1e-6 seeded-initial-condition spread",
    }
    silence = silence_gate()
    superposition = two_core_adjudication()
    full_record = {
        "schema_version": 1,
        "task": "ORB-10755",
        "seed": SEED,
        "frozen_rule": {
            "consumption": "s = n * sqrt((3/2) * e_dev:e_dev), coefficient exactly 1",
            "radial_reduction": "s = n * abs(-du/dr + u/r) for inward speed u",
            "transport": "nearest-neighbour Fick flux; velocity is flux divided by face area and local density",
            "core": "fixed inward substrate flux at the inner boundary",
            "outer_boundary": "unit-density reservoir",
            "profile_policy": "no velocity profile is imposed",
        },
        "apparatus": {
            "radial_domain": [INNER_RADIUS, OUTER_RADIUS],
            "primary_shells": PRIMARY_SHELLS,
            "diffusivity": DIFFUSIVITY,
            "primary_time_step": PRIMARY_TIME_STEP,
            "fit_radius_range": list(FIT_RADIUS_RANGE),
            "steady_tolerance": STEADY_TOLERANCE,
        },
        "gates": {
            "attractor": attractor,
            "amplitude": amplitude_gate,
            "silence": silence,
        },
        "two_core_superposition_adjudication": superposition,
        "limitations": [
            "The radial finite-volume apparatus is a spherical reduction, not a full 3-D shell tessellation.",
            "Finite density depletion shifts the fitted exponent slightly below -1/2 at stronger core fluxes; the declared attractor comparison uses the weak-depletion primary run.",
            "The two-core resolution gate tests the inherited flux-type Gaussian core boundary model; it does not test the distinct level-core closure raised in ORB-10932.",
            "The deepest cataloged two-core point is the strongest stable predeclared sweep point, not evidence that still deeper steady branches cannot exist with a different solver.",
        ],
    }
    summary = {
        "schema_version": 1,
        "task": "ORB-10755",
        "run_record": f"runs/{RUN_RECORD}",
        "frozen_rule": full_record["frozen_rule"],
        "gate_results": {
            "attractor": {
                "passed": attractor["passed"],
                "exponent": primary["flow_fit"]["exponent"],
                "regression_standard_error": primary["flow_fit"]["standard_error"],
                "combined_numerical_error": attractor["combined_exponent_error"],
                "initial_condition_exponent_difference": initial_exponent_difference,
                "resolution_exponents": [
                    row["flow_fit"]["exponent"] for row in resolution
                ],
                "time_step_exponents": [
                    row["flow_fit"]["exponent"] for row in time_steps
                ],
            },
            "amplitude": {
                "passed": amplitude_gate["passed"],
                "monotonic_increasing": amplitude_gate["monotonic_increasing"],
                "core_fluxes": [row["core_flux"] for row in amplitude_sweep],
                "amplitudes": [row["amplitude"] for row in amplitude_sweep],
                "maximum_initial_condition_relative_difference": amplitude_gate[
                    "maximum_initial_condition_relative_difference"
                ],
            },
            "silence": silence,
        },
        "two_core_superposition": {
            "role": superposition["role"],
            "grid_sizes": list(TWO_CORE_GRID_SIZES),
            "depletion_ranges": superposition["range_check"][
                "measured_ranges_by_grid"
            ],
            "matched_fit_protocol": superposition["matched_fit_protocol"],
            "fits_by_grid": [
                {
                    "grid_size": size,
                    "modulation_fits": rung["modulation_fits"],
                }
                for size, rung in zip(
                    TWO_CORE_GRID_SIZES, superposition["resolution_ladder"]
                )
            ],
            "convergence": superposition["convergence"],
            "range_check": superposition["range_check"],
            "separation_control": {
                "consistent_with_A_of_D": superposition["separation_control"][
                    "consistent_with_A_of_D"
                ],
                "log_rmse_against_A_of_D": superposition[
                    "separation_control"
                ]["log_rmse_against_A_of_D"],
                "maximum_absolute_log_residual": superposition[
                    "separation_control"
                ]["maximum_absolute_log_residual"],
                "measurements": superposition["separation_control"][
                    "measurements"
                ],
            },
            "top_rung_step_halving": superposition["top_rung_step_halving"],
            "level_core_scope_note": superposition["level_core_scope_note"],
        },
        "verdict": {
            "apparatus_level": superposition["convergence"]["verdict"],
            "theory_reconciliation": "deferred to kepler; principia is intentionally untouched",
        },
    }
    return summary, full_record


def encoded(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    summary, full_record = experiment()
    if args.check_determinism:
        repeated_summary, repeated_record = experiment()
        if encoded(summary) != encoded(repeated_summary) or encoded(full_record) != encoded(
            repeated_record
        ):
            raise RuntimeError("determinism check failed")
    if not args.no_write:
        root = Path(__file__).parent
        runs = root / "runs"
        runs.mkdir(exist_ok=True)
        (root / "summary.json").write_bytes(encoded(summary))
        (runs / RUN_RECORD).write_bytes(encoded(full_record))
    digest = hashlib.sha256(encoded(full_record)).hexdigest()
    fit = summary["gate_results"]["attractor"]
    two_core = summary["two_core_superposition"]
    print(
        f"attractor p={fit['exponent']:.8f} +/- {fit['combined_numerical_error']:.2e}; "
        f"amplitude={summary['gate_results']['amplitude']['passed']}; "
        f"silence={summary['gate_results']['silence']['maximum_consumption_density']:.2e}; "
        f"two_core={two_core['convergence']['verdict']}; "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()
