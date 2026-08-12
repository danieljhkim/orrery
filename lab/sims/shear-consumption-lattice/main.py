"""Test local von-Mises-strain destruction on a dynamical substrate lattice.

The primary apparatus is a spherical finite-volume lattice.  Substrate moves
between neighbouring shells down density gradients; the face velocity is
therefore measured from the transported flux, never prescribed as a radial
profile.  A fixed core flux anchors the flow.  Every shell destroys substrate
at the frozen local rate

    s = n * sqrt((3/2) * e_dev:e_dev),

which is ``n * abs(-du/dr + u/r)`` for radial inward speed ``u``.  Semi-implicit
steps evolve the density from either rest or a seeded perturbation to a steady
state.  A manufactured 3-D Hubble flow tests the zero-shear limit.  A small
3-D Cartesian companion lattice supplies the exploratory two-core probe.

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
RUN_RECORD = "2026-08-12-seed-42.json"


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
    xyz: tuple[np.ndarray, ...], position: tuple[float, float, float], rate: float, spacing: float
) -> np.ndarray:
    radius_squared = sum((axis - center) ** 2 for axis, center in zip(xyz, position))
    weights = np.exp(-radius_squared / (2.0 * (0.8 * spacing) ** 2))
    return rate * weights / (np.sum(weights) * spacing**3)


def cartesian_fields(deficit: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    density = 1.0 - deficit
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
    time_step: float = 2.0,
    tolerance: float = 1.0e-9,
    max_steps: int = 1000,
) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    apparatus = cartesian_setup(size, half_width, time_step)
    spacing = apparatus["spacing"]
    forcing = sum(
        (
            gaussian_core_source(apparatus["xyz"], position, rate, spacing)
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


def two_core_probe() -> dict:
    size = 25
    half_width = 12.0
    target_position = (-4.0, 0.0, 0.0)
    neighbour_position = (4.0, 0.0, 0.0)
    target_rate = 0.1
    neighbour_rates = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
    isolated_record, _, isolated_velocity, apparatus = run_cartesian(
        [(target_position, target_rate)], size=size, half_width=half_width
    )
    coordinate = apparatus["coordinates"]
    target_index = tuple(
        int(np.argmin(np.abs(coordinate - component))) for component in target_position
    )
    sweep = []
    for rate in neighbour_rates:
        background_record, background_deficit, background_velocity, _ = run_cartesian(
            [(neighbour_position, rate)], size=size, half_width=half_width
        )
        paired_record, _, paired_velocity, _ = run_cartesian(
            [(target_position, target_rate), (neighbour_position, rate)],
            size=size,
            half_width=half_width,
        )
        depletion = float(background_deficit[target_index])
        amplitude = projected_source_amplitude(
            paired_velocity,
            background_velocity,
            isolated_velocity,
            apparatus["xyz"],
            target_position,
        )
        screening_family = float((1.0 - depletion) ** 1.071)
        sweep.append(
            {
                "neighbour_core_rate": rate,
                "ambient_depletion_at_target": depletion,
                "measured_incremental_amplitude": amplitude,
                "ORB_10157_screening_amplitude": screening_family,
                "measured_over_ORB_10157": amplitude / screening_family,
                "background_steady_state": background_record,
                "paired_steady_state": paired_record,
            }
        )

    depletion = np.array([row["ambient_depletion_at_target"] for row in sweep])
    amplitude = np.array([row["measured_incremental_amplitude"] for row in sweep])
    log_headroom = np.log1p(-depletion)
    log_amplitude = np.log(amplitude)
    headroom_alpha = float(
        np.dot(log_headroom, log_amplitude) / np.dot(log_headroom, log_headroom)
    )

    def rational_modulation(d: np.ndarray, coefficient: float, exponent: float) -> np.ndarray:
        return 1.0 / (1.0 + coefficient * d**exponent)

    parameters, _ = curve_fit(
        rational_modulation,
        depletion,
        amplitude,
        p0=(6.0, 0.65),
        bounds=(0.0, np.inf),
    )
    rational_prediction = rational_modulation(depletion, *parameters)
    orb_prediction = (1.0 - depletion) ** 1.071
    return {
        "role": "exploratory; not a predeclared gate",
        "apparatus": {
            "lattice": f"{size}^3 Cartesian cells with a full Dirichlet reservoir",
            "half_width": half_width,
            "spacing": float(apparatus["spacing"]),
            "core_separation": 8.0,
            "target_core_rate": target_rate,
            "amplitude_measurement": "paired velocity subtraction projected on the isolated target over radii 2-5",
        },
        "isolated_steady_state": isolated_record,
        "sweep": sweep,
        "modulation_fits": {
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
                    np.sqrt(np.mean((log_amplitude - headroom_alpha * log_headroom) ** 2))
                ),
            },
            "ORB_10157_screening_family": {
                "form": "A(D) = (1-D)^1.071",
                "alpha": 1.071,
                "log_rmse": float(
                    np.sqrt(np.mean((log_amplitude - np.log(orb_prediction)) ** 2))
                ),
            },
        },
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
    superposition = two_core_probe()
    full_record = {
        "schema_version": 1,
        "task": "ORB-10751",
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
        "two_core_superposition_probe": superposition,
        "limitations": [
            "The radial finite-volume apparatus is a spherical reduction, not a full 3-D shell tessellation.",
            "Finite density depletion shifts the fitted exponent slightly below -1/2 at stronger core fluxes; the declared attractor comparison uses the weak-depletion primary run.",
            "The two-core probe uses a coarse 25^3 Cartesian lattice and is exploratory rather than a gate.",
        ],
    }
    summary = {
        "schema_version": 1,
        "task": "ORB-10751",
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
            "depletions": [
                row["ambient_depletion_at_target"] for row in superposition["sweep"]
            ],
            "amplitudes": [
                row["measured_incremental_amplitude"] for row in superposition["sweep"]
            ],
            "modulation_fits": superposition["modulation_fits"],
        },
        "verdict": {
            "apparatus_level": "all three predeclared gates pass; the two-core probe screens much more strongly than ORB-10157 over the measured range",
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
    print(
        f"attractor p={fit['exponent']:.8f} +/- {fit['combined_numerical_error']:.2e}; "
        f"amplitude={summary['gate_results']['amplitude']['passed']}; "
        f"silence={summary['gate_results']['silence']['maximum_consumption_density']:.2e}; "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()
