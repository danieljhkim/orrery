"""Verify the linear mode content of the two-substance packet Hamiltonian.

The fixture independently constructs the Fourier-space linear operator and
evolves isolated Fourier perturbations with the packet simulator's actual
centered derivatives, chemical potentials, and kick-drift-kick update.  It
classifies longitudinal sound, frozen transverse currents, equal-speed
degeneracy, coupling-off controls, and locally stable mixed backgrounds.

Run from the repository root with:
    uv run lab/sims/two-substance-packet-linear-symbol/main.py --check-determinism
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "assets" / "results.json"
APPARATUS_PATH = ROOT / "lab/sims/two-substance-dynamical-packet/main.py"

ORRERY_SOURCE_COMMIT = "35a3d47ebc650212e8e7eb59b13752db093f1c28"
PRINCIPIA_SPEC_COMMIT = "38cd4a77d8f230f1fbe8a036c180f24919cacc0d"
APPARATUS_SHA256 = "bd3d02d5182446b94159e5eaf5d777da575f476d2c1d3e88a552517a5b881fe4"
SPEC_PATH = "theory/two-substance-vortex-vacuum/packet-hamiltonian-linearization.md"
DOMAIN_LENGTH = 2.0 * np.pi
BASE_DENSITY = 1.0
AMPLITUDE = 1.0e-6
PRIMARY_SIZE = 128
PRIMARY_DT = 0.01
DURATION = 2.0
SPATIAL_RUNGS = (64, 96, 128)
TEMPORAL_RUNGS = (0.04, 0.02, 0.01)
WAVEVECTORS = ((1, 0), (2, 0), (4, 0), (2, 2))
ANISOTROPY_PAIR = ((5, 0), (3, 4))

TOLERANCES = {
    "continuum_frequency_relative": 0.02,
    "matrix_time_frequency_relative": 0.002,
    "longitudinal_projector_minimum": 0.98,
    "transverse_normalized_frequency_maximum": 0.02,
    "transverse_curl_relative_drift": 1.0e-10,
    "vacuum_lambda_eigenvalue_relative": 1.0e-8,
    "background_squared_speed_relative": 0.02,
    "refinement_finest_frequency_relative": 0.002,
    "operator_eigenvalue_relative": 1.0e-11,
}

CLASSIFICATIONS = (
    "longitudinal-acoustic",
    "transverse-frozen",
    "unexpected-propagating-transverse",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    background_plus: float
    background_minus: float
    coupling: float
    c_plus: float
    c_minus: float


CASES = (
    Case("U0", 0.0, 0.0, 0.0, 0.65, 1.0),
    Case("U1", 0.0, 0.0, 0.0, 1.0, 1.0),
    Case("U2-unequal", 0.0, 0.0, 0.8, 0.65, 1.0),
    Case("U2-equal", 0.0, 0.0, 0.8, 1.0, 1.0),
    Case("B0-unequal", 0.0, 0.05, 0.8, 0.65, 1.0),
    Case("B0-equal", 0.0, 0.05, 0.8, 1.0, 1.0),
)


def derivative(values: np.ndarray, axis: int, dx: float) -> np.ndarray:
    """The actual packet apparatus's periodic centered derivative."""
    return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (
        2.0 * dx
    )


def chemical_potentials(
    n1: np.ndarray,
    n2: np.ndarray,
    case: Case,
) -> tuple[np.ndarray, np.ndarray]:
    """The exact nonlinear chemical potentials frozen by ORB-11219."""
    n_plus = n1 + n2 - 2.0 * BASE_DENSITY
    n_minus = n1 - n2
    common_linear = 0.5 * case.c_plus**2 * n_plus
    relative_linear = 0.5 * case.c_minus**2 * n_minus
    common_from_relative = 0.25 * case.coupling * n_minus**2
    relative_from_common = 0.5 * case.coupling * n_plus * n_minus
    return (
        common_linear
        + relative_linear
        + common_from_relative
        + relative_from_common,
        common_linear
        - relative_linear
        + common_from_relative
        - relative_from_common,
    )


def step(state: list[np.ndarray], dt: float, dx: float, case: Case) -> None:
    """One symmetric kick-drift-kick step, matching the packet apparatus."""
    n1, n2, j1x, j1y, j2x, j2y = state
    mu1, mu2 = chemical_potentials(n1, n2, case)
    j1x -= 0.5 * dt * derivative(mu1, 0, dx)
    j1y -= 0.5 * dt * derivative(mu1, 1, dx)
    j2x -= 0.5 * dt * derivative(mu2, 0, dx)
    j2y -= 0.5 * dt * derivative(mu2, 1, dx)
    n1 -= dt * (derivative(j1x, 0, dx) + derivative(j1y, 1, dx))
    n2 -= dt * (derivative(j2x, 0, dx) + derivative(j2y, 1, dx))
    mu1, mu2 = chemical_potentials(n1, n2, case)
    j1x -= 0.5 * dt * derivative(mu1, 0, dx)
    j1y -= 0.5 * dt * derivative(mu1, 1, dx)
    j2x -= 0.5 * dt * derivative(mu2, 0, dx)
    j2y -= 0.5 * dt * derivative(mu2, 1, dx)


def hessian(case: Case) -> np.ndarray:
    return np.array(
        [
            [case.c_plus**2, case.coupling * case.background_minus],
            [
                case.coupling * case.background_minus,
                case.c_minus**2 + case.coupling * case.background_plus,
            ],
        ],
        dtype=float,
    )


def physical_and_discrete_wavevector(
    modes: tuple[int, int], size: int
) -> tuple[np.ndarray, np.ndarray, float]:
    dx = DOMAIN_LENGTH / size
    physical = 2.0 * np.pi * np.asarray(modes, dtype=float) / DOMAIN_LENGTH
    discrete = np.sin(physical * dx) / dx
    return physical, discrete, dx


def linearized_operator(case: Case, modes: tuple[int, int], size: int) -> np.ndarray:
    """Construct a 6x6 species-basis operator independently of time stepping."""
    _, q, _ = physical_and_discrete_wavevector(modes, size)
    transform = np.array([[1.0, 1.0], [1.0, -1.0]])
    species_hessian = 0.5 * transform @ hessian(case) @ transform
    operator = np.zeros((6, 6), dtype=complex)
    operator[0, 2:4] = -1j * q
    operator[1, 4:6] = -1j * q
    for species in range(2):
        rows = slice(2 + 2 * species, 4 + 2 * species)
        operator[rows, 0] = -1j * q * species_hessian[species, 0]
        operator[rows, 1] = -1j * q * species_hessian[species, 1]
    return operator


def projector_content(vector: np.ndarray, q_hat: np.ndarray) -> tuple[float, float]:
    currents = (vector[2:4], vector[4:6])
    total = sum(float(np.vdot(current, current).real) for current in currents)
    if total <= np.finfo(float).tiny:
        return 0.0, 0.0
    longitudinal = sum(abs(np.vdot(q_hat, current)) ** 2 for current in currents)
    longitudinal_fraction = float(np.sqrt(longitudinal / total))
    return longitudinal_fraction, float(np.sqrt(max(0.0, 1.0 - longitudinal / total)))


def operator_record(case: Case, modes: tuple[int, int], size: int) -> dict[str, Any]:
    physical, q, dx = physical_and_discrete_wavevector(modes, size)
    matrix = linearized_operator(case, modes, size)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    speed_squared, _ = np.linalg.eigh(hessian(case))
    stable = bool(np.min(speed_squared) > 0.0)
    scale = max(float(np.sqrt(np.max(np.abs(speed_squared))) * np.linalg.norm(q)), 1.0)
    zero_indices = [
        index for index, value in enumerate(eigenvalues) if abs(value) <= 1.0e-10 * scale
    ]
    propagating_indices = [index for index in range(6) if index not in zero_indices]
    frequencies = sorted(abs(float(eigenvalues[index].imag)) for index in propagating_indices)
    expected = sorted(
        [float(np.sqrt(value) * np.linalg.norm(q)) for value in speed_squared for _ in range(2)]
    ) if stable else []
    operator_error = (
        max(
            abs(measured - predicted) / max(predicted, np.finfo(float).tiny)
            for measured, predicted in zip(frequencies, expected, strict=True)
        )
        if stable
        else None
    )
    longitudinal_contents = [
        projector_content(eigenvectors[:, index], q / np.linalg.norm(q))[0]
        for index in propagating_indices
    ]
    zero_transverse_contents = [
        projector_content(eigenvectors[:, index], q / np.linalg.norm(q))[1]
        for index in zero_indices
    ]
    continuum = sorted(
        [float(np.sqrt(value) * np.linalg.norm(physical)) for value in speed_squared]
    ) if stable else []
    positive_frequencies = [frequencies[index] for index in range(0, len(frequencies), 2)]
    continuum_errors = [
        abs(measured - predicted) / predicted
        for measured, predicted in zip(positive_frequencies, continuum, strict=True)
    ] if stable else []
    return {
        "case_id": case.case_id,
        "size": size,
        "dx": dx,
        "mode_indices": list(modes),
        "physical_wavevector": physical.tolist(),
        "centered_difference_wavevector": q.tolist(),
        "hessian": hessian(case).tolist(),
        "hessian_eigenvalues_squared_speed": speed_squared.tolist(),
        "locally_stable": stable,
        "propagating_root_count": len(propagating_indices),
        "zero_frequency_mode_count": len(zero_indices),
        "positive_frequencies": positive_frequencies,
        "maximum_operator_eigenvalue_relative_error": operator_error,
        "maximum_continuum_frequency_relative_error": max(continuum_errors, default=None),
        "minimum_propagating_longitudinal_projector": min(
            longitudinal_contents, default=None
        ),
        "minimum_zero_mode_transverse_projector": min(
            zero_transverse_contents, default=None
        ),
        "classification": "longitudinal-acoustic" if stable else "unstable-rejected",
    }


def grid_phase(size: int, modes: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    dx = DOMAIN_LENGTH / size
    axis = np.arange(size, dtype=float) * dx
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    phase = modes[0] * xx + modes[1] * yy
    return np.cos(phase), np.exp(-1j * phase), dx


def initialize_longitudinal(
    case: Case,
    modes: tuple[int, int],
    size: int,
    dt: float,
    branch: int,
) -> tuple[list[np.ndarray], np.ndarray, float, np.ndarray]:
    cosine, _, _ = grid_phase(size, modes)
    _, q, _ = physical_and_discrete_wavevector(modes, size)
    speed_squared, vectors = np.linalg.eigh(hessian(case))
    if np.min(speed_squared) <= 0.0:
        raise ValueError(f"case {case.case_id} is outside the local stability bound")
    mode_vector = vectors[:, branch]
    speed = float(np.sqrt(speed_squared[branch]))
    semidiscrete_frequency = speed * float(np.linalg.norm(q))
    half_step_factor = float(np.sqrt(1.0 - (dt * semidiscrete_frequency) ** 2 / 4.0))
    q_hat = q / np.linalg.norm(q)
    delta_plus = AMPLITUDE * mode_vector[0] * cosine
    delta_minus = AMPLITUDE * mode_vector[1] * cosine
    n_plus = case.background_plus + delta_plus
    n_minus = case.background_minus + delta_minus
    n1 = BASE_DENSITY + 0.5 * (n_plus + n_minus)
    n2 = BASE_DENSITY + 0.5 * (n_plus - n_minus)
    j_plus_x = speed * half_step_factor * delta_plus * q_hat[0]
    j_plus_y = speed * half_step_factor * delta_plus * q_hat[1]
    j_minus_x = speed * half_step_factor * delta_minus * q_hat[0]
    j_minus_y = speed * half_step_factor * delta_minus * q_hat[1]
    state = [
        n1.copy(),
        n2.copy(),
        0.5 * (j_plus_x + j_minus_x),
        0.5 * (j_plus_y + j_minus_y),
        0.5 * (j_plus_x - j_minus_x),
        0.5 * (j_plus_y - j_minus_y),
    ]
    return state, mode_vector, semidiscrete_frequency, q_hat


def fourier_coefficient(values: np.ndarray, phase_conjugate: np.ndarray) -> complex:
    return complex(2.0 * np.mean(values * phase_conjugate))


def hamiltonian_density(state: list[np.ndarray], case: Case) -> np.ndarray:
    n1, n2, j1x, j1y, j2x, j2y = state
    n_plus = n1 + n2 - 2.0 * BASE_DENSITY
    n_minus = n1 - n2
    return (
        0.5 * (j1x**2 + j1y**2 + j2x**2 + j2y**2)
        + 0.25 * case.c_plus**2 * n_plus**2
        + 0.25 * case.c_minus**2 * n_minus**2
        + 0.25 * case.coupling * n_plus * n_minus**2
    )


def evolve_longitudinal(
    case: Case,
    modes: tuple[int, int],
    branch: int,
    size: int = PRIMARY_SIZE,
    dt: float = PRIMARY_DT,
) -> dict[str, Any]:
    state, mode_vector, matrix_frequency, q_hat = initialize_longitudinal(
        case, modes, size, dt, branch
    )
    _, phase_conjugate, dx = grid_phase(size, modes)
    steps = int(round(DURATION / dt))
    times: list[float] = []
    phases: list[float] = []
    minimum_longitudinal = 1.0
    minimum_energy = float("inf")
    maximum_energy = -float("inf")

    def sample(index: int) -> None:
        nonlocal minimum_longitudinal, minimum_energy, maximum_energy
        n_plus = state[0] + state[1] - 2.0 * BASE_DENSITY - case.background_plus
        n_minus = state[0] - state[1] - case.background_minus
        density_coefficients = np.array(
            [
                fourier_coefficient(n_plus, phase_conjugate),
                fourier_coefficient(n_minus, phase_conjugate),
            ]
        )
        scalar = complex(np.vdot(mode_vector, density_coefficients))
        times.append(index * dt)
        phases.append(float(np.angle(scalar)))
        j1 = np.array(
            [
                fourier_coefficient(state[2], phase_conjugate),
                fourier_coefficient(state[3], phase_conjugate),
            ]
        )
        j2 = np.array(
            [
                fourier_coefficient(state[4], phase_conjugate),
                fourier_coefficient(state[5], phase_conjugate),
            ]
        )
        total = float(np.vdot(j1, j1).real + np.vdot(j2, j2).real)
        longitudinal = float(abs(np.vdot(q_hat, j1)) ** 2 + abs(np.vdot(q_hat, j2)) ** 2)
        if total > np.finfo(float).tiny:
            minimum_longitudinal = min(minimum_longitudinal, np.sqrt(longitudinal / total))
        energy = hamiltonian_density(state, case)
        minimum_energy = min(minimum_energy, float(np.min(energy)))
        maximum_energy = max(maximum_energy, float(np.max(energy)))

    sample(0)
    for index in range(1, steps + 1):
        step(state, dt, dx, case)
        sample(index)
    unwrapped = np.unwrap(np.asarray(phases))
    slope, intercept = np.polyfit(np.asarray(times), unwrapped, 1)
    measured_frequency = float(-slope)
    residual = unwrapped - (slope * np.asarray(times) + intercept)
    physical, _, _ = physical_and_discrete_wavevector(modes, size)
    speed_squared = np.linalg.eigvalsh(hessian(case))[branch]
    continuum_frequency = float(np.sqrt(speed_squared) * np.linalg.norm(physical))
    return {
        "case_id": case.case_id,
        "branch_index_ascending_speed": branch,
        "size": size,
        "dx": dx,
        "dt": dt,
        "duration": steps * dt,
        "mode_indices": list(modes),
        "matrix_semidiscrete_frequency": matrix_frequency,
        "time_evolution_frequency": measured_frequency,
        "matrix_time_frequency_relative_error": abs(measured_frequency - matrix_frequency)
        / matrix_frequency,
        "continuum_frequency": continuum_frequency,
        "continuum_frequency_relative_error": abs(measured_frequency - continuum_frequency)
        / continuum_frequency,
        "phase_fit_maximum_residual_radians": float(np.max(np.abs(residual))),
        "minimum_longitudinal_projector": minimum_longitudinal,
        "classification": "longitudinal-acoustic"
        if minimum_longitudinal >= TOLERANCES["longitudinal_projector_minimum"]
        else "unexpected-propagating-transverse",
        "sampled_hamiltonian_density": {"minimum": minimum_energy, "maximum": maximum_energy},
    }


def discrete_curl(jx: np.ndarray, jy: np.ndarray, dx: float) -> np.ndarray:
    return derivative(jy, 0, dx) - derivative(jx, 1, dx)


def evolve_transverse(case: Case, size: int = PRIMARY_SIZE, dt: float = PRIMARY_DT) -> dict[str, Any]:
    modes = (2, 0)
    cosine, phase_conjugate, dx = grid_phase(size, modes)
    n_plus = np.full_like(cosine, case.background_plus)
    n_minus = np.full_like(cosine, case.background_minus)
    n1 = BASE_DENSITY + 0.5 * (n_plus + n_minus)
    n2 = BASE_DENSITY + 0.5 * (n_plus - n_minus)
    zeros = np.zeros_like(cosine)
    state = [
        n1.copy(),
        n2.copy(),
        zeros.copy(),
        AMPLITUDE * cosine,
        zeros.copy(),
        0.7 * AMPLITUDE * cosine,
    ]
    initial_curls = [discrete_curl(state[2], state[3], dx), discrete_curl(state[4], state[5], dx)]
    initial_coefficients = [fourier_coefficient(state[3], phase_conjugate), fourier_coefficient(state[5], phase_conjugate)]
    steps = int(round(DURATION / dt))
    for _ in range(steps):
        step(state, dt, dx, case)
    final_curls = [discrete_curl(state[2], state[3], dx), discrete_curl(state[4], state[5], dx)]
    curl_drifts = [
        float(np.linalg.norm(final - initial) / np.linalg.norm(initial))
        for initial, final in zip(initial_curls, final_curls, strict=True)
    ]
    final_coefficients = [fourier_coefficient(state[3], phase_conjugate), fourier_coefficient(state[5], phase_conjugate)]
    normalized_frequency_bounds = [
        abs(float(np.angle(final / initial)))
        / max(steps * dt * max(case.c_plus, case.c_minus) * 2.0, np.finfo(float).tiny)
        for initial, final in zip(initial_coefficients, final_coefficients, strict=True)
    ]
    maximum_bound = max(normalized_frequency_bounds)
    classification = (
        "transverse-frozen"
        if maximum_bound <= TOLERANCES["transverse_normalized_frequency_maximum"]
        and max(curl_drifts) <= TOLERANCES["transverse_curl_relative_drift"]
        else "unexpected-propagating-transverse"
    )
    return {
        "case_id": case.case_id,
        "seed": "axial k=(2,0), j1y=amplitude*cos(kx), j2y=0.7*amplitude*cos(kx), densities uniform",
        "size": size,
        "dx": dx,
        "dt": dt,
        "duration": steps * dt,
        "maximum_normalized_frequency_bound": maximum_bound,
        "curl_relative_drift_each_species": curl_drifts,
        "classification": classification,
    }


def case_dict(case: Case) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "background_n_plus": case.background_plus,
        "background_n_minus": case.background_minus,
        "coupling_lambda": case.coupling,
        "c_plus": case.c_plus,
        "c_minus": case.c_minus,
    }


def anisotropy_records() -> list[dict[str, Any]]:
    records = []
    for size in SPATIAL_RUNGS:
        ratios = []
        for modes in ANISOTROPY_PAIR:
            physical, discrete, dx = physical_and_discrete_wavevector(modes, size)
            ratios.append(float(np.linalg.norm(discrete) / np.linalg.norm(physical)))
        records.append(
            {
                "size": size,
                "dx": dx,
                "equal_magnitude_mode_pair": [list(value) for value in ANISOTROPY_PAIR],
                "normalized_phase_speeds": ratios,
                "orientation_anisotropy_absolute": abs(ratios[0] - ratios[1]),
                "continuum_dispersion_errors": [abs(value - 1.0) for value in ratios],
            }
        )
    return records


def refinement_records() -> dict[str, Any]:
    probes = ((CASES[0], (4, 0)), (CASES[1], (2, 2)))
    spatial = []
    temporal = []
    for case, modes in probes:
        for branch in (0, 1):
            spatial_rows = [
                evolve_longitudinal(case, modes, branch, size=size, dt=PRIMARY_DT)
                for size in SPATIAL_RUNGS
            ]
            temporal_rows = [
                evolve_longitudinal(case, modes, branch, size=PRIMARY_SIZE, dt=dt)
                for dt in TEMPORAL_RUNGS
            ]
            spatial.append(
                {
                    "case_id": case.case_id,
                    "branch": branch,
                    "mode_indices": list(modes),
                    "rungs": [
                        {
                            "size": row["size"],
                            "dx": row["dx"],
                            "matrix_time_frequency_relative_error": row[
                                "matrix_time_frequency_relative_error"
                            ],
                            "continuum_frequency_relative_error": row[
                                "continuum_frequency_relative_error"
                            ],
                        }
                        for row in spatial_rows
                    ],
                }
            )
            temporal.append(
                {
                    "case_id": case.case_id,
                    "branch": branch,
                    "mode_indices": list(modes),
                    "rungs": [
                        {
                            "dt": row["dt"],
                            "matrix_time_frequency_relative_error": row[
                                "matrix_time_frequency_relative_error"
                            ],
                        }
                        for row in temporal_rows
                    ],
                }
            )
    return {"spatial": spatial, "temporal": temporal}


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def run_fixture() -> dict[str, Any]:
    actual_sha = hashlib.sha256(APPARATUS_PATH.read_bytes()).hexdigest()
    operator = [
        operator_record(case, modes, PRIMARY_SIZE)
        for case in CASES
        for modes in WAVEVECTORS
    ]
    longitudinal = []
    for case in CASES:
        selected_modes = WAVEVECTORS if case.case_id in {"U0", "U1"} else ((2, 0),)
        for modes in selected_modes:
            for branch in (0, 1):
                longitudinal.append(evolve_longitudinal(case, modes, branch))
    transverse_cases = (CASES[0], CASES[1], CASES[2], CASES[3])
    transverse = [evolve_transverse(case) for case in transverse_cases]
    refinements = refinement_records()
    anisotropy = anisotropy_records()

    operator_errors = [row["maximum_operator_eigenvalue_relative_error"] for row in operator]
    vacuum_operator = [row for row in operator if row["case_id"] in {"U0", "U1"}]
    vacuum_continuum_errors = [
        row["maximum_continuum_frequency_relative_error"] for row in vacuum_operator
    ]
    lambda_pairs = (("U0", "U2-unequal"), ("U1", "U2-equal"))
    lambda_differences = []
    for left_id, right_id in lambda_pairs:
        for modes in WAVEVECTORS:
            left = next(row for row in operator if row["case_id"] == left_id and row["mode_indices"] == list(modes))
            right = next(row for row in operator if row["case_id"] == right_id and row["mode_indices"] == list(modes))
            differences = [
                abs(a - b) / max(abs(a), np.finfo(float).tiny)
                for a, b in zip(left["positive_frequencies"], right["positive_frequencies"], strict=True)
            ]
            lambda_differences.append(max(differences))
    background_rows = [row for row in longitudinal if row["case_id"].startswith("B0")]
    background_errors = []
    for row in background_rows:
        k_norm = float(np.linalg.norm(row["mode_indices"]))
        measured_squared_speed = (row["time_evolution_frequency"] / k_norm) ** 2
        expected_squared_speed = np.linalg.eigvalsh(
            hessian(next(case for case in CASES if case.case_id == row["case_id"]))
        )[row["branch_index_ascending_speed"]]
        background_errors.append(abs(measured_squared_speed - expected_squared_speed) / expected_squared_speed)

    spatial_finest = [
        record["rungs"][-1]["matrix_time_frequency_relative_error"]
        for record in refinements["spatial"]
    ]
    spatial_finest_continuum = [
        record["rungs"][-1]["continuum_frequency_relative_error"]
        for record in refinements["spatial"]
    ]
    temporal_finest = [
        record["rungs"][-1]["matrix_time_frequency_relative_error"]
        for record in refinements["temporal"]
    ]
    spatial_converges = all(
        all(
            left["continuum_frequency_relative_error"]
            > right["continuum_frequency_relative_error"]
            for left, right in zip(
                record["rungs"][:-1], record["rungs"][1:], strict=True
            )
        )
        for record in refinements["spatial"]
    )
    temporal_converges = all(
        all(
            left["matrix_time_frequency_relative_error"]
            > right["matrix_time_frequency_relative_error"]
            for left, right in zip(
                record["rungs"][:-1], record["rungs"][1:], strict=True
            )
        )
        for record in refinements["temporal"]
    )
    checks = [
        gate("apparatus_source_sha256", actual_sha == APPARATUS_SHA256, actual_sha),
        gate(
            "operator_dispersion_and_mode_count",
            max(operator_errors) <= TOLERANCES["operator_eigenvalue_relative"]
            and all(row["propagating_root_count"] == 4 for row in operator)
            and all(row["zero_frequency_mode_count"] == 2 for row in operator),
            {"maximum_relative_error": max(operator_errors), "field_count": 6},
        ),
        gate(
            "vacuum_continuum_frequency",
            max(vacuum_continuum_errors) <= TOLERANCES["continuum_frequency_relative"],
            {"maximum_relative_error": max(vacuum_continuum_errors)},
        ),
        gate(
            "matrix_time_frequency_agreement",
            max(row["matrix_time_frequency_relative_error"] for row in longitudinal)
            <= TOLERANCES["matrix_time_frequency_relative"],
            {
                "maximum_matrix_time_relative_error": max(
                    row["matrix_time_frequency_relative_error"] for row in longitudinal
                ),
            },
        ),
        gate(
            "propagating_mode_polarization",
            min(row["minimum_longitudinal_projector"] for row in longitudinal)
            >= TOLERANCES["longitudinal_projector_minimum"],
            {
                "minimum_longitudinal_projector": min(
                    row["minimum_longitudinal_projector"] for row in longitudinal
                )
            },
        ),
        gate(
            "equal_speed_degeneracy_remains_longitudinal",
            all(
                row["classification"] == "longitudinal-acoustic"
                for row in longitudinal
                if row["case_id"] in {"U1", "U2-equal", "B0-equal"}
            ),
            {"equal_speed_vacuum_cases": ["U1", "U2-equal"]},
        ),
        gate(
            "transverse_frozen",
            all(row["classification"] == "transverse-frozen" for row in transverse),
            {
                "maximum_frequency_bound": max(
                    row["maximum_normalized_frequency_bound"] for row in transverse
                ),
                "maximum_curl_relative_drift": max(
                    max(row["curl_relative_drift_each_species"]) for row in transverse
                ),
            },
        ),
        gate(
            "vacuum_lambda_control",
            max(lambda_differences) <= TOLERANCES["vacuum_lambda_eigenvalue_relative"],
            {"maximum_relative_difference": max(lambda_differences)},
        ),
        gate(
            "small_background_hessian",
            max(background_errors) <= TOLERANCES["background_squared_speed_relative"],
            {"maximum_squared_speed_relative_error": max(background_errors)},
        ),
        gate(
            "spatial_and_temporal_refinement",
            len(SPATIAL_RUNGS) >= 3
            and len(TEMPORAL_RUNGS) >= 3
            and spatial_converges
            and temporal_converges
            and max(spatial_finest + temporal_finest)
            <= TOLERANCES["refinement_finest_frequency_relative"]
            and max(spatial_finest_continuum)
            <= TOLERANCES["continuum_frequency_relative"],
            {
                "spatial_rungs": list(SPATIAL_RUNGS),
                "temporal_rungs": list(TEMPORAL_RUNGS),
                "spatial_continuum_errors_decrease": spatial_converges,
                "temporal_matrix_errors_decrease": temporal_converges,
                "maximum_finest_continuum_relative_error": max(
                    spatial_finest_continuum
                ),
                "maximum_finest_matrix_time_relative_error": max(
                    spatial_finest + temporal_finest
                ),
            },
        ),
    ]

    unstable_case = Case("X0-stability-guard", -2.0, 2.0, 0.8, 0.65, 1.0)
    unstable_eigenvalues = np.linalg.eigvalsh(hessian(unstable_case))
    stability_guard = {
        "case": case_dict(unstable_case),
        "hessian_eigenvalues_squared_speed": unstable_eigenvalues.tolist(),
        "classification": "rejected-local-instability"
        if np.min(unstable_eigenvalues) <= 0.0
        else "admissible",
        "time_evolution_attempted": False,
    }
    checks.append(
        gate(
            "unstable_parameter_guard",
            stability_guard["classification"] == "rejected-local-instability",
            stability_guard,
        )
    )

    numerical_validation_names = {
        "apparatus_source_sha256",
        "operator_dispersion_and_mode_count",
        "vacuum_continuum_frequency",
        "matrix_time_frequency_agreement",
        "small_background_hessian",
        "spatial_and_temporal_refinement",
        "unstable_parameter_guard",
    }
    validation_passed = all(
        check["passed"] for check in checks if check["name"] in numerical_validation_names
    )
    unexpected_transverse = any(
        row["classification"] == "unexpected-propagating-transverse" for row in transverse
    ) or any(
        row["classification"] == "unexpected-propagating-transverse"
        for row in longitudinal
        if row["case_id"] in {"U1", "U2-equal"}
    )
    if not validation_passed:
        verdict = "unresolved"
    elif unexpected_transverse:
        verdict = "supported"
    else:
        verdict = "refuted"

    return {
        "schema_version": 1,
        "fixture": "two-substance-packet-linear-symbol",
        "research_question": (
            "Does equal-speed tuning of the existing two-density/gradient-flux Hamiltonian "
            "produce a Maxwell-like propagating transverse sector?"
        ),
        "hypothesis_verdict": verdict,
        "verdict_scope": (
            "Only the frozen scalar-density packet Hamiltonian is classified. This is not "
            "a no-go theorem for other emergent-gauge objects and makes no nature claim."
        ),
        "verdict_rule": {
            "unresolved": "any numerical/source/refinement gate fails",
            "supported": "numerical validation passes and an equal-speed arm is classified unexpected-propagating-transverse",
            "refuted": "all validation gates pass and transverse seeds remain frozen while propagating modes are longitudinal",
        },
        "source": {
            "orrery_commit_before_fixture": ORRERY_SOURCE_COMMIT,
            "principia_orb_11219_commit": PRINCIPIA_SPEC_COMMIT,
            "principia_specification": SPEC_PATH,
            "packet_apparatus_path": str(APPARATUS_PATH.relative_to(ROOT)),
            "packet_apparatus_sha256": APPARATUS_SHA256,
            "orbit_tasks": ["ORB-11219", "ORB-11220"],
        },
        "equations": {
            "hamiltonian_density": "(|j1|^2+|j2|^2)/2 + c_plus^2*n_plus^2/4 + c_minus^2*n_minus^2/4 + lambda*n_plus*n_minus^2/4",
            "continuity": "partial_t n_i + div(j_i) = 0",
            "flux": "partial_t j_i + grad(mu_i) = 0, mu_i=delta H/delta n_i",
            "linear_mode_hessian": "[[c_plus^2, lambda*nbar_minus], [lambda*nbar_minus, c_minus^2+lambda*nbar_plus]]",
            "discrete_derivative_symbol": "q_a=sin(k_a*dx)/dx",
            "global_boundedness": "unbounded for every nonzero lambda; local Hessian stability does not cure the negative quartic remainder",
        },
        "solver": {
            "time_integrator": "symmetric kick-drift-kick",
            "spatial_operator": "periodic second-order centered first derivative",
            "matrix_solver": "numpy.linalg.eig on independently assembled 6x6 species-basis Fourier operator",
            "mode_solver": "numpy.linalg.eigh on the 2x2 common/relative Hessian",
            "frequency_estimator": "least-squares slope of unwrapped seeded-mode Fourier phase",
            "dimensions": 2,
            "field_count": 6,
            "algebraic_constraint_count": 0,
        },
        "parameters": {
            "domain_length": DOMAIN_LENGTH,
            "amplitude": AMPLITUDE,
            "duration": DURATION,
            "primary_size": PRIMARY_SIZE,
            "primary_dt": PRIMARY_DT,
            "spatial_refinement_sizes": list(SPATIAL_RUNGS),
            "temporal_refinement_dt": list(TEMPORAL_RUNGS),
            "wavevector_mode_indices": [list(value) for value in WAVEVECTORS],
            "cases": [case_dict(case) for case in CASES],
            "seeds": {
                "random_seed": None,
                "reason": "No RNG is used; exact named cosine Fourier modes are deterministic.",
                "longitudinal": "each ascending Hessian eigenvector with discrete traveling-wave flux",
                "transverse": "T0 axial j_y-only currents in both species",
            },
        },
        "thresholds": TOLERANCES,
        "classification_vocabulary": list(CLASSIFICATIONS),
        "mode_count": {
            "propagating_roots": 4,
            "propagating_positive_frequency_branches": 2,
            "propagating_polarization_relative_to_wavevector": "longitudinal",
            "zero_frequency_modes": 2,
            "zero_mode_polarization_relative_to_wavevector": "transverse",
            "gauss_or_gauge_constraints": 0,
            "interpretation": "Equal scalar sound speeds are branch degeneracy, not Maxwell polarization evidence.",
        },
        "operator_results": operator,
        "time_evolution_results": {"longitudinal": longitudinal, "transverse": transverse},
        "refinement": refinements,
        "lattice_anisotropy": {
            "interpretation": "orientation-dependent centered-difference dispersion, reported separately from continuum mode count",
            "records": anisotropy,
        },
        "stability_guard": stability_guard,
        "gates": checks,
        "validation_passed": validation_passed,
        "all_predeclared_expected_outcomes_observed": all(
            check["passed"] for check in checks
        ),
        "rerun_commands": [
            "uv run lab/sims/two-substance-packet-linear-symbol/main.py --check-determinism",
            "uv run lab/sims/two-substance-packet-linear-symbol/main.py --output /tmp/two-substance-packet-linear-symbol.json",
            "python3 lab/tools/build-gallery.py",
        ],
    }


def encoded(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-determinism", action="store_true")
    args = parser.parse_args()
    result = run_fixture()
    payload = encoded(result)
    if args.check_determinism:
        repeated = encoded(run_fixture())
        if payload != repeated:
            raise RuntimeError("fixture results were not byte-reproducible")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(
        f"{args.output}: verdict={result['hypothesis_verdict']} "
        f"validation_passed={result['validation_passed']}"
    )


if __name__ == "__main__":
    main()
