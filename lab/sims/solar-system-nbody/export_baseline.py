"""Export the untuned Newtonian solar-system baseline for ORB-10093.

Question: how far does a nine-body Newtonian point-mass integration drift from
Tycho's JPL Horizons planetary-system ephemerides over the exact 2016--2025
exchange grid when frame, origin, epochs, and initial states are matched?

The Sun and eight planetary-system barycenters are integrated in an inertial,
zero-momentum frame with SciPy DOP853 (rtol=1e-12, atol=1e-14, maximum step two
days). Output is sampled at JD 2457388.5 + 10*k TDB, k=0..365, and transformed
to heliocentric ICRF by subtracting the Sun's instantaneous state. Planetary-
system GMs are JPL DE440 values. Known omissions are relativity, asteroids,
solar oblateness/mass loss, and individually resolved moons; residuals are
reported without tuning the apparatus to shrink them.

Usage: uv run lab/sims/solar-system-nbody/export_baseline.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.integrate import solve_ivp

AU_KM = 149_597_870.7
SECONDS_PER_DAY = 86_400.0
START_JD_TDB = 2_457_388.5
EPOCH_COUNT = 366
EPOCH_STEP_DAYS = 10.0
DEFAULT_MAX_STEP_DAYS = 2.0
RTOL = 1.0e-12
ATOL = 1.0e-14
TARGETS = ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")

# JPL DE440 astrodynamic parameters, km^3/s^2. Earth system is Earth + Moon.
GM_KM3_S2 = {
    "sun": 132_712_440_041.279419,
    "mercury": 22_031.868551,
    "venus": 324_858.592000,
    "earth": 398_600.435507 + 4_902.800118,
    "mars": 42_828.375816,
    "jupiter": 126_712_764.100000,
    "saturn": 37_940_584.841800,
    "uranus": 5_794_556.400000,
    "neptune": 6_836_527.100580,
}


def default_ephemeris_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "astrolabe/data/processed/ephemeris"


def gm_au3_d2() -> np.ndarray:
    conversion = SECONDS_PER_DAY**2 / AU_KM**3
    return np.array([GM_KM3_S2[name] for name in ("sun", *TARGETS)]) * conversion


def load_initial_conditions(ephemeris_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs = START_JD_TDB + EPOCH_STEP_DAYS * np.arange(EPOCH_COUNT)
    relative_positions, relative_velocities = [], []
    for target in TARGETS:
        table = pq.read_table(ephemeris_dir / f"{target}_2016_2026.parquet")
        actual = table["epoch_jd_tdb"].to_numpy(zero_copy_only=False).astype(float)
        if len(actual) != EPOCH_COUNT or not np.array_equal(actual, epochs):
            raise ValueError(f"{target}: ephemeris is not on the exact 366-epoch grid")
        relative_positions.append([float(table[name][0].as_py()) for name in ("x_au", "y_au", "z_au")])
        relative_velocities.append([float(table[name][0].as_py()) for name in ("vx_au_d", "vy_au_d", "vz_au_d")])

    gm = gm_au3_d2()
    relative_positions = np.asarray(relative_positions)
    relative_velocities = np.asarray(relative_velocities)
    total_gm = float(np.sum(gm))
    sun_position = -np.sum(gm[1:, None] * relative_positions, axis=0) / total_gm
    sun_velocity = -np.sum(gm[1:, None] * relative_velocities, axis=0) / total_gm
    positions = np.vstack((sun_position, relative_positions + sun_position))
    velocities = np.vstack((sun_velocity, relative_velocities + sun_velocity))
    return epochs, positions, velocities


def derivative(_time_days: float, state: np.ndarray, gm: np.ndarray) -> np.ndarray:
    count = len(gm)
    positions = state[: count * 3].reshape(count, 3)
    velocities = state[count * 3 :].reshape(count, 3)
    displacement = positions[None, :, :] - positions[:, None, :]
    distance_squared = np.sum(displacement**2, axis=2)
    np.fill_diagonal(distance_squared, np.inf)
    acceleration = np.sum(
        displacement * distance_squared[:, :, None] ** -1.5 * gm[None, :, None], axis=1
    )
    return np.concatenate((velocities.ravel(), acceleration.ravel()))


def total_specific_energy(positions: np.ndarray, velocities: np.ndarray, gm: np.ndarray) -> float:
    energy = 0.5 * float(np.sum(gm[:, None] * velocities**2))
    for left in range(len(gm)):
        for right in range(left + 1, len(gm)):
            energy -= float(gm[left] * gm[right] / np.linalg.norm(positions[right] - positions[left]))
    return energy


def integrate(ephemeris_dir: Path, max_step_days: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    epochs, positions, velocities = load_initial_conditions(ephemeris_dir)
    gm = gm_au3_d2()
    times = epochs - epochs[0]
    solution = solve_ivp(
        derivative, (float(times[0]), float(times[-1])),
        np.concatenate((positions.ravel(), velocities.ravel())), args=(gm,),
        method="DOP853", t_eval=times, rtol=RTOL, atol=ATOL, max_step=max_step_days,
    )
    if not solution.success or solution.y.shape[1] != EPOCH_COUNT:
        raise RuntimeError(f"integration failed: {solution.message}")
    count = len(gm)
    integrated_positions = solution.y[: count * 3].T.reshape(EPOCH_COUNT, count, 3)
    integrated_velocities = solution.y[count * 3 :].T.reshape(EPOCH_COUNT, count, 3)
    relative_positions = integrated_positions[:, 1:, :] - integrated_positions[:, :1, :]
    relative_velocities = integrated_velocities[:, 1:, :] - integrated_velocities[:, :1, :]
    energies = np.array([
        total_specific_energy(integrated_positions[i], integrated_velocities[i], gm)
        for i in range(EPOCH_COUNT)
    ])
    return epochs, relative_positions, relative_velocities, {
        "function_evaluations": int(solution.nfev),
        "max_relative_energy_drift": float(np.max(np.abs((energies - energies[0]) / energies[0]))),
    }


def residual_summary(ephemeris_dir: Path, epochs: np.ndarray, positions: np.ndarray) -> dict:
    summaries = {}
    for target_index, target in enumerate(TARGETS):
        table = pq.read_table(ephemeris_dir / f"{target}_2016_2026.parquet")
        actual_epochs = table["epoch_jd_tdb"].to_numpy(zero_copy_only=False).astype(float)
        if not np.array_equal(actual_epochs, epochs):
            raise ValueError(f"{target}: epoch mismatch during residual validation")
        expected = np.column_stack([
            table[name].to_numpy(zero_copy_only=False).astype(float)
            for name in ("x_au", "y_au", "z_au")
        ])
        dr = np.linalg.norm(positions[:, target_index, :] - expected, axis=1)
        summaries[target] = {
            "n_epochs": len(dr),
            "rms_dr_au": float(np.sqrt(np.mean(dr**2))),
            "max_dr_au": float(np.max(dr)),
            "max_dr_epoch_jd_tdb": float(epochs[int(np.argmax(dr))]),
        }
    return summaries


def export(output_dir: Path, ephemeris_dir: Path, max_step_days: float) -> None:
    epochs, positions, velocities, diagnostics = integrate(ephemeris_dir, max_step_days)
    _, refined_positions, _, refined_diagnostics = integrate(
        ephemeris_dir, max_step_days / 2.0
    )
    convergence = {
        "refined_max_step_days": max_step_days / 2.0,
        "max_position_shift_au": float(np.max(np.abs(positions - refined_positions))),
        "refined_function_evaluations": refined_diagnostics["function_evaluations"],
    }
    table = pa.table({
        "target": np.repeat(np.array(TARGETS), EPOCH_COUNT),
        "epoch_jd_tdb": np.tile(epochs, len(TARGETS)),
        "x_au": positions.transpose(1, 0, 2).reshape(-1, 3)[:, 0],
        "y_au": positions.transpose(1, 0, 2).reshape(-1, 3)[:, 1],
        "z_au": positions.transpose(1, 0, 2).reshape(-1, 3)[:, 2],
        "vx_au_d": velocities.transpose(1, 0, 2).reshape(-1, 3)[:, 0],
        "vy_au_d": velocities.transpose(1, 0, 2).reshape(-1, 3)[:, 1],
        "vz_au_d": velocities.transpose(1, 0, 2).reshape(-1, 3)[:, 2],
    })
    summaries = residual_summary(ephemeris_dir, epochs, positions)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "newtonian_2016_2026.parquet"
    pq.write_table(table, parquet_path, compression="zstd")
    metadata = {
        "task": "ORB-10093",
        "question": "How far does the untuned Newtonian nine-body baseline drift from matched Horizons ephemerides?",
        "frame": "heliocentric ICRF (inertial integration; instantaneous Sun state subtracted at output)",
        "epochs": {"timescale": "TDB", "start_jd": START_JD_TDB, "step_days": EPOCH_STEP_DAYS, "count": EPOCH_COUNT},
        "integrator": {"method": "DOP853", "rtol": RTOL, "atol": ATOL, "max_step_days": max_step_days, **diagnostics, "convergence": convergence},
        "gm_source": "JPL DE440 astrodynamic parameters (https://ssd.jpl.nasa.gov/astro_par.html)",
        "gm_km3_s2": GM_KM3_S2,
        "initial_conditions": "first epoch of Astrolabe ephemeris/<target>_2016_2026, heliocentric ICRF",
        "omissions": ["relativistic corrections", "asteroids", "solar oblateness and mass loss", "individually resolved moons"],
        "residuals_baseline_minus_horizons": summaries,
        "output": {"path": "baseline/newtonian_2016_2026.parquet", "rows": len(table), "targets": list(TARGETS)},
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for target, summary in summaries.items():
        print(f"{target}: rms={summary['rms_dr_au']:.6g} AU; max={summary['max_dr_au']:.6g} AU")
    print(f"energy drift: {diagnostics['max_relative_energy_drift']:.3e}")
    print(f"max 2d-vs-1d position shift: {convergence['max_position_shift_au']:.3e} AU")
    print(f"wrote {parquet_path} ({len(table)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ephemeris-dir", type=Path, default=default_ephemeris_dir())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "baseline")
    parser.add_argument("--max-step-days", type=float, default=DEFAULT_MAX_STEP_DAYS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export(arguments.output_dir, arguments.ephemeris_dir, arguments.max_step_days)
