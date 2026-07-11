"""Test the fitted galactic scarcity gradient in the solar-system apparatus.

Question: under the multiplicative reading, does the unchanged ORB-10082
scarcity factor produce planet-position deviations larger than the measured
Newtonian-omission floors over ORB-10093's exact 2016--2026 protocol?

Newtonian and scarcity systems are integrated concurrently from the same
heliocentric ICRF states.  For each body, the full Newtonian heliocentric
acceleration is multiplied by q(r_gal)/q(R0).  At AU scales this ratio is
evaluated as the numerically exact local form

    exp[(beta F(R0) / R0^2) (r_gal - R0)],

whose omitted second-order term is below 1e-15 in ln(q) even for Neptune.  The
fitted beta=5.25 kpc and mass-profile F are not retuned.  The screened reading
is the explicit q=1 control and therefore has identically zero signature.

Usage: uv run lab/sims/solar-system-nbody/integrate_scarcity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.integrate import solve_ivp

from export_baseline import (
    ATOL,
    DEFAULT_MAX_STEP_DAYS,
    EPOCH_COUNT,
    EPOCH_STEP_DAYS,
    RTOL,
    START_JD_TDB,
    TARGETS,
    gm_au3_d2,
    load_initial_conditions,
)


AU_PER_KPC = 206_264_806.24709636
BETA_KPC = 5.25
DEFAULT_R0_KPC = 8.25
ALT_R0_KPC = 8.122
DISK_SCALE_KPC = 2.6
BULGE_FRACTION = 0.2
BULGE_SCALE_KPC = 0.7

# Conventional J2000/ICRF direction from the Sun toward the Galactic center.
GC_RA_DEG = 266.4051
GC_DEC_DEG = -28.936175


def default_ephemeris_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "astrolabe/data/processed/ephemeris"


def default_derived_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "astrolabe/data/processed/derived"


def enclosed_fraction(radius_kpc: float | np.ndarray) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=float)
    bulge = BULGE_FRACTION * radius**2 / (radius + BULGE_SCALE_KPC) ** 2
    x = radius / DISK_SCALE_KPC
    disk = (1.0 - BULGE_FRACTION) * (1.0 - (1.0 + x) * np.exp(-x))
    return bulge + disk


def galactic_outward_axis() -> np.ndarray:
    ra = np.deg2rad(GC_RA_DEG)
    dec = np.deg2rad(GC_DEC_DEG)
    toward_center = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    return -toward_center


def relative_acceleration(relative_positions: np.ndarray, gm: np.ndarray) -> np.ndarray:
    total_gm = float(np.sum(gm))
    sun_position = -np.sum(gm[1:, None] * relative_positions, axis=0) / total_gm
    positions = np.vstack((sun_position, relative_positions + sun_position))
    displacement = positions[None, :, :] - positions[:, None, :]
    distance_squared = np.sum(displacement**2, axis=2)
    np.fill_diagonal(distance_squared, np.inf)
    acceleration = np.sum(
        displacement * distance_squared[:, :, None] ** -1.5 * gm[None, :, None], axis=1
    )
    return acceleration[1:] - acceleration[:1]


def scarcity_factor(relative_positions: np.ndarray, axis: np.ndarray, r0_kpc: float) -> np.ndarray:
    galactocentric = r0_kpc * axis[None, :] + relative_positions / AU_PER_KPC
    radius = np.linalg.norm(galactocentric, axis=1)
    logarithmic_gradient_per_kpc = BETA_KPC * float(enclosed_fraction(r0_kpc)) / r0_kpc**2
    return np.exp(logarithmic_gradient_per_kpc * (radius - r0_kpc))


def unpack_system(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = len(TARGETS) * 3
    return state[:size].reshape(len(TARGETS), 3), state[size:].reshape(len(TARGETS), 3)


def derivative_pair(_time_days: float, state: np.ndarray, gm: np.ndarray, axis: np.ndarray, r0_kpc: float) -> np.ndarray:
    system_size = len(TARGETS) * 6
    newton_positions, newton_velocities = unpack_system(state[:system_size])
    scarcity_positions, scarcity_velocities = unpack_system(state[system_size:])
    newton_acceleration = relative_acceleration(newton_positions, gm)
    scarcity_newtonian_acceleration = relative_acceleration(scarcity_positions, gm)
    factor = scarcity_factor(scarcity_positions, axis, r0_kpc)
    scarcity_acceleration = scarcity_newtonian_acceleration * factor[:, None]
    return np.concatenate(
        (
            newton_velocities.ravel(),
            newton_acceleration.ravel(),
            scarcity_velocities.ravel(),
            scarcity_acceleration.ravel(),
        )
    )


def integrate_pair(ephemeris_dir: Path, axis: np.ndarray, r0_kpc: float, max_step_days: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    epochs, barycentric_positions, barycentric_velocities = load_initial_conditions(ephemeris_dir)
    relative_positions = barycentric_positions[1:] - barycentric_positions[:1]
    relative_velocities = barycentric_velocities[1:] - barycentric_velocities[:1]
    initial = np.concatenate((relative_positions.ravel(), relative_velocities.ravel()))
    paired_initial = np.concatenate((initial, initial))
    times = epochs - epochs[0]
    solution = solve_ivp(
        derivative_pair,
        (float(times[0]), float(times[-1])),
        paired_initial,
        args=(gm_au3_d2(), axis, r0_kpc),
        method="DOP853",
        t_eval=times,
        rtol=RTOL,
        atol=ATOL,
        max_step=max_step_days,
    )
    if not solution.success or solution.y.shape[1] != EPOCH_COUNT:
        raise RuntimeError(f"integration failed: {solution.message}")
    system_size = len(TARGETS) * 6
    position_size = len(TARGETS) * 3
    newton = solution.y[:position_size].T.reshape(EPOCH_COUNT, len(TARGETS), 3)
    scarcity = solution.y[system_size : system_size + position_size].T.reshape(EPOCH_COUNT, len(TARGETS), 3)
    return epochs, newton, scarcity, {"function_evaluations": int(solution.nfev), "max_step_days": max_step_days}


def signature_summary(newton: np.ndarray, scarcity: np.ndarray) -> tuple[dict, np.ndarray]:
    displacement = scarcity - newton
    distance = np.linalg.norm(displacement, axis=2)
    summaries = {}
    for index, target in enumerate(TARGETS):
        summaries[target] = {
            "rms_dr_au": float(np.sqrt(np.mean(distance[:, index] ** 2))),
            "max_dr_au": float(np.max(distance[:, index])),
            "final_dr_au": float(distance[-1, index]),
        }
    return summaries, displacement


def load_floors(derived_dir: Path) -> dict:
    floors = {}
    for target in TARGETS:
        path = derived_dir / f"{target}_newtonian_residuals.parquet"
        table = pq.read_table(path, columns=["dr_au"])
        values = table["dr_au"].to_numpy(zero_copy_only=False).astype(float)
        floors[target] = {
            "dataset": f"derived/{target}_newtonian_residuals",
            "rms_dr_au": float(np.sqrt(np.mean(values**2))),
            "max_dr_au": float(np.max(values)),
        }
    return floors


def compare_with_floors(signature: dict, floors: dict) -> dict:
    rows = {}
    for target in TARGETS:
        predicted = signature[target]
        floor = floors[target]
        rows[target] = {
            "predicted_rms_dr_au": predicted["rms_dr_au"],
            "floor_rms_dr_au": floor["rms_dr_au"],
            "rms_ratio_to_floor": predicted["rms_dr_au"] / floor["rms_dr_au"],
            "rms_above_floor": predicted["rms_dr_au"] > floor["rms_dr_au"],
            "predicted_max_dr_au": predicted["max_dr_au"],
            "floor_max_dr_au": floor["max_dr_au"],
            "max_ratio_to_floor": predicted["max_dr_au"] / floor["max_dr_au"],
            "max_above_floor": predicted["max_dr_au"] > floor["max_dr_au"],
        }
    return rows


def load_exported_baseline(path: Path, epochs: np.ndarray) -> np.ndarray:
    table = pq.read_table(path)
    output = np.empty((len(epochs), len(TARGETS), 3), dtype=float)
    targets = table["target"].to_numpy(zero_copy_only=False).astype(str)
    table_epochs = table["epoch_jd_tdb"].to_numpy(zero_copy_only=False).astype(float)
    for index, target in enumerate(TARGETS):
        mask = targets == target
        if not np.array_equal(table_epochs[mask], epochs):
            raise ValueError(f"{target}: exported baseline epoch mismatch")
        output[:, index, :] = np.column_stack(
            [table[name].to_numpy(zero_copy_only=False).astype(float)[mask] for name in ("x_au", "y_au", "z_au")]
        )
    return output


def export_signature(path: Path, epochs: np.ndarray, displacement: np.ndarray) -> None:
    flat = displacement.transpose(1, 0, 2).reshape(-1, 3)
    table = pa.table(
        {
            "target": np.repeat(np.array(TARGETS), len(epochs)),
            "epoch_jd_tdb": np.tile(epochs, len(TARGETS)),
            "dx_au": flat[:, 0],
            "dy_au": flat[:, 1],
            "dz_au": flat[:, 2],
            "dr_au": np.linalg.norm(flat, axis=1),
        }
    )
    pq.write_table(table, path, compression="zstd")


def make_plot(path: Path, comparison: dict) -> None:
    x = np.arange(len(TARGETS))
    predicted = np.array([comparison[target]["predicted_rms_dr_au"] for target in TARGETS])
    floors = np.array([comparison[target]["floor_rms_dr_au"] for target in TARGETS])
    fig, axis = plt.subplots(figsize=(10, 5.5))
    width = 0.38
    axis.bar(x - width / 2, predicted, width, label="multiplicative scarcity − Newtonian")
    axis.bar(x + width / 2, floors, width, label="Newtonian-omission floor")
    axis.set_yscale("log")
    axis.set_xticks(x, TARGETS, rotation=25)
    axis.set_ylabel("RMS position deviation (AU)")
    axis.set_title("ORB-10097: fitted scarcity gradient against measured floors")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ephemeris-dir", type=Path, default=default_ephemeris_dir())
    parser.add_argument("--derived-dir", type=Path, default=default_derived_dir())
    parser.add_argument("--baseline", type=Path, default=Path(__file__).resolve().parent / "baseline/newtonian_2016_2026.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "scarcity")
    parser.add_argument("--max-step-days", type=float, default=DEFAULT_MAX_STEP_DAYS)
    parser.add_argument("--quick", action="store_true", help="Run only the primary orientation (smoke testing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_axis = galactic_outward_axis()
    axes = {
        "galactic-outward": primary_axis,
        "+icrf-x": np.array([1.0, 0.0, 0.0]),
        "-icrf-x": np.array([-1.0, 0.0, 0.0]),
        "+icrf-y": np.array([0.0, 1.0, 0.0]),
        "-icrf-y": np.array([0.0, -1.0, 0.0]),
        "+icrf-z": np.array([0.0, 0.0, 1.0]),
        "-icrf-z": np.array([0.0, 0.0, -1.0]),
    }
    if args.quick:
        axes = {"galactic-outward": primary_axis}
    orientation_results, primary_payload = {}, None
    for name, axis in axes.items():
        epochs, newton, scarcity, diagnostics = integrate_pair(args.ephemeris_dir, axis, DEFAULT_R0_KPC, args.max_step_days)
        summary, displacement = signature_summary(newton, scarcity)
        orientation_results[name] = {"axis_icrf": [float(x) for x in axis], "per_planet": summary, "integrator": diagnostics}
        if name == "galactic-outward":
            primary_payload = (epochs, newton, scarcity, displacement, summary)
    assert primary_payload is not None
    epochs, newton, scarcity, displacement, primary = primary_payload
    floors = load_floors(args.derived_dir)
    comparison = compare_with_floors(primary, floors)
    exported_baseline = load_exported_baseline(args.baseline, epochs)
    baseline_crosscheck = float(np.max(np.abs(newton - exported_baseline)))
    r0_sensitivity = {}
    if not args.quick:
        _, alt_newton, alt_scarcity, alt_diagnostics = integrate_pair(args.ephemeris_dir, primary_axis, ALT_R0_KPC, args.max_step_days)
        alt_summary, _ = signature_summary(alt_newton, alt_scarcity)
        r0_sensitivity = {
            "R0_8.25_kpc": primary,
            "R0_8.122_kpc": alt_summary,
            "integrator_8.122": alt_diagnostics,
        }
        _, refined_newton, refined_scarcity, refined_diagnostics = integrate_pair(args.ephemeris_dir, primary_axis, DEFAULT_R0_KPC, args.max_step_days / 2.0)
        refined_displacement = refined_scarcity - refined_newton
        convergence = {
            "refined_max_step_days": args.max_step_days / 2.0,
            "max_signature_coordinate_change_au": float(np.max(np.abs(refined_displacement - displacement))),
            "refined_integrator": refined_diagnostics,
        }
    else:
        convergence = {"not_run": "quick mode"}
    export_signature(args.output_dir / "multiplicative_signature.parquet", epochs, displacement)
    make_plot(args.output_dir / "signature-vs-floor.png", comparison)
    orientation_envelope = {}
    for target in TARGETS:
        rms_values = [entry["per_planet"][target]["rms_dr_au"] for entry in orientation_results.values()]
        max_values = [entry["per_planet"][target]["max_dr_au"] for entry in orientation_results.values()]
        orientation_envelope[target] = {
            "rms_min_au": float(min(rms_values)),
            "rms_max_au": float(max(rms_values)),
            "max_min_au": float(min(max_values)),
            "max_max_au": float(max(max_values)),
        }
    results = {
        "task": "ORB-10097",
        "source_tasks": ["ORB-10093", "ORB-10094", "ORB-10096", "ORB-10082", "ORB-10077"],
        "question": "Does the unchanged fitted scarcity gradient exceed the measured Newtonian-omission floor under the multiplicative solar-field reading?",
        "protocol": {
            "frame": "heliocentric ICRF; relative equations integrated directly",
            "epochs": {"timescale": "TDB", "start_jd": START_JD_TDB, "step_days": EPOCH_STEP_DAYS, "count": EPOCH_COUNT},
            "integrator": {"method": "DOP853", "rtol": RTOL, "atol": ATOL, "max_step_days": args.max_step_days},
            "initial_conditions": "identical ORB-10093 first-epoch Horizons states",
            "comparison": "concurrently integrated multiplicative scarcity minus pure Newtonian",
        },
        "apparatus": {
            "beta_kpc": BETA_KPC,
            "mass_profile": {"disk_scale_kpc": DISK_SCALE_KPC, "bulge_fraction": BULGE_FRACTION, "bulge_scale_kpc": BULGE_SCALE_KPC, "F_at_R0_8.25": float(enclosed_fraction(DEFAULT_R0_KPC))},
            "R0_kpc": DEFAULT_R0_KPC,
            "multiplicative_equation": "a_helio = a_Newton,helio * exp[(beta F(R0)/R0^2)(r_gal-R0)]",
            "parameter_policy": "Imported unchanged from ORB-10077/10082; no fit or retuning was performed.",
            "primary_orientation": {"definition": "Galactocentric outward axis, opposite the conventional ICRF Galactic-center direction", "galactic_center_ra_deg": GC_RA_DEG, "galactic_center_dec_deg": GC_DEC_DEG, "axis_icrf": [float(x) for x in primary_axis]},
            "local_approximation_limit": "Second-order correction to ln(q) is below 1e-15 across the planetary system.",
        },
        "primary_multiplicative_signature": primary,
        "measured_newtonian_omission_floors": floors,
        "comparison_with_floors": comparison,
        "screened_control": {"q_ratio": 1.0, "rms_dr_au": 0.0, "max_dr_au": 0.0, "statement": "Local screening leaves the ORB-10093 Newtonian equations unchanged, so the scarcity-minus-Newtonian signature is identically zero by construction."},
        "orientation_sensitivity": {"runs": orientation_results, "per_planet_envelope": orientation_envelope},
        "R0_sensitivity": r0_sensitivity,
        "numerical_checks": {"concurrent_newtonian_vs_ORB_10093_export_max_coordinate_au": baseline_crosscheck, "step_halving": convergence},
        "limitations": [
            "The multiplicative rule is a deliberately tested interpretation, not a derived local field equation.",
            "The fixed galactocentric axis neglects the Sun's Galactic motion over ten years, whose fractional directional change is negligible for this test.",
            "The Newtonian-omission residual is an empirical precision floor for this apparatus, not a full observational covariance model.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for target in TARGETS:
        row = comparison[target]
        label = "ABOVE" if row["rms_above_floor"] else "below"
        print(f"{target}: rms={row['predicted_rms_dr_au']:.6g} AU; floor={row['floor_rms_dr_au']:.6g} AU; {label} ({row['rms_ratio_to_floor']:.3g}x)")
    print(f"concurrent Newtonian vs ORB-10093 max coordinate: {baseline_crosscheck:.3e} AU")


if __name__ == "__main__":
    main()
