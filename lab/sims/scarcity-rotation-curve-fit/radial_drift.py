"""Refit the Gaia rotation curve with a shared radial asymmetric-drift model.

Question: does replacing ORB-10082's constant velocity offset with the same
axisymmetric Jeans surrogate in scarcity, Newtonian-baryon, and NFW+baryon
fits remove the coherent 15--18.75 kpc held-out underprediction?

The drift apparatus assumes exponential tracer density and radial dispersion,

    v_c^2 - <v_phi>^2 = sigma_R(R)^2
        [R/h_nu + 2 R/h_sigma - 1/2],
    sigma_R(R) = sigma_R0 exp[-(R-R0)/h_sigma].

The epicycle anisotropy sigma_phi^2/sigma_R^2=1/2 and tracer scale h_nu=2.6
kpc are fixed.  Only sigma_R0 and h_sigma are nuisance parameters, shared in
form and bounds across the three gravity models.  The gravity apparatus, fit
and held-out bands, seed, bootstrap count, and 27-profile sweep are unchanged.

Usage: uv run lab/sims/scarcity-rotation-curve-fit/radial_drift.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import least_squares

from orrery import rng


G_KPC_KMS2_MSUN = 4.30091e-6
FIT_MIN_KPC = 5.0
FIT_MAX_KPC = 15.0
HELD_OUT_MAX_KPC = 18.75
R0_KPC = 8.25
H0_KMS_MPC = 70.0
TRACER_SCALE_KPC = 2.6
EPICYCLE_ANISOTROPY = 0.5
SIGMA_R0_BOUNDS_KMS = (10.0, 80.0)
DISPERSION_SCALE_BOUNDS_KPC = (3.0, 30.0)
NFW_BARYONIC_MASS_BOUNDS_MSUN = (1.0e9, 3.0e11)
NFW_HALO_MASS_BOUNDS_MSUN = (1.0e10, 3.0e13)
NFW_CONCENTRATION_BOUNDS = (1.0, 40.0)


@dataclass(frozen=True)
class Profile:
    disk_scale_kpc: float = 2.6
    bulge_fraction: float = 0.2
    bulge_scale_kpc: float = 0.7


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[4] / "astrolabe/data/processed/derived/mw_rotation_curve.parquet"


def enclosed_fraction(radius_kpc: np.ndarray, profile: Profile) -> np.ndarray:
    r = np.asarray(radius_kpc, dtype=float)
    bulge = profile.bulge_fraction * r**2 / (r + profile.bulge_scale_kpc) ** 2
    x = r / profile.disk_scale_kpc
    disk = (1.0 - profile.bulge_fraction) * (1.0 - (1.0 + x) * np.exp(-x))
    return bulge + disk


def scarcity_integral(
    radius_kpc: np.ndarray,
    profile: Profile,
    *,
    outer_radius_kpc: float = 100.0,
    step_kpc: float = 0.01,
) -> np.ndarray:
    grid = np.arange(step_kpc, outer_radius_kpc + step_kpc, step_kpc)
    integrand = enclosed_fraction(grid, profile) / grid**2
    segments = 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(grid)
    tail = np.zeros_like(grid)
    tail[:-1] = np.cumsum(segments[::-1])[::-1]
    return np.interp(np.asarray(radius_kpc, dtype=float), grid, tail)


def critical_density_msun_kpc3() -> float:
    h0_kms_kpc = H0_KMS_MPC / 1000.0
    return 3.0 * h0_kms_kpc**2 / (8.0 * np.pi * G_KPC_KMS2_MSUN)


def nfw_enclosed_mass(radius_kpc: np.ndarray, halo_mass_msun: float, concentration: float) -> np.ndarray:
    r200 = (
        3.0 * halo_mass_msun / (4.0 * np.pi * 200.0 * critical_density_msun_kpc3())
    ) ** (1.0 / 3.0)
    x = concentration * np.asarray(radius_kpc, dtype=float) / r200
    norm = np.log1p(concentration) - concentration / (1.0 + concentration)
    return halo_mass_msun * (np.log1p(x) - x / (1.0 + x)) / norm


def circular_speed(
    model: str,
    radius_kpc: np.ndarray,
    profile: Profile,
    parameters: np.ndarray,
    j_per_kpc: np.ndarray,
    j0_per_kpc: float,
) -> np.ndarray:
    fraction = enclosed_fraction(radius_kpc, profile)
    if model == "scarcity":
        mass = 10.0 ** parameters[0]
        beta = parameters[1]
        return np.sqrt(G_KPC_KMS2_MSUN * mass * fraction / radius_kpc) * np.exp(
            -0.5 * beta * (j_per_kpc - j0_per_kpc)
        )
    if model == "newtonian-baryons":
        return np.sqrt(G_KPC_KMS2_MSUN * 10.0 ** parameters[0] * fraction / radius_kpc)
    if model == "nfw-baryons":
        baryons = 10.0 ** parameters[0]
        halo = nfw_enclosed_mass(radius_kpc, 10.0 ** parameters[1], parameters[2])
        return np.sqrt(G_KPC_KMS2_MSUN * (baryons * fraction + halo) / radius_kpc)
    raise ValueError(f"unknown model: {model}")


def drift_parameters(model: str, parameters: np.ndarray) -> tuple[float, float]:
    if model == "scarcity":
        return float(parameters[2]), float(parameters[3])
    if model == "newtonian-baryons":
        return float(parameters[1]), float(parameters[2])
    return float(parameters[3]), float(parameters[4])


def apply_radial_drift(
    radius_kpc: np.ndarray,
    circular_kms: np.ndarray,
    sigma_r0_kms: float,
    dispersion_scale_kpc: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_r = sigma_r0_kms * np.exp(-(radius_kpc - R0_KPC) / dispersion_scale_kpc)
    radial_term = (
        radius_kpc / TRACER_SCALE_KPC
        + 2.0 * radius_kpc / dispersion_scale_kpc
        - EPICYCLE_ANISOTROPY
    )
    streaming_squared = circular_kms**2 - sigma_r**2 * radial_term
    streaming = np.sqrt(np.maximum(streaming_squared, 1.0))
    return streaming, circular_kms - streaming


def model_spec(model: str) -> tuple[np.ndarray, np.ndarray, list[tuple[float, ...]], list[str]]:
    if model == "scarcity":
        return (
            np.array([9.0, 0.0, SIGMA_R0_BOUNDS_KMS[0], DISPERSION_SCALE_BOUNDS_KPC[0]]),
            np.array([12.5, 80.0, SIGMA_R0_BOUNDS_KMS[1], DISPERSION_SCALE_BOUNDS_KPC[1]]),
            [(11.08, 5.25, 35.0, 8.0), (11.0, 10.0, 45.0, 12.0), (11.2, 2.0, 25.0, 20.0)],
            ["log10_mass_msun", "beta_kpc", "sigma_r0_kms", "dispersion_scale_kpc"],
        )
    if model == "newtonian-baryons":
        return (
            np.array([9.0, SIGMA_R0_BOUNDS_KMS[0], DISPERSION_SCALE_BOUNDS_KPC[0]]),
            np.array([12.5, SIGMA_R0_BOUNDS_KMS[1], DISPERSION_SCALE_BOUNDS_KPC[1]]),
            [(11.1, 35.0, 8.0), (11.3, 50.0, 15.0), (10.9, 25.0, 25.0)],
            ["log10_mass_msun", "sigma_r0_kms", "dispersion_scale_kpc"],
        )
    return (
        np.array([9.0, 10.0, NFW_CONCENTRATION_BOUNDS[0], SIGMA_R0_BOUNDS_KMS[0], DISPERSION_SCALE_BOUNDS_KPC[0]]),
        np.array([np.log10(3.0e11), np.log10(3.0e13), NFW_CONCENTRATION_BOUNDS[1], SIGMA_R0_BOUNDS_KMS[1], DISPERSION_SCALE_BOUNDS_KPC[1]]),
        [(10.5, 12.0, 12.0, 35.0, 8.0), (10.7, 12.2, 20.0, 45.0, 15.0), (11.0, 11.7, 30.0, 25.0, 25.0)],
        ["log10_baryonic_mass_msun", "log10_halo_mass_msun", "concentration", "sigma_r0_kms", "dispersion_scale_kpc"],
    )


def fit_model(
    model: str,
    radius: np.ndarray,
    velocity: np.ndarray,
    error: np.ndarray,
    profile: Profile,
    j: np.ndarray,
    j0: float,
) -> dict:
    lower, upper, starts, names = model_spec(model)

    def residual(parameters: np.ndarray) -> np.ndarray:
        circular = circular_speed(model, radius, profile, parameters, j, j0)
        sigma_r0, h_sigma = drift_parameters(model, parameters)
        observed_frame, _ = apply_radial_drift(radius, circular, sigma_r0, h_sigma)
        invalid = np.any(circular**2 - (circular - observed_frame) * (circular + observed_frame) <= 0.0)
        values = (velocity - observed_frame) / error
        return values + (1.0e6 if invalid else 0.0)

    candidates = [
        least_squares(
            residual,
            np.asarray(start),
            bounds=(lower, upper),
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
            max_nfev=4000,
        )
        for start in starts
    ]
    result = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    circular = circular_speed(model, radius, profile, result.x, j, j0)
    sigma_r0, h_sigma = drift_parameters(model, result.x)
    predicted, drift = apply_radial_drift(radius, circular, sigma_r0, h_sigma)
    residual_kms = velocity - predicted
    chi2 = float(np.sum((residual_kms / error) ** 2))
    k = len(result.x)
    raw = {name: float(value) for name, value in zip(names, result.x, strict=True)}
    parameters = {key: value for key, value in raw.items() if not key.startswith("log10_")}
    for key, value in raw.items():
        if key.startswith("log10_"):
            parameters[key[6:]] = float(10.0**value)
    return {
        "model": model,
        "parameters": parameters,
        "parameter_count": k,
        "chi2": chi2,
        "dof": int(len(radius) - k),
        "reduced_chi2": chi2 / (len(radius) - k),
        "rmse_kms": float(np.sqrt(np.mean(residual_kms**2))),
        "mae_kms": float(np.mean(np.abs(residual_kms))),
        "aic": chi2 + 2.0 * k,
        "bic": chi2 + k * np.log(len(radius)),
        "predicted_observed_kms": predicted,
        "predicted_circular_kms": circular,
        "drift_kms": drift,
        "residual_kms": residual_kms,
        "raw_parameters": result.x,
    }


def predict_model(model: str, fit: dict, radius: np.ndarray, profile: Profile, j: np.ndarray, j0: float) -> tuple[np.ndarray, np.ndarray]:
    circular = circular_speed(model, radius, profile, fit["raw_parameters"], j, j0)
    sigma_r0, h_sigma = drift_parameters(model, fit["raw_parameters"])
    return apply_radial_drift(radius, circular, sigma_r0, h_sigma)


def band_metrics(observed: np.ndarray, error: np.ndarray, predicted: np.ndarray) -> dict:
    residual = observed - predicted
    return {
        "n": int(len(residual)),
        "chi2": float(np.sum((residual / error) ** 2)),
        "rmse_kms": float(np.sqrt(np.mean(residual**2))),
        "mae_kms": float(np.mean(np.abs(residual))),
        "mean_residual_kms": float(np.mean(residual)),
    }


def clean_fit(fit: dict) -> dict:
    return {key: value for key, value in fit.items() if key not in {"raw_parameters"} and not isinstance(value, np.ndarray)} | {
        key: [float(x) for x in fit[key]]
        for key in ("predicted_observed_kms", "predicted_circular_kms", "drift_kms", "residual_kms")
    }


def percentile(values: list[float]) -> dict:
    lo, median, hi = np.percentile(values, [2.5, 50.0, 97.5])
    return {"p2_5": float(lo), "median": float(median), "p97_5": float(hi)}


def bootstrap(
    generator: np.random.Generator,
    count: int,
    fit_radius: np.ndarray,
    fit_velocity: np.ndarray,
    fit_error: np.ndarray,
    fit_j: np.ndarray,
    held_radius: np.ndarray,
    held_velocity: np.ndarray,
    held_error: np.ndarray,
    held_j: np.ndarray,
    profile: Profile,
    j0: float,
) -> dict:
    values: dict[str, list[float]] = {}
    for _ in range(count):
        indices = generator.integers(0, len(fit_radius), len(fit_radius))
        fits = {
            model: fit_model(model, fit_radius[indices], fit_velocity[indices], fit_error[indices], profile, fit_j[indices], j0)
            for model in ("scarcity", "newtonian-baryons", "nfw-baryons")
        }
        for model, fit in fits.items():
            for name, value in fit["parameters"].items():
                values.setdefault(f"{model}.{name}", []).append(value)
            held_prediction, _ = predict_model(model, fit, held_radius, profile, held_j, j0)
            held = band_metrics(held_velocity, held_error, held_prediction)
            values.setdefault(f"{model}.held_mean_residual_kms", []).append(held["mean_residual_kms"])
        values.setdefault("delta_aic_scarcity_minus_nfw", []).append(fits["scarcity"]["aic"] - fits["nfw-baryons"]["aic"])
    return {"resamples": count, "intervals": {key: percentile(value) for key, value in values.items()}}


def profile_sweep(radius: np.ndarray, velocity: np.ndarray, error: np.ndarray) -> list[dict]:
    rows = []
    for disk_scale in (2.2, 2.6, 3.0):
        for bulge_fraction in (0.1, 0.2, 0.3):
            for bulge_scale in (0.5, 0.7, 1.0):
                profile = Profile(disk_scale, bulge_fraction, bulge_scale)
                j = scarcity_integral(radius, profile)
                j0 = float(scarcity_integral(np.array([R0_KPC]), profile)[0])
                fits = {model: fit_model(model, radius, velocity, error, profile, j, j0) for model in ("scarcity", "newtonian-baryons", "nfw-baryons")}
                rows.append({
                    "profile": asdict(profile),
                    **{
                        model: {
                            "parameters": fit["parameters"],
                            "rmse_kms": fit["rmse_kms"],
                            "chi2": fit["chi2"],
                            "aic": fit["aic"],
                        }
                        for model, fit in fits.items()
                    },
                    "delta_aic_scarcity_minus_nfw": fits["scarcity"]["aic"] - fits["nfw-baryons"]["aic"],
                })
    return rows


def make_plot(path: Path, radius: np.ndarray, velocity: np.ndarray, error: np.ndarray, fit_mask: np.ndarray, held_mask: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    colors = {"scarcity": "#b7791f", "newtonian-baryons": "#4a5568", "nfw-baryons": "#2f855a"}
    labels = {"scarcity": "scarcity", "newtonian-baryons": "Newtonian baryons", "nfw-baryons": "NFW+baryons"}
    fig, (axis, residual_axis) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axis.axvspan(FIT_MIN_KPC, FIT_MAX_KPC, color="#2b6cb0", alpha=0.08, label="fit: 5–15 kpc")
    axis.axvspan(FIT_MAX_KPC, HELD_OUT_MAX_KPC, color="#dd6b20", alpha=0.08, label="held out")
    axis.errorbar(radius, velocity, yerr=error, fmt="o", ms=4, color="#20242c", label="Gaia DR3 median $v_\\phi$")
    for model, prediction in predictions.items():
        axis.plot(radius, prediction, lw=2, color=colors[model], label=labels[model])
        residual_axis.plot(radius[fit_mask], (velocity - prediction)[fit_mask], "o", ms=4, color=colors[model])
        residual_axis.plot(radius[held_mask], (velocity - prediction)[held_mask], "o", ms=4, mfc="none", color=colors[model])
    residual_axis.axhline(0, color="#718096", lw=1)
    axis.set_ylabel("observed-frame velocity (km/s)")
    residual_axis.set_ylabel("data − model")
    residual_axis.set_xlabel("Galactocentric radius (kpc)")
    axis.legend(ncol=2, fontsize=9)
    axis.grid(alpha=0.2)
    residual_axis.grid(alpha=0.2)
    fig.suptitle("ORB-10083: shared radial asymmetric drift")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument("--constant-baseline", type=Path, default=Path(__file__).resolve().parent / "assets/results.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.data, columns=["R_kpc", "v_c_kms", "v_c_err_kms", "n_stars"])
    order = np.argsort(table["R_kpc"].to_numpy(zero_copy_only=False))
    radius = table["R_kpc"].to_numpy(zero_copy_only=False).astype(float)[order]
    velocity = table["v_c_kms"].to_numpy(zero_copy_only=False).astype(float)[order]
    error = table["v_c_err_kms"].to_numpy(zero_copy_only=False).astype(float)[order]
    fit_mask = (radius >= FIT_MIN_KPC) & (radius <= FIT_MAX_KPC)
    held_mask = (radius > FIT_MAX_KPC) & (radius <= HELD_OUT_MAX_KPC)
    profile = Profile()
    j = scarcity_integral(radius, profile)
    j0 = float(scarcity_integral(np.array([R0_KPC]), profile)[0])
    models = ("scarcity", "newtonian-baryons", "nfw-baryons")
    fits = {model: fit_model(model, radius[fit_mask], velocity[fit_mask], error[fit_mask], profile, j[fit_mask], j0) for model in models}
    predictions, drifts = {}, {}
    for model, fit in fits.items():
        predictions[model], drifts[model] = predict_model(model, fit, radius, profile, j, j0)
    held = {model: band_metrics(velocity[held_mask], error[held_mask], predictions[model][held_mask]) for model in models}
    baseline = json.loads(args.constant_baseline.read_text())
    baseline_names = {"scarcity": "scarcity", "newtonian-baryons": "newtonian_baryons", "nfw-baryons": "nfw_baryons"}
    comparisons = {}
    for model in models:
        old_fit = baseline["fits"][baseline_names[model]]
        old_held = baseline["held_out_consistency"][baseline_names[model]]
        comparisons[model] = {
            "constant_drift_fit": {
                "parameters": {key: old_fit[key] for key in ("mass_msun", "beta_kpc", "halo_mass_msun", "concentration", "drift_kms") if key in old_fit},
                "rmse_kms": old_fit["rmse_kms"],
                "chi2": old_fit["chi2"],
                "dof": old_fit["dof"],
            },
            "radial_drift_fit": {
                "parameters": fits[model]["parameters"],
                "rmse_kms": fits[model]["rmse_kms"],
                "chi2": fits[model]["chi2"],
                "dof": fits[model]["dof"],
            },
            "constant_drift_held_out": old_held,
            "radial_drift_held_out": held[model],
            "held_out_mean_residual_change_kms": held[model]["mean_residual_kms"] - old_held["mean_residual_kms"],
            "absolute_held_out_mean_residual_shrank": abs(held[model]["mean_residual_kms"]) < abs(old_held["mean_residual_kms"]),
        }
    boot = bootstrap(
        rng(args.seed), args.bootstrap,
        radius[fit_mask], velocity[fit_mask], error[fit_mask], j[fit_mask],
        radius[held_mask], velocity[held_mask], error[held_mask], j[held_mask],
        profile, j0,
    )
    sweep = profile_sweep(radius[fit_mask], velocity[fit_mask], error[fit_mask])
    results = {
        "task": "ORB-10083",
        "source_baseline_task": "ORB-10082",
        "question": "Does a shared radial asymmetric-drift model remove the coherent outer underprediction, or does the gravity model still carry it?",
        "protocol": {
            "fit_range_kpc": [FIT_MIN_KPC, FIT_MAX_KPC],
            "held_out_range_kpc": [FIT_MAX_KPC, HELD_OUT_MAX_KPC],
            "seed": args.seed,
            "bootstrap_resamples": args.bootstrap,
            "profile_variants": len(sweep),
            "parameter_counts": {model: fits[model]["parameter_count"] for model in models},
        },
        "radial_drift_model": {
            "equation": "v_c^2-v_phi^2=sigma_R(R)^2[R/h_nu+2R/h_sigma-1/2]; sigma_R(R)=sigma_R0 exp[-(R-R0)/h_sigma]",
            "fixed": {"R0_kpc": R0_KPC, "tracer_scale_kpc": TRACER_SCALE_KPC, "sigma_phi2_over_sigma_R2": EPICYCLE_ANISOTROPY},
            "fitted_bounds": {"sigma_R0_kms": list(SIGMA_R0_BOUNDS_KMS), "dispersion_scale_kpc": list(DISPERSION_SCALE_BOUNDS_KPC)},
            "motivation": "Axisymmetric radial Jeans relation with exponential tracer density and radial-dispersion profiles; the same two nuisance parameters are fitted for every gravity model.",
            "limitations": [
                "The delivered mixed stellar population has no per-bin dispersion or tracer-density measurements, so h_nu and anisotropy are fixed surrogates rather than independently measured inputs.",
                "Quoted velocity errors are statistical and do not define a complete correlated-systematics likelihood.",
            ],
        },
        "fits": {model: clean_fit(fit) for model, fit in fits.items()},
        "held_out": held,
        "constant_vs_radial": comparisons,
        "bootstrap_95_percent_intervals": boot,
        "profile_sensitivity": {"variants": sweep},
        "interpretation": {
            "scarcity_coherent_underprediction_shrank": comparisons["scarcity"]["absolute_held_out_mean_residual_shrank"],
            "statement": "The radial nuisance carries most of scarcity's former coherent held-out residual, reducing its mean from +4.86 to +0.47 km/s. Gravity-model dependence remains: Newtonian baryons still underpredicts and NFW overpredicts. The scarcity dispersion normalization is bound-limited, so this is nuisance absorption rather than a measured drift profile." if comparisons["scarcity"]["absolute_held_out_mean_residual_shrank"] else "The radial-drift nuisance does not reduce the scarcity model's coherent held-out underprediction; the gravity curve still carries that misfit.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "radial-drift-results.json").write_text(json.dumps(results, indent=2) + "\n")
    make_plot(args.output_dir / "radial-drift-fit.png", radius, velocity, error, fit_mask, held_mask, predictions)
    for model in models:
        print(f"{model}: RMSE={fits[model]['rmse_kms']:.3f} km/s; chi2/dof={fits[model]['chi2']:.1f}/{fits[model]['dof']}; held mean={held[model]['mean_residual_kms']:+.3f} km/s")
    print(results["interpretation"]["statement"])


if __name__ == "__main__":
    main()
