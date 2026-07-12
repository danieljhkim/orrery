"""Test whether one fixed scarcity length fits external-galaxy rotation curves.

The apparatus consumes Astrolabe's SPARC delivery, reconstructs the standard
baryonic velocity curve (signed gas plus stellar components), and promotes it
to the ORB-10077 continuum scarcity form

    v_s(r) = A_i^1/2 v_bar(r) exp[-beta J_i(r)/2],
    J_i(r) = integral_r^rmax F_i(u)/u^2 du.

Here F_i is the normalized spherical enclosed-mass surrogate r*v_bar^2 and A_i
is one fitted baryonic normalization per galaxy.  The arbitrary additive
constant in J is absorbed by A_i, so no galaxy-specific calibration radius is
introduced.  The declared comparison fits either one beta to the whole sample
or one beta_i per galaxy, counts every normalization in AIC, tests beta_i
against disk scale and characteristic baryonic acceleration, and runs a
one-global-a0 MOND control under the same one-normalization-per-galaxy policy.

Usage: uv run lab/sims/sparc-scarcity-universality/main.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import spearmanr


ML_DISK = 0.5
ML_BULGE = 0.7
MIN_POINTS = 5
BETA_BOUNDS_KPC = (0.0, 100.0)
MASS_NORMALIZATION_BOUNDS = (0.01, 100.0)
KMS2_PER_KPC_TO_MS2 = 1.0e6 / 3.085677581491367e19


@dataclass(frozen=True)
class GalaxyCurve:
    name: str
    radius_kpc: np.ndarray
    observed_kms: np.ndarray
    error_kms: np.ndarray
    baryonic_v2_kms2: np.ndarray
    scarcity_j_per_kpc: np.ndarray
    disk_scale_kpc: float
    characteristic_gbar_ms2: float


@dataclass(frozen=True)
class CurveFit:
    beta_kpc: float
    normalization: float
    chi2: float
    beta_sigma_kpc: float
    predicted_kms: np.ndarray


def default_data_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "orrery":
            return parent.parent / "astrolabe" / "data" / "processed"
    raise RuntimeError("could not locate sibling astrolabe checkout")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scarcity_integral(radius_kpc: np.ndarray, baryonic_v2_kms2: np.ndarray) -> np.ndarray:
    """Integrate the normalized spherical surrogate on the observed lattice."""
    radius = np.asarray(radius_kpc, dtype=float)
    enclosed_surrogate = radius * np.asarray(baryonic_v2_kms2, dtype=float)
    scale = float(np.max(enclosed_surrogate))
    if scale <= 0 or np.any(np.diff(radius) <= 0):
        raise ValueError("scarcity profile needs increasing radii and positive baryonic mass")
    fraction = enclosed_surrogate / scale
    integrand = fraction / radius**2
    segments = 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(radius)
    result = np.zeros_like(radius)
    result[:-1] = np.cumsum(segments[::-1])[::-1]
    return result


def _column(table: pq.ParquetFile | object, name: str) -> np.ndarray:
    return np.asarray(table[name].to_numpy(zero_copy_only=False))


def load_sample(rotcurves_path: Path, galaxies_path: Path) -> list[GalaxyCurve]:
    rot = pq.read_table(rotcurves_path)
    gal = pq.read_table(galaxies_path)
    properties = {
        str(name): (int(quality), float(incl), float(scale))
        for name, quality, incl, scale in zip(
            _column(gal, "source_id"), _column(gal, "quality"),
            _column(gal, "incl_deg"), _column(gal, "r_disk_kpc"), strict=True
        )
    }
    names = _column(rot, "source_id").astype(str)
    radius = _column(rot, "r_kpc").astype(float)
    observed = _column(rot, "v_obs_kms").astype(float)
    error = _column(rot, "v_obs_err_kms").astype(float)
    gas = _column(rot, "v_gas_kms").astype(float)
    disk = _column(rot, "v_disk_kms").astype(float)
    bulge = (
        _column(rot, "v_bul_kms").astype(float)
        if "v_bul_kms" in rot.column_names else np.zeros_like(radius)
    )
    baryonic_v2 = (
        gas * np.abs(gas)
        + ML_DISK * disk * np.abs(disk)
        + ML_BULGE * bulge * np.abs(bulge)
    )

    sample: list[GalaxyCurve] = []
    for name in sorted(set(names)):
        if name not in properties:
            continue
        quality, inclination, disk_scale = properties[name]
        mask = names == name
        keep = (
            np.isfinite(radius[mask]) & np.isfinite(observed[mask])
            & np.isfinite(error[mask]) & np.isfinite(baryonic_v2[mask])
            & (radius[mask] > 0) & (observed[mask] > 0)
            & (error[mask] > 0) & (baryonic_v2[mask] > 0)
        )
        if quality > 2 or inclination < 30 or not np.isfinite(disk_scale) or disk_scale <= 0:
            continue
        indices = np.flatnonzero(mask)[keep]
        if len(indices) < MIN_POINTS:
            continue
        indices = indices[np.argsort(radius[indices])]
        r = radius[indices]
        vb2 = baryonic_v2[indices]
        gbar = vb2 / r * KMS2_PER_KPC_TO_MS2
        sample.append(
            GalaxyCurve(
                name=name,
                radius_kpc=r,
                observed_kms=observed[indices],
                error_kms=error[indices],
                baryonic_v2_kms2=vb2,
                scarcity_j_per_kpc=scarcity_integral(r, vb2),
                disk_scale_kpc=disk_scale,
                characteristic_gbar_ms2=float(np.median(gbar)),
            )
        )
    if not sample:
        raise ValueError("no galaxies survive Q<=2, inclination>=30 deg, and data cuts")
    return sample


def fit_at_beta(curve: GalaxyCurve, beta_kpc: float) -> CurveFit:
    basis = np.sqrt(curve.baryonic_v2_kms2) * np.exp(
        -0.5 * beta_kpc * curve.scarcity_j_per_kpc
    )
    weights = 1.0 / curve.error_kms**2
    amplitude = float(np.sum(weights * basis * curve.observed_kms) / np.sum(weights * basis**2))
    amplitude = float(np.clip(amplitude, np.sqrt(MASS_NORMALIZATION_BOUNDS[0]),
                              np.sqrt(MASS_NORMALIZATION_BOUNDS[1])))
    predicted = amplitude * basis
    value = float(np.sum(((curve.observed_kms - predicted) / curve.error_kms) ** 2))
    return CurveFit(beta_kpc, amplitude**2, value, float("nan"), predicted)


def fit_curve(curve: GalaxyCurve) -> CurveFit:
    objective = lambda beta: fit_at_beta(curve, float(beta)).chi2
    result = minimize_scalar(objective, bounds=BETA_BOUNDS_KPC, method="bounded",
                             options={"xatol": 1e-6})
    fit = fit_at_beta(curve, float(result.x))
    step = max(0.02, 0.01 * max(fit.beta_kpc, 1.0))
    if fit.beta_kpc - step <= BETA_BOUNDS_KPC[0] or fit.beta_kpc + step >= BETA_BOUNDS_KPC[1]:
        sigma = float("nan")
    else:
        curvature = (
            objective(fit.beta_kpc + step) - 2.0 * fit.chi2
            + objective(fit.beta_kpc - step)
        ) / step**2
        sigma = float(np.sqrt(2.0 / curvature)) if curvature > 0 else float("nan")
    return CurveFit(fit.beta_kpc, fit.normalization, fit.chi2, sigma, fit.predicted_kms)


def fit_global_beta(sample: list[GalaxyCurve]) -> tuple[float, list[CurveFit], float]:
    objective = lambda beta: sum(fit_at_beta(curve, float(beta)).chi2 for curve in sample)
    result = minimize_scalar(objective, bounds=BETA_BOUNDS_KPC, method="bounded",
                             options={"xatol": 1e-6})
    beta = float(result.x)
    fits = [fit_at_beta(curve, beta) for curve in sample]
    return beta, fits, float(sum(fit.chi2 for fit in fits))


def mond_prediction(curve: GalaxyCurve, normalization: float, a0_ms2: float) -> np.ndarray:
    gbar = normalization * curve.baryonic_v2_kms2 / curve.radius_kpc
    a0 = a0_ms2 / KMS2_PER_KPC_TO_MS2
    gmond = 0.5 * (gbar + np.sqrt(gbar**2 + 4.0 * a0 * gbar))
    return np.sqrt(gmond * curve.radius_kpc)


def fit_mond_at_a0(curve: GalaxyCurve, a0_ms2: float) -> dict:
    def objective(log_normalization: float) -> float:
        predicted = mond_prediction(curve, 10.0**log_normalization, a0_ms2)
        return float(np.sum(((curve.observed_kms - predicted) / curve.error_kms) ** 2))

    bounds = tuple(np.log10(MASS_NORMALIZATION_BOUNDS))
    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    normalization = float(10.0**result.x)
    predicted = mond_prediction(curve, normalization, a0_ms2)
    return {"normalization": normalization, "chi2": float(result.fun), "predicted_kms": predicted}


def fit_global_mond(sample: list[GalaxyCurve]) -> tuple[float, list[dict], float]:
    def objective(log_a0: float) -> float:
        return sum(fit_mond_at_a0(curve, 10.0**log_a0)["chi2"] for curve in sample)

    result = minimize_scalar(objective, bounds=(-13.0, -8.0), method="bounded",
                             options={"xatol": 1e-5})
    a0 = float(10.0**result.x)
    fits = [fit_mond_at_a0(curve, a0) for curve in sample]
    return a0, fits, float(sum(fit["chi2"] for fit in fits))


def correlation(x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    statistic, pvalue = spearmanr(x[valid], y[valid])
    return {"n": int(np.sum(valid)), "spearman_rho": float(statistic), "p_value": float(pvalue)}


def fit_metrics(curve: GalaxyCurve, predicted: np.ndarray) -> dict:
    residual = curve.observed_kms - predicted
    log_residual = np.log10(curve.observed_kms / predicted)
    return {
        "rmse_kms": float(np.sqrt(np.mean(residual**2))),
        "mean_residual_kms": float(np.mean(residual)),
        "median_log10_velocity_ratio": float(np.median(log_residual)),
        "median_abs_log10_velocity_ratio": float(np.median(np.abs(log_residual))),
    }


def make_plot(path: Path, sample: list[GalaxyCurve], per_fits: list[CurveFit],
              global_fits: list[CurveFit]) -> None:
    scales = np.array([curve.disk_scale_kpc for curve in sample])
    accelerations = np.array([curve.characteristic_gbar_ms2 for curve in sample])
    betas = np.array([fit.beta_kpc for fit in per_fits])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].scatter(scales, betas, s=16, alpha=0.7)
    axes[0].set(xscale="log", xlabel="disk scale length (kpc)", ylabel=r"per-galaxy $\beta_i$ (kpc)")
    axes[1].scatter(accelerations, betas, s=16, alpha=0.7)
    axes[1].set(xscale="log", xlabel=r"median $g_{bar}$ (m s$^{-2}$)", ylabel=r"per-galaxy $\beta_i$ (kpc)")
    for curve, fit in zip(sample, global_fits, strict=True):
        gbar = curve.baryonic_v2_kms2 / curve.radius_kpc * KMS2_PER_KPC_TO_MS2
        gpred = fit.predicted_kms**2 / curve.radius_kpc * KMS2_PER_KPC_TO_MS2
        axes[2].scatter(gbar, gpred, s=4, alpha=0.25, color="#b7791f")
    limits = (1e-13, 1e-8)
    axes[2].plot(limits, limits, "--", color="#4a5568", lw=1)
    axes[2].set(xscale="log", yscale="log", xlim=limits, ylim=limits,
                xlabel=r"published $g_{bar}$ (m s$^{-2}$)",
                ylabel=r"global-$\beta$ predicted $g$ (m s$^{-2}$)")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("SPARC fixed-length scarcity universality test (ORB-10169)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def serializable_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def run(sample: list[GalaxyCurve]) -> dict:
    global_beta, global_fits, global_chi2 = fit_global_beta(sample)
    per_fits = [fit_curve(curve) for curve in sample]
    per_chi2 = float(sum(fit.chi2 for fit in per_fits))
    global_a0, mond_fits, mond_chi2 = fit_global_mond(sample)
    n_galaxies = len(sample)
    n_points = sum(len(curve.radius_kpc) for curve in sample)
    scales = np.array([curve.disk_scale_kpc for curve in sample])
    accelerations = np.array([curve.characteristic_gbar_ms2 for curve in sample])
    betas = np.array([fit.beta_kpc for fit in per_fits])
    sigmas = np.array([fit.beta_sigma_kpc for fit in per_fits])
    global_residual_strength = np.array([
        fit_metrics(curve, fit.predicted_kms)["median_abs_log10_velocity_ratio"]
        for curve, fit in zip(sample, global_fits, strict=True)
    ])
    beta_size = correlation(scales, betas)
    beta_acceleration = correlation(accelerations, betas)
    residual_size = correlation(scales, global_residual_strength)
    residual_acceleration = correlation(accelerations, global_residual_strength)
    valid_sigma = np.isfinite(sigmas) & (sigmas > 0)
    if int(np.sum(valid_sigma)) >= 2:
        weights = 1.0 / sigmas[valid_sigma]**2
        constant = float(np.sum(weights * betas[valid_sigma]) / np.sum(weights))
        heterogeneity_chi2 = float(np.sum(weights * (betas[valid_sigma] - constant) ** 2))
        heterogeneity_dof = int(np.sum(valid_sigma) - 1)
        heterogeneity_p = float(chi2_distribution.sf(heterogeneity_chi2, heterogeneity_dof))
    else:
        constant = heterogeneity_chi2 = heterogeneity_p = float("nan")
        heterogeneity_dof = 0
    fixed_length_refuted = bool(
        beta_size["spearman_rho"] > 0
        and beta_size["p_value"] < 0.01
        and heterogeneity_p < 0.01
    )
    global_parameters = n_galaxies + 1
    per_parameters = 2 * n_galaxies
    mond_parameters = n_galaxies + 1
    rows = []
    for curve, global_fit, per_fit, mond_fit in zip(
        sample, global_fits, per_fits, mond_fits, strict=True
    ):
        rows.append({
            "source_id": curve.name,
            "n_points": int(len(curve.radius_kpc)),
            "r_disk_kpc": curve.disk_scale_kpc,
            "characteristic_gbar_ms2": curve.characteristic_gbar_ms2,
            "global_scarcity": {
                "normalization": global_fit.normalization,
                "chi2": global_fit.chi2,
                **fit_metrics(curve, global_fit.predicted_kms),
            },
            "per_galaxy_scarcity": {
                "beta_kpc": per_fit.beta_kpc,
                "beta_sigma_kpc": serializable_float(per_fit.beta_sigma_kpc),
                "normalization": per_fit.normalization,
                "chi2": per_fit.chi2,
                **fit_metrics(curve, per_fit.predicted_kms),
            },
            "mond_control": {
                "normalization": mond_fit["normalization"],
                "chi2": mond_fit["chi2"],
                **fit_metrics(curve, mond_fit["predicted_kms"]),
            },
        })
    return {
        "task": "ORB-10169",
        "protocol": {
            "sample_cuts": {"quality_max": 2, "inclination_min_deg": 30, "min_points": MIN_POINTS},
            "mass_to_light": {"disk": ML_DISK, "bulge": ML_BULGE},
            "nuisance_policy": "one positive baryonic normalization per galaxy; no drift term",
            "scarcity_equation": "v=sqrt(A)*v_bar*exp[-beta*integral(F/u^2 du)/2]",
            "parameter_counts": {"global_beta": global_parameters, "per_galaxy_beta": per_parameters,
                                 "mond_global_a0": mond_parameters},
            "kill_condition": "positive beta_i--disk-scale Spearman correlation with p<0.01 and rejection of constant beta_i at p<0.01",
        },
        "sample": {"galaxies": n_galaxies, "points": n_points},
        "global_beta_scarcity": {
            "beta_kpc": global_beta, "milky_way_beta_kpc": 5.25,
            "chi2": global_chi2, "parameters": global_parameters,
            "dof": n_points - global_parameters,
            "aic": global_chi2 + 2 * global_parameters,
        },
        "per_galaxy_beta_scarcity": {
            "chi2": per_chi2, "parameters": per_parameters,
            "dof": n_points - per_parameters, "aic": per_chi2 + 2 * per_parameters,
            "beta_kpc": {"median": float(np.median(betas)), "p16": float(np.percentile(betas, 16)),
                         "p84": float(np.percentile(betas, 84)), "min": float(np.min(betas)),
                         "max": float(np.max(betas))},
            "constant_heterogeneity": {"inverse_variance_beta_kpc": serializable_float(constant),
                                       "chi2": serializable_float(heterogeneity_chi2),
                                       "dof": heterogeneity_dof, "p_value": serializable_float(heterogeneity_p)},
        },
        "aic_comparison": {
            "delta_aic_global_minus_per_galaxy": global_chi2 + 2 * global_parameters - (per_chi2 + 2 * per_parameters)
        },
        "correlations": {
            "beta_i_vs_disk_scale": beta_size,
            "beta_i_vs_characteristic_acceleration": beta_acceleration,
            "global_residual_strength_vs_disk_scale": residual_size,
            "global_residual_strength_vs_characteristic_acceleration": residual_acceleration,
        },
        "mond_control": {
            "a0_ms2": global_a0, "chi2": mond_chi2, "parameters": mond_parameters,
            "dof": n_points - mond_parameters, "aic": mond_chi2 + 2 * mond_parameters,
        },
        "verdict": {
            "fixed_length_kill_condition_met": fixed_length_refuted,
            "statement": (
                "fixed-beta scarcity is refuted by the predeclared size-tracking kill condition"
                if fixed_length_refuted else
                "the predeclared size-tracking kill condition is not met"
            ),
        },
        "galaxies": rows,
    }


def parse_args() -> argparse.Namespace:
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotcurves", type=Path, default=data_dir / "catalog" / "sparc_rotcurves.parquet")
    parser.add_argument("--galaxies", type=Path, default=data_dir / "catalog" / "sparc_galaxies.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.rotcurves, args.galaxies):
        if not path.exists():
            raise FileNotFoundError(f"missing Astrolabe SPARC delivery: {path}; run astrolabe/scripts/deliver_sparc.py")
    sample = load_sample(args.rotcurves, args.galaxies)
    results = run(sample)
    results["inputs"] = {
        "rotcurves": str(args.rotcurves), "rotcurves_sha256": file_sha256(args.rotcurves),
        "galaxies": str(args.galaxies), "galaxies_sha256": file_sha256(args.galaxies),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    global_beta, global_fits, _ = fit_global_beta(sample)
    per_fits = [fit_curve(curve) for curve in sample]
    make_plot(args.output_dir / "diagnostics.png", sample, per_fits, global_fits)
    print(f"sample: {results['sample']['galaxies']} galaxies, {results['sample']['points']} points")
    print(f"global beta: {global_beta:.4g} kpc")
    print(f"delta AIC (global - per-galaxy): {results['aic_comparison']['delta_aic_global_minus_per_galaxy']:.1f}")
    print(results["verdict"]["statement"])


if __name__ == "__main__":
    main()
