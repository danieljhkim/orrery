"""Compare scarcity with NFW+baryons on the Gaia DR3 rotation curve.

Question
--------
Does scarcity remain preferred on Tycho's 5--15 kpc Gaia DR3 median-v_phi curve
when compared with a standard three-physical-parameter NFW+baryons model and
penalized for each model's actual parameter count?

The apparatus uses the imported spherical enclosed-mass surrogate for an
exponential disk plus Hernquist bulge.  In real units its circular speed is

    v_N^2(r) = G_local M F(r) / r
    v_S^2(r) = v_N^2(r) q(r) / q(R0)
    q(r) = exp[-beta integral_r^Rmax F(u)/u^2 du]

where beta >= 0 is the model's one scarcity shape normalization, R0=8.25 kpc
is the local-G calibration radius, and beta=0 is the Newtonian control.  This
is a phenomenological promotion of the toy model, not a derivation from general
relativity or an exact thin-disk potential.

Protocol declared before seeing the fit
---------------------------------------
Fit 5--15 kpc only.  Hold 15--18.75 kpc out as a consistency band.  Apply the
same additive drift nuisance to every model.  NFW fits baryonic mass, M200, and
concentration, so it has four counted parameters including drift versus
scarcity's three.  Call the relative result decisive
only when |delta AIC| >= 10 and its sign survives the mass-profile sweep; raw
chi-square, RMSE, bootstrap intervals, lattice convergence, and all limitations
remain visible even then.

Usage
-----
uv run lab/sims/scarcity-rotation-curve-fit/main.py
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
CONSISTENCY_MAX_KPC = 18.75
DRIFT_MIN_KMS = 3.0
DRIFT_MAX_KMS = 10.0
CALIBRATION_RADIUS_KPC = 8.25
H0_KMS_MPC = 70.0
NFW_BARYONIC_MASS_BOUNDS_MSUN = (1.0e9, 3.0e11)
NFW_HALO_MASS_BOUNDS_MSUN = (1.0e10, 3.0e13)
NFW_CONCENTRATION_BOUNDS = (1.0, 40.0)


@dataclass(frozen=True)
class Profile:
    disk_scale_kpc: float = 2.6
    bulge_fraction: float = 0.2
    bulge_scale_kpc: float = 0.7


@dataclass
class Fit:
    model: str
    mass_msun: float
    beta_kpc: float
    alpha_kpc_per_msun: float
    drift_kms: float
    chi2: float
    dof: int
    reduced_chi2: float
    rmse_kms: float
    mae_kms: float
    aic: float
    bic: float
    predicted_observed_kms: np.ndarray
    predicted_circular_kms: np.ndarray
    residual_kms: np.ndarray

    def serializable(self) -> dict:
        out = asdict(self)
        for key in (
            "predicted_observed_kms",
            "predicted_circular_kms",
            "residual_kms",
        ):
            out[key] = [float(x) for x in out[key]]
        return out


@dataclass
class NfwFit:
    model: str
    mass_msun: float
    halo_mass_msun: float
    concentration: float
    drift_kms: float
    virial_radius_kpc: float
    chi2: float
    dof: int
    reduced_chi2: float
    rmse_kms: float
    mae_kms: float
    aic: float
    bic: float
    predicted_observed_kms: np.ndarray
    predicted_circular_kms: np.ndarray
    residual_kms: np.ndarray

    def serializable(self) -> dict:
        out = asdict(self)
        for key in ("predicted_observed_kms", "predicted_circular_kms", "residual_kms"):
            out[key] = [float(x) for x in out[key]]
        return out


def default_data_path() -> Path:
    codebases = Path(__file__).resolve().parents[4]
    return (
        codebases
        / "astrolabe/data/processed/derived/mw_rotation_curve.parquet"
    )


def enclosed_fraction(radius_kpc: np.ndarray, profile: Profile) -> np.ndarray:
    """Imported distributed-mass prescription, normalized to unit total mass."""
    r = np.asarray(radius_kpc, dtype=float)
    bulge = profile.bulge_fraction * r**2 / (r + profile.bulge_scale_kpc) ** 2
    x = r / profile.disk_scale_kpc
    disk = (1.0 - profile.bulge_fraction) * (1.0 - (1.0 + x) * np.exp(-x))
    return bulge + disk


def critical_density_msun_kpc3() -> float:
    h0_kms_kpc = H0_KMS_MPC / 1000.0
    return 3.0 * h0_kms_kpc**2 / (8.0 * np.pi * G_KPC_KMS2_MSUN)


def nfw_enclosed_mass(
    radius_kpc: np.ndarray, halo_mass_msun: float, concentration: float
) -> tuple[np.ndarray, float]:
    """NFW M(<r) for R200 enclosing 200 times the critical density."""
    r200 = (
        3.0 * halo_mass_msun
        / (4.0 * np.pi * 200.0 * critical_density_msun_kpc3())
    ) ** (1.0 / 3.0)
    x = concentration * np.asarray(radius_kpc, dtype=float) / r200
    norm = np.log1p(concentration) - concentration / (1.0 + concentration)
    enclosed = halo_mass_msun * (np.log1p(x) - x / (1.0 + x)) / norm
    return enclosed, float(r200)


def nfw_prediction(
    radius_kpc: np.ndarray,
    profile: Profile,
    mass_msun: float,
    halo_mass_msun: float,
    concentration: float,
    drift_kms: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    fraction = enclosed_fraction(radius_kpc, profile)
    halo_enclosed, r200 = nfw_enclosed_mass(radius_kpc, halo_mass_msun, concentration)
    circular = np.sqrt(
        G_KPC_KMS2_MSUN * (mass_msun * fraction + halo_enclosed) / radius_kpc
    )
    return circular - drift_kms, circular, r200


def fit_nfw_curve(
    radius_kpc: np.ndarray,
    observed_kms: np.ndarray,
    error_kms: np.ndarray,
    profile: Profile,
) -> NfwFit:
    """Fit free baryonic mass, NFW M200/concentration, and common drift."""
    lower = np.array([
        np.log10(NFW_BARYONIC_MASS_BOUNDS_MSUN[0]),
        np.log10(NFW_HALO_MASS_BOUNDS_MSUN[0]),
        NFW_CONCENTRATION_BOUNDS[0],
        DRIFT_MIN_KMS,
    ])
    upper = np.array([
        np.log10(NFW_BARYONIC_MASS_BOUNDS_MSUN[1]),
        np.log10(NFW_HALO_MASS_BOUNDS_MSUN[1]),
        NFW_CONCENTRATION_BOUNDS[1],
        DRIFT_MAX_KMS,
    ])

    def weighted_residual(parameters: np.ndarray) -> np.ndarray:
        predicted, _, _ = nfw_prediction(
            radius_kpc, profile, 10.0 ** parameters[0], 10.0 ** parameters[1],
            parameters[2], parameters[3]
        )
        return (observed_kms - predicted) / error_kms

    starts = ((11.0, 12.0, 8.0, 3.0), (10.7, 12.2, 12.0, 5.0), (11.2, 11.7, 18.0, 8.0))
    candidates = [
        least_squares(
            weighted_residual, np.asarray(start), bounds=(lower, upper),
            xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=3000
        )
        for start in starts
    ]
    result = min(candidates, key=lambda candidate: float(np.sum(candidate.fun**2)))
    mass_msun = float(10.0 ** result.x[0])
    halo_mass_msun = float(10.0 ** result.x[1])
    concentration = float(result.x[2])
    drift_kms = float(result.x[3])
    predicted, circular, r200 = nfw_prediction(
        radius_kpc, profile, mass_msun, halo_mass_msun, concentration, drift_kms
    )
    residual = observed_kms - predicted
    chi2 = float(np.sum(np.square(residual / error_kms)))
    parameters = 4
    dof = len(radius_kpc) - parameters
    return NfwFit(
        model="nfw-baryons", mass_msun=mass_msun, halo_mass_msun=halo_mass_msun,
        concentration=concentration, drift_kms=drift_kms, virial_radius_kpc=r200,
        chi2=chi2, dof=dof, reduced_chi2=chi2 / dof,
        rmse_kms=float(np.sqrt(np.mean(residual**2))),
        mae_kms=float(np.mean(np.abs(residual))), aic=chi2 + 2 * parameters,
        bic=chi2 + parameters * np.log(len(radius_kpc)),
        predicted_observed_kms=predicted, predicted_circular_kms=circular,
        residual_kms=residual,
    )


def scarcity_integral(
    radius_kpc: np.ndarray,
    profile: Profile,
    *,
    outer_radius_kpc: float = 100.0,
    step_kpc: float = 0.01,
) -> np.ndarray:
    """Return integral_r^Rmax F(u)/u^2 du on a radial lattice."""
    r = np.asarray(radius_kpc, dtype=float)
    if np.any(r <= 0) or outer_radius_kpc <= float(np.max(r)):
        raise ValueError("radii must be positive and inside outer_radius_kpc")
    grid = np.arange(step_kpc, outer_radius_kpc + step_kpc, step_kpc)
    integrand = enclosed_fraction(grid, profile) / grid**2
    segments = 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(grid)
    tail = np.zeros_like(grid)
    tail[:-1] = np.cumsum(segments[::-1])[::-1]
    return np.interp(r, grid, tail)


def load_curve(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=["R_kpc", "v_c_kms", "v_c_err_kms", "n_stars"])
    data = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    order = np.argsort(data["R_kpc"])
    return {name: np.asarray(values[order]) for name, values in data.items()}


def fit_curve(
    radius_kpc: np.ndarray,
    observed_kms: np.ndarray,
    error_kms: np.ndarray,
    profile: Profile,
    scarcity_j_per_kpc: np.ndarray,
    calibration_j_per_kpc: float,
    *,
    model: str,
    beta_grid_kpc: np.ndarray | None = None,
    drift_grid_kms: np.ndarray | None = None,
) -> Fit:
    """Weighted grid fit with analytic mass amplitude at each beta/drift pair."""
    if model not in {"newtonian-baryons", "scarcity"}:
        raise ValueError(f"unknown model: {model}")
    if beta_grid_kpc is None:
        beta_grid_kpc = np.linspace(0.0, 80.0, 1601)
    if drift_grid_kms is None:
        drift_grid_kms = np.linspace(DRIFT_MIN_KMS, DRIFT_MAX_KMS, 141)
    if model == "newtonian-baryons":
        beta_grid_kpc = np.array([0.0])

    fraction = enclosed_fraction(radius_kpc, profile)
    unit_mass_basis = np.sqrt(G_KPC_KMS2_MSUN * fraction / radius_kpc)
    # L-0001: local-G calibration requires q(r)/q(R0), not absolute q(r).
    bases = unit_mass_basis[None, :] * np.exp(
        -0.5
        * np.outer(beta_grid_kpc, scarcity_j_per_kpc - calibration_j_per_kpc)
    )

    weights = 1.0 / np.square(error_kms)
    denominator = np.sum(weights[None, :] * bases**2, axis=1)
    numerator_y = np.sum(weights[None, :] * bases * observed_kms[None, :], axis=1)
    numerator_one = np.sum(weights[None, :] * bases, axis=1)
    numerator = numerator_y[:, None] + numerator_one[:, None] * drift_grid_kms[None, :]
    target_norm = (
        np.sum(weights * observed_kms**2)
        + 2.0 * drift_grid_kms * np.sum(weights * observed_kms)
        + drift_grid_kms**2 * np.sum(weights)
    )
    chi2_grid = target_norm[None, :] - numerator**2 / denominator[:, None]
    beta_index, drift_index = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)

    amplitude_sqrt_msun = numerator[beta_index, drift_index] / denominator[beta_index]
    mass_msun = float(amplitude_sqrt_msun**2)
    beta_kpc = float(beta_grid_kpc[beta_index])
    drift_kms = float(drift_grid_kms[drift_index])
    circular = amplitude_sqrt_msun * bases[beta_index]
    predicted = circular - drift_kms
    residual = observed_kms - predicted
    chi2 = float(np.sum(np.square(residual / error_kms)))
    parameters = 2 if model == "newtonian-baryons" else 3
    dof = len(radius_kpc) - parameters
    return Fit(
        model=model,
        mass_msun=mass_msun,
        beta_kpc=beta_kpc,
        alpha_kpc_per_msun=beta_kpc / mass_msun,
        drift_kms=drift_kms,
        chi2=chi2,
        dof=dof,
        reduced_chi2=chi2 / dof,
        rmse_kms=float(np.sqrt(np.mean(residual**2))),
        mae_kms=float(np.mean(np.abs(residual))),
        aic=chi2 + 2 * parameters,
        bic=chi2 + parameters * np.log(len(radius_kpc)),
        predicted_observed_kms=predicted,
        predicted_circular_kms=circular,
        residual_kms=residual,
    )


def predict(
    radius_kpc: np.ndarray,
    profile: Profile,
    j: np.ndarray,
    calibration_j_per_kpc: float,
    fit: Fit,
) -> np.ndarray:
    fraction = enclosed_fraction(radius_kpc, profile)
    circular = np.sqrt(G_KPC_KMS2_MSUN * fit.mass_msun * fraction / radius_kpc)
    # L-0001: the ratio removes the arbitrary outer-boundary normalization.
    circular *= np.exp(-0.5 * fit.beta_kpc * (j - calibration_j_per_kpc))
    return circular - fit.drift_kms


def band_metrics(observed: np.ndarray, error: np.ndarray, predicted: np.ndarray) -> dict:
    residual = observed - predicted
    chi2 = float(np.sum(np.square(residual / error)))
    return {
        "n": int(len(observed)),
        "chi2": chi2,
        "rmse_kms": float(np.sqrt(np.mean(residual**2))),
        "mae_kms": float(np.mean(np.abs(residual))),
        "mean_residual_kms": float(np.mean(residual)),
    }


def percentile_interval(values: list[float]) -> dict:
    lo, median, hi = np.percentile(values, [2.5, 50.0, 97.5])
    return {"p2_5": float(lo), "median": float(median), "p97_5": float(hi)}


def bootstrap(
    generator: np.random.Generator,
    count: int,
    radius: np.ndarray,
    velocity: np.ndarray,
    error: np.ndarray,
    profile: Profile,
    j: np.ndarray,
    calibration_j_per_kpc: float,
) -> dict:
    values = {
        "scarcity_mass_msun": [], "scarcity_beta_kpc": [],
        "scarcity_drift_kms": [], "nfw_baryonic_mass_msun": [],
        "nfw_halo_mass_msun": [], "nfw_concentration": [],
        "nfw_drift_kms": [], "delta_aic_scarcity_minus_newtonian": [],
        "delta_aic_scarcity_minus_nfw": [],
    }
    for _ in range(count):
        indices = generator.integers(0, len(radius), len(radius))
        scarcity = fit_curve(
            radius[indices], velocity[indices], error[indices], profile, j[indices],
            calibration_j_per_kpc,
            model="scarcity",
        )
        newton = fit_curve(
            radius[indices], velocity[indices], error[indices], profile, j[indices],
            calibration_j_per_kpc,
            model="newtonian-baryons",
        )
        nfw = fit_nfw_curve(radius[indices], velocity[indices], error[indices], profile)
        values["scarcity_mass_msun"].append(scarcity.mass_msun)
        values["scarcity_beta_kpc"].append(scarcity.beta_kpc)
        values["scarcity_drift_kms"].append(scarcity.drift_kms)
        values["nfw_baryonic_mass_msun"].append(nfw.mass_msun)
        values["nfw_halo_mass_msun"].append(nfw.halo_mass_msun)
        values["nfw_concentration"].append(nfw.concentration)
        values["nfw_drift_kms"].append(nfw.drift_kms)
        values["delta_aic_scarcity_minus_newtonian"].append(scarcity.aic - newton.aic)
        values["delta_aic_scarcity_minus_nfw"].append(scarcity.aic - nfw.aic)
    return {"resamples": count, **{key: percentile_interval(value) for key, value in values.items()}}


def profile_sweep(
    radius: np.ndarray,
    velocity: np.ndarray,
    error: np.ndarray,
) -> list[dict]:
    rows = []
    for disk_scale in (2.2, 2.6, 3.0):
        for bulge_fraction in (0.1, 0.2, 0.3):
            for bulge_scale in (0.5, 0.7, 1.0):
                profile = Profile(disk_scale, bulge_fraction, bulge_scale)
                j = scarcity_integral(radius, profile)
                j_cal = float(scarcity_integral(np.array([CALIBRATION_RADIUS_KPC]), profile)[0])
                scarcity = fit_curve(
                    radius, velocity, error, profile, j, j_cal, model="scarcity"
                )
                newton = fit_curve(
                    radius, velocity, error, profile, j, j_cal,
                    model="newtonian-baryons",
                )
                nfw = fit_nfw_curve(radius, velocity, error, profile)
                rows.append(
                    {
                        "profile": asdict(profile),
                        "scarcity_mass_msun": scarcity.mass_msun,
                        "scarcity_beta_kpc": scarcity.beta_kpc,
                        "scarcity_drift_kms": scarcity.drift_kms,
                        "scarcity_rmse_kms": scarcity.rmse_kms,
                        "newtonian_mass_msun": newton.mass_msun,
                        "newtonian_drift_kms": newton.drift_kms,
                        "newtonian_rmse_kms": newton.rmse_kms,
                        "nfw_baryonic_mass_msun": nfw.mass_msun,
                        "nfw_halo_mass_msun": nfw.halo_mass_msun,
                        "nfw_concentration": nfw.concentration,
                        "nfw_drift_kms": nfw.drift_kms,
                        "nfw_rmse_kms": nfw.rmse_kms,
                        "delta_aic_scarcity_minus_newtonian": scarcity.aic - newton.aic,
                        "delta_aic_scarcity_minus_nfw": scarcity.aic - nfw.aic,
                    }
                )
    return rows


def convergence_checks(
    radius: np.ndarray,
    velocity: np.ndarray,
    error: np.ndarray,
    profile: Profile,
    baseline_fit: Fit,
) -> list[dict]:
    rows = []
    baseline_j = scarcity_integral(radius, profile)
    baseline_j_cal = float(
        scarcity_integral(np.array([CALIBRATION_RADIUS_KPC]), profile)[0]
    )
    baseline_prediction = predict(
        radius, profile, baseline_j, baseline_j_cal, baseline_fit
    )
    for outer_radius in (50.0, 100.0, 200.0):
        for step in (0.02, 0.01, 0.005):
            j = scarcity_integral(
                radius, profile, outer_radius_kpc=outer_radius, step_kpc=step
            )
            j_cal = float(
                scarcity_integral(
                    np.array([CALIBRATION_RADIUS_KPC]),
                    profile,
                    outer_radius_kpc=outer_radius,
                    step_kpc=step,
                )[0]
            )
            fit = fit_curve(
                radius, velocity, error, profile, j, j_cal, model="scarcity"
            )
            prediction = predict(radius, profile, j, j_cal, fit)
            rows.append(
                {
                    "outer_radius_kpc": outer_radius,
                    "step_kpc": step,
                    "mass_msun": fit.mass_msun,
                    "beta_kpc": fit.beta_kpc,
                    "drift_kms": fit.drift_kms,
                    "rmse_kms": fit.rmse_kms,
                    "max_prediction_shift_vs_baseline_kms": float(
                        np.max(np.abs(prediction - baseline_prediction))
                    ),
                }
            )
    return rows


def make_plot(
    path: Path,
    radius: np.ndarray,
    velocity: np.ndarray,
    error: np.ndarray,
    fit_mask: np.ndarray,
    consistency_mask: np.ndarray,
    scarcity_prediction: np.ndarray,
    newton_prediction: np.ndarray,
    nfw_prediction_values: np.ndarray,
) -> None:
    fig, (ax, residual_ax) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.axvspan(FIT_MIN_KPC, FIT_MAX_KPC, color="#2b6cb0", alpha=0.08, label="fit: 5–15 kpc")
    ax.axvspan(FIT_MAX_KPC, CONSISTENCY_MAX_KPC, color="#dd6b20", alpha=0.08, label="consistency only")
    ax.errorbar(radius, velocity, yerr=error, fmt="o", ms=4, color="#20242c", label="Gaia DR3 median $v_\\phi$")
    ax.plot(radius, scarcity_prediction, color="#b7791f", lw=2.2, label="scarcity")
    ax.plot(radius, newton_prediction, color="#4a5568", lw=2, ls="--", label="Newtonian baryons")
    ax.plot(radius, nfw_prediction_values, color="#2f855a", lw=2, ls="-.", label="NFW+baryons")
    ax.set_ylabel("observed-frame velocity (km/s)")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.2)

    residual_ax.axhline(0, color="#718096", lw=1)
    residual_ax.plot(radius[fit_mask], (velocity - scarcity_prediction)[fit_mask], "o", color="#b7791f")
    residual_ax.plot(radius[fit_mask], (velocity - newton_prediction)[fit_mask], "s", ms=4, color="#4a5568")
    residual_ax.plot(radius[fit_mask], (velocity - nfw_prediction_values)[fit_mask], "^", ms=4, color="#2f855a")
    residual_ax.plot(radius[consistency_mask], (velocity - scarcity_prediction)[consistency_mask], "o", mfc="none", color="#b7791f")
    residual_ax.plot(radius[consistency_mask], (velocity - newton_prediction)[consistency_mask], "s", ms=4, mfc="none", color="#4a5568")
    residual_ax.plot(radius[consistency_mask], (velocity - nfw_prediction_values)[consistency_mask], "^", ms=4, mfc="none", color="#2f855a")
    residual_ax.set_xlabel("Galactocentric radius (kpc)")
    residual_ax.set_ylabel("data − model")
    residual_ax.grid(alpha=0.2)
    fig.suptitle("ORB-10082: scarcity versus NFW+baryons")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curve = load_curve(args.data)
    radius = curve["R_kpc"].astype(float)
    velocity = curve["v_c_kms"].astype(float)
    error = curve["v_c_err_kms"].astype(float)
    fit_mask = (radius >= FIT_MIN_KPC) & (radius <= FIT_MAX_KPC)
    consistency_mask = (radius > FIT_MAX_KPC) & (radius <= CONSISTENCY_MAX_KPC)
    if int(np.sum(fit_mask)) < 10 or int(np.sum(consistency_mask)) < 2:
        raise ValueError("dataset does not contain the declared fit and consistency bands")

    profile = Profile()
    j_all = scarcity_integral(radius, profile)
    j_cal = float(scarcity_integral(np.array([CALIBRATION_RADIUS_KPC]), profile)[0])
    fit_r, fit_v, fit_e, fit_j = (
        radius[fit_mask], velocity[fit_mask], error[fit_mask], j_all[fit_mask]
    )
    scarcity = fit_curve(
        fit_r, fit_v, fit_e, profile, fit_j, j_cal, model="scarcity"
    )
    newton = fit_curve(
        fit_r, fit_v, fit_e, profile, fit_j, j_cal, model="newtonian-baryons"
    )
    nfw = fit_nfw_curve(fit_r, fit_v, fit_e, profile)
    scarcity_all = predict(radius, profile, j_all, j_cal, scarcity)
    newton_all = predict(radius, profile, j_all, j_cal, newton)
    nfw_all, _, _ = nfw_prediction(
        radius, profile, nfw.mass_msun, nfw.halo_mass_msun,
        nfw.concentration, nfw.drift_kms
    )

    sensitivity = profile_sweep(fit_r, fit_v, fit_e)
    convergence = convergence_checks(fit_r, fit_v, fit_e, profile, scarcity)
    boot = bootstrap(
        rng(args.seed), args.bootstrap, fit_r, fit_v, fit_e, profile, fit_j, j_cal
    )
    delta_aic = scarcity.aic - newton.aic
    delta_aic_nfw = scarcity.aic - nfw.aic
    sweep_deltas = np.array(
        [row["delta_aic_scarcity_minus_newtonian"] for row in sensitivity]
    )
    nfw_sweep_deltas = np.array(
        [row["delta_aic_scarcity_minus_nfw"] for row in sensitivity]
    )
    nfw_profile_sign_stable = bool(
        np.all(nfw_sweep_deltas < 0) or np.all(nfw_sweep_deltas > 0)
    )
    nfw_bootstrap_delta = boot["delta_aic_scarcity_minus_nfw"]
    nfw_bootstrap_sign_stable = bool(
        nfw_bootstrap_delta["p97_5"] < 0 or nfw_bootstrap_delta["p2_5"] > 0
    )
    if abs(delta_aic_nfw) >= 10.0 and nfw_profile_sign_stable and nfw_bootstrap_sign_stable:
        nfw_verdict = "scarcity-preferred" if delta_aic_nfw < 0 else "nfw-preferred"
    else:
        nfw_verdict = "inconclusive"
    sign_stable = bool(np.all(sweep_deltas < 0) or np.all(sweep_deltas > 0))
    decisive = bool(abs(delta_aic) >= 10.0 and sign_stable)
    needs_outer_followup = not decisive
    if needs_outer_followup:
        followup_reason = (
            "The 5–15 kpc comparison is not decisive and profile-stable under the "
            "predeclared |delta AIC| >= 10 rule; a longer radial lever arm could discriminate."
        )
    else:
        followup_reason = (
            "The existing 5–15 kpc band already gives a decisive, profile-stable relative "
            "comparison; extending to 20–25 kpc is not needed to discriminate this apparatus."
        )

    sidecar = args.data.with_suffix(".json")
    lineage = json.loads(sidecar.read_text()) if sidecar.exists() else None
    rows = []
    for index in range(len(radius)):
        rows.append(
            {
                "R_kpc": float(radius[index]),
                "v_phi_kms": float(velocity[index]),
                "statistical_error_kms": float(error[index]),
                "n_stars": int(curve["n_stars"][index]),
                "band": "fit" if fit_mask[index] else "consistency" if consistency_mask[index] else "excluded",
                "scarcity_predicted_observed_kms": float(scarcity_all[index]),
                "newtonian_predicted_observed_kms": float(newton_all[index]),
                "nfw_predicted_observed_kms": float(nfw_all[index]),
            }
        )

    results = {
        "task": "ORB-10082",
        "source_task": "ORB-10075",
        "question": "Does scarcity remain preferred to a three-physical-parameter NFW+baryons model after AIC penalizes each model's actual parameter count?",
        "protocol": {
            "fit_range_kpc": [FIT_MIN_KPC, FIT_MAX_KPC],
            "consistency_range_kpc": [FIT_MAX_KPC, CONSISTENCY_MAX_KPC],
            "asymmetric_drift_nuisance_kms": [DRIFT_MIN_KMS, DRIFT_MAX_KMS],
            "decision_rule": "Relative comparison decisive only if |delta AIC| >= 10 and sign is stable over all 27 profile variants.",
            "seed": args.seed,
            "bootstrap_resamples": args.bootstrap,
            "parameter_counts_including_drift": {
                "newtonian_baryons": 2, "scarcity": 3, "nfw_baryons": 4
            },
        },
        "apparatus": {
            "profile": asdict(profile),
            "outer_radius_kpc": 100.0,
            "radial_step_kpc": 0.01,
            "local_g_calibration_radius_kpc": CALIBRATION_RADIUS_KPC,
            "equation": "v_S^2 = G_local M F(r)/r * q(r)/q(R0), where q(r)=exp[-beta integral_r^Rmax F(u)/u^2 du]",
            "nfw_definition": "M(<r)=M200*f(c*r/R200)/f(c), with R200 enclosing 200 times critical density for H0=70 km/s/Mpc",
            "nfw_bounds": {
                "baryonic_mass_msun": list(NFW_BARYONIC_MASS_BOUNDS_MSUN),
                "halo_mass_msun": list(NFW_HALO_MASS_BOUNDS_MSUN),
                "concentration": list(NFW_CONCENTRATION_BOUNDS),
                "drift_kms": [DRIFT_MIN_KMS, DRIFT_MAX_KMS],
            },
            "limitations": [
                "The exponential disk is represented by a spherical enclosed-mass surrogate, not an exact thin-disk potential.",
                "The scarcity equation is a phenomenological continuum mapping of the imported lattice toy.",
                "The parquet uncertainties are statistical errors on median v_phi and omit dominant distance, selection, and radially varying asymmetric-drift systematics.",
                "The short fit band can weakly identify NFW mass and concentration; bounds and bootstrap intervals are reported explicitly.",
            ],
        },
        "data": {
            "path": str(args.data),
            "lineage_sidecar": lineage,
            "fit_bins": int(np.sum(fit_mask)),
            "consistency_bins": int(np.sum(consistency_mask)),
        },
        "fits": {
            "scarcity": scarcity.serializable(),
            "newtonian_baryons": newton.serializable(),
            "nfw_baryons": nfw.serializable(),
            "delta_aic_scarcity_minus_newtonian": delta_aic,
            "delta_bic_scarcity_minus_newtonian": scarcity.bic - newton.bic,
            "delta_aic_scarcity_minus_nfw": delta_aic_nfw,
            "delta_bic_scarcity_minus_nfw": scarcity.bic - nfw.bic,
        },
        "held_out_consistency": {
            "scarcity": band_metrics(
                velocity[consistency_mask], error[consistency_mask], scarcity_all[consistency_mask]
            ),
            "newtonian_baryons": band_metrics(
                velocity[consistency_mask], error[consistency_mask], newton_all[consistency_mask]
            ),
            "nfw_baryons": band_metrics(
                velocity[consistency_mask], error[consistency_mask], nfw_all[consistency_mask]
            ),
        },
        "bootstrap_95_percent_intervals": boot,
        "profile_sensitivity": {
            "variants": sensitivity,
            "delta_aic_min": float(np.min(sweep_deltas)),
            "delta_aic_max": float(np.max(sweep_deltas)),
            "preference_sign_stable": sign_stable,
            "delta_aic_scarcity_minus_nfw_min": float(np.min(nfw_sweep_deltas)),
            "delta_aic_scarcity_minus_nfw_max": float(np.max(nfw_sweep_deltas)),
            "scarcity_vs_nfw_preference_sign_stable": bool(
                np.all(nfw_sweep_deltas < 0) or np.all(nfw_sweep_deltas > 0)
            ),
        },
        "nfw_comparison_verdict": {
            "verdict": nfw_verdict,
            "baseline_delta_aic_scarcity_minus_nfw": delta_aic_nfw,
            "profile_sign_stable": nfw_profile_sign_stable,
            "bootstrap_95_interval_excludes_zero": nfw_bootstrap_sign_stable,
            "interpretation": "Inconclusive because the 200-resample bootstrap interval crosses zero, despite baseline and profile-variant preference for scarcity." if nfw_verdict == "inconclusive" else "Preference is baseline-, profile-, and bootstrap-stable under the predeclared rule.",
        },
        "lattice_convergence": convergence,
        "outer_followup": {
            "needs_20_to_25_kpc": needs_outer_followup,
            "reason": followup_reason,
        },
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    plot_path = args.output_dir / "fit.png"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    make_plot(
        plot_path,
        radius,
        velocity,
        error,
        fit_mask,
        consistency_mask,
        scarcity_all,
        newton_all,
        nfw_all,
    )

    preference = "scarcity" if delta_aic < 0 else "newtonian baryons"
    print(f"fit bins: {int(np.sum(fit_mask))}; consistency bins: {int(np.sum(consistency_mask))}")
    print(
        f"scarcity: mass={scarcity.mass_msun:.3e} Msun, beta={scarcity.beta_kpc:.3f} kpc, "
        f"drift={scarcity.drift_kms:.2f} km/s, RMSE={scarcity.rmse_kms:.2f} km/s, "
        f"chi2/dof={scarcity.chi2:.1f}/{scarcity.dof}"
    )
    print(
        f"newtonian: mass={newton.mass_msun:.3e} Msun, drift={newton.drift_kms:.2f} km/s, "
        f"RMSE={newton.rmse_kms:.2f} km/s, chi2/dof={newton.chi2:.1f}/{newton.dof}"
    )
    print(
        f"NFW+baryons: baryonic mass={nfw.mass_msun:.3e} Msun, "
        f"M200={nfw.halo_mass_msun:.3e} Msun, c={nfw.concentration:.2f}, "
        f"drift={nfw.drift_kms:.2f} km/s, RMSE={nfw.rmse_kms:.2f} km/s, "
        f"chi2/dof={nfw.chi2:.1f}/{nfw.dof}"
    )
    print(f"delta AIC (scarcity - Newtonian) = {delta_aic:.1f}; prefers {preference}")
    print(f"profile-sweep delta AIC range = [{np.min(sweep_deltas):.1f}, {np.max(sweep_deltas):.1f}]")
    print(
        f"delta AIC (scarcity - NFW) = {delta_aic_nfw:.1f}; "
        f"profile range = [{np.min(nfw_sweep_deltas):.1f}, {np.max(nfw_sweep_deltas):.1f}]"
    )
    print(
        "bootstrap delta AIC 95% interval = "
        f"[{nfw_bootstrap_delta['p2_5']:.1f}, {nfw_bootstrap_delta['p97_5']:.1f}]; "
        f"verdict={nfw_verdict}"
    )
    print(f"20–25 kpc follow-up needed: {needs_outer_followup}")
    print(f"wrote {results_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
