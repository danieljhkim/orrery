"""Catalog the delayed-center shortcut's four analytic negative controls.

The fixture uses quadrature, finite differences, line integration, and direct
evaluation to test the radius-independent Legendre shape, non-vacuum exterior
Poisson profile, curl of the retarded-direction rule, and the declared
potential rule's 2-beta (not 4-beta) acceleration contrast.  It is a test of
an already-refuted shortcut, not of the live boost-violating Branch C gate.

Usage:
    uv run lab/sims/delayed-center-analytic-checks/main.py
    uv run lab/sims/delayed-center-analytic-checks/main.py --check-determinism
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legval

SLUG = "delayed-center-analytic-checks"
TASK_ID = "ORB-11169"
RUN_DATE = "2026-09-04"
ROOT = Path(__file__).parent
RESULT_PATH = ROOT / "assets" / "results.json"
RUN_PATH = ROOT / "runs" / f"{RUN_DATE}.json"


def delayed_shape(mu: np.ndarray | float, beta: float) -> np.ndarray:
    """Return -R Phi/(GM) for the delayed-center potential."""
    return np.asarray(1.0 + 2.0 * beta * np.asarray(mu) + beta**2) ** -0.5


def potential(points: np.ndarray, beta: float) -> np.ndarray:
    """Return Phi for GM=1 and the lag axis along +x."""
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=-1)
    mu = points[..., 0] / radius
    return -delayed_shape(mu, beta) / radius


def legendre_coefficients(
    beta: float, order: int, maximum_n: int, radius: float = 1.0
) -> np.ndarray:
    mu, weights = leggauss(order)
    points = np.column_stack(
        [radius * mu, radius * np.sqrt(1.0 - mu**2), np.zeros_like(mu)]
    )
    shape = -radius * potential(points, beta)
    coefficients = []
    for n in range(maximum_n + 1):
        basis = np.zeros(n + 1)
        basis[n] = 1.0
        coefficient = (2 * n + 1) * np.sum(weights * shape * legval(mu, basis)) / 2
        coefficients.append(coefficient)
    return np.asarray(coefficients)


def legendre_check() -> dict:
    beta = 0.3
    radii = np.asarray([2.0, 5.0, 11.0])
    orders = [16, 32, 64, 128]
    maximum_n = 6
    expected = np.asarray([(-beta) ** n for n in range(maximum_n + 1)])
    ladder = []
    for order in orders:
        coefficients = legendre_coefficients(beta, order, maximum_n)
        ladder.append(
            {
                "quadrature_order": order,
                "coefficients_n0_through_n6": coefficients.tolist(),
                "maximum_absolute_coefficient_error": float(
                    np.max(np.abs(coefficients - expected))
                ),
            }
        )

    fine = np.asarray(ladder[-1]["coefficients_n0_through_n6"])
    # Extracting from Phi and multiplying by -R gives the same angular shape.
    by_radius = np.stack(
        [legendre_coefficients(beta, orders[-1], maximum_n, radius) for radius in radii]
    )
    maximum_radius_spread = float(np.max(np.ptp(by_radius, axis=0)))
    inferred_offsets = -radii * by_radius[:, 1] / by_radius[:, 0]
    slope, intercept = np.polyfit(radii, inferred_offsets, 1)
    tolerances = {
        "coefficient_absolute": 1.0e-12,
        "n_ge_1_presence": 1.0e-5,
        "radius_shape_absolute": 1.0e-13,
        "dipole_offset_slope_absolute": 1.0e-12,
        "dipole_offset_intercept_absolute": 1.0e-12,
    }
    passed = bool(
        ladder[-1]["maximum_absolute_coefficient_error"]
        <= tolerances["coefficient_absolute"]
        and np.all(np.abs(fine[1:]) >= tolerances["n_ge_1_presence"])
        and maximum_radius_spread <= tolerances["radius_shape_absolute"]
        and abs(slope - beta) <= tolerances["dipole_offset_slope_absolute"]
        and abs(intercept) <= tolerances["dipole_offset_intercept_absolute"]
    )
    return {
        "id": "legendre_not_laplace",
        "claim_ids": [
            "moving-source-legendre-not-laplace",
            "retarded-wake-delayed-center-dipole",
        ],
        "method": "Gauss-Legendre projection of -R Phi/(GM)",
        "parameters": {"beta_g": beta, "radii": radii.tolist(), "maximum_n": maximum_n},
        "tolerances": tolerances,
        "ladder": ladder,
        "fine_measurements": {
            "expected_coefficients_n0_through_n6": expected.tolist(),
            "maximum_radius_spread": maximum_radius_spread,
            "coefficients_by_radius": [
                {
                    "radius": float(radius),
                    "coefficients_n0_through_n6": coefficients.tolist(),
                }
                for radius, coefficients in zip(radii, by_radius, strict=True)
            ],
            "inferred_offset_over_radius": (inferred_offsets / radii).tolist(),
            "inferred_offset_linear_slope": float(slope),
            "inferred_offset_linear_intercept": float(intercept),
            "laplace_forbidden_n_ge_1_present": bool(
                np.all(np.abs(fine[1:]) >= tolerances["n_ge_1_presence"])
            ),
        },
        "predeclared_pass": "n>=1 coefficients are present, their normalized angular shape is radius-independent, and the inferred dipole offset grows linearly with radius",
        "predeclared_kill": "all n>=1 coefficients vanish",
        "kill_condition_triggered": not passed,
        "passed": passed,
        "verdict": "PASS" if passed else "KILL",
        "claim_assessments": {
            "moving-source-legendre-not-laplace": "supports",
            "retarded-wake-delayed-center-dipole": "supports",
        },
    }


def analytic_laplacian(points: np.ndarray, beta: float) -> np.ndarray:
    """Return the exact Cartesian Laplacian of Phi for GM=1."""
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=-1)
    mu = points[..., 0] / radius
    q = 1.0 + 2.0 * beta * mu + beta**2
    angular = 2.0 * beta * mu * q + 3.0 * beta**2 * (1.0 - mu**2)
    return -angular / (radius**3 * q ** 2.5)


def finite_difference_laplacian(points: np.ndarray, beta: float, spacing: float) -> np.ndarray:
    center = potential(points, beta)
    total = np.zeros_like(center)
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = spacing
        total += potential(points + shift, beta) - 2.0 * center + potential(points - shift, beta)
    return total / spacing**2


def poisson_check() -> dict:
    beta = 0.2
    radius = 4.0
    mus = np.linspace(-0.9, 0.9, 19)
    points = np.column_stack(
        [radius * mus, radius * np.sqrt(1.0 - mus**2), np.zeros_like(mus)]
    )
    reference = analytic_laplacian(points, beta)
    spacings = [0.4, 0.2, 0.1, 0.05, 0.025]
    quadrature_mu, quadrature_weights = leggauss(96)
    quadrature_points = np.column_stack(
        [
            radius * quadrature_mu,
            radius * np.sqrt(1.0 - quadrature_mu**2),
            np.zeros_like(quadrature_mu),
        ]
    )
    ladder = []
    for spacing in spacings:
        measured = finite_difference_laplacian(points, beta, spacing)
        relative_l2 = float(np.linalg.norm(measured - reference) / np.linalg.norm(reference))
        projected = finite_difference_laplacian(quadrature_points, beta, spacing)
        laplacian_dipole = float(
            1.5 * np.sum(quadrature_weights * projected * quadrature_mu)
        )
        rho_dipole = laplacian_dipole / (4.0 * np.pi)
        dipole_moment_slope = (4.0 * np.pi / 3.0) * rho_dipole * radius**3
        ladder.append(
            {
                "grid_spacing": spacing,
                "spacing_over_radius": spacing / radius,
                "profile_relative_l2_error": relative_l2,
                "laplacian_dipole_coefficient": laplacian_dipole,
                "effective_density_dipole_coefficient": rho_dipole,
                "effective_dipole_moment_slope": dipole_moment_slope,
            }
        )
    fine = ladder[-1]
    observed_orders = [
        float(np.log(ladder[index]["profile_relative_l2_error"] / ladder[index + 1]["profile_relative_l2_error"]) / np.log(2.0))
        for index in range(len(ladder) - 1)
    ]
    expected_laplacian_dipole = -2.0 * beta / radius**3
    expected_density_dipole = expected_laplacian_dipole / (4.0 * np.pi)
    expected_moment_slope = -2.0 * beta / 3.0
    tolerances = {
        "fine_profile_relative_l2": 5.0e-4,
        "fine_dipole_relative": 5.0e-4,
        "minimum_last_pair_convergence_order": 1.8,
        "nonzero_profile_l2_floor": 1.0e-4,
    }
    dipole_relative = abs(
        fine["laplacian_dipole_coefficient"] / expected_laplacian_dipole - 1.0
    )
    passed = bool(
        fine["profile_relative_l2_error"] <= tolerances["fine_profile_relative_l2"]
        and dipole_relative <= tolerances["fine_dipole_relative"]
        and observed_orders[-1] >= tolerances["minimum_last_pair_convergence_order"]
        and np.linalg.norm(reference) >= tolerances["nonzero_profile_l2_floor"]
    )
    return {
        "id": "poisson_exterior_dipole",
        "claim_ids": ["moving-source-poisson-dipole-divergence"],
        "method": "seven-point Cartesian finite-difference Laplacian on an exterior angular profile",
        "parameters": {"beta_g": beta, "radius": radius, "profile_mu": mus.tolist()},
        "tolerances": tolerances,
        "ladder": ladder,
        "fine_measurements": {
            "analytic_profile": reference.tolist(),
            "finite_difference_profile": finite_difference_laplacian(points, beta, spacings[-1]).tolist(),
            "observed_convergence_orders": observed_orders,
            "expected_laplacian_dipole_coefficient": expected_laplacian_dipole,
            "expected_effective_density_dipole_coefficient": expected_density_dipole,
            "expected_effective_dipole_moment_slope": expected_moment_slope,
            "fine_dipole_relative_error": float(dipole_relative),
            "interpretation": "rho_eff = laplacian(Phi)/(4 pi G); its l=1 moment accumulates linearly with outer radius",
        },
        "predeclared_pass": "the nonzero exterior Laplacian profile and effective dipole density converge at second order under grid refinement",
        "predeclared_kill": "the exterior profile converges to zero or fails to converge",
        "kill_condition_triggered": not passed,
        "passed": passed,
        "verdict": "PASS" if passed else "KILL",
        "claim_assessments": {"moving-source-poisson-dipole-divergence": "supports"},
    }


def retarded_acceleration(points: np.ndarray, beta: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=-1)
    displaced = points.copy()
    displaced[..., 0] += beta * radius
    return -displaced / np.linalg.norm(displaced, axis=-1)[..., None] ** 3


def integrate_segment(points: np.ndarray, derivatives: np.ndarray, beta: float, parameter: np.ndarray) -> float:
    integrand = np.sum(retarded_acceleration(points, beta) * derivatives, axis=1)
    return float(np.trapezoid(integrand, parameter))


def circle_work(beta: float, radius: float, samples: int) -> float:
    theta = np.linspace(0.0, 2.0 * np.pi, samples + 1)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])
    derivatives = np.column_stack([-radius * np.sin(theta), radius * np.cos(theta), np.zeros_like(theta)])
    return integrate_segment(points, derivatives, beta, theta)


def generic_loop_work(beta: float, inner: float, outer: float, theta_a: float, theta_b: float, samples: int) -> float:
    radial_out = np.linspace(inner, outer, samples + 1)
    points_1 = np.column_stack([radial_out * np.cos(theta_a), radial_out * np.sin(theta_a), np.zeros_like(radial_out)])
    deriv_1 = np.tile([np.cos(theta_a), np.sin(theta_a), 0.0], (samples + 1, 1))
    theta_out = np.linspace(theta_a, theta_b, samples + 1)
    points_2 = np.column_stack([outer * np.cos(theta_out), outer * np.sin(theta_out), np.zeros_like(theta_out)])
    deriv_2 = np.column_stack([-outer * np.sin(theta_out), outer * np.cos(theta_out), np.zeros_like(theta_out)])
    radial_in = np.linspace(outer, inner, samples + 1)
    points_3 = np.column_stack([radial_in * np.cos(theta_b), radial_in * np.sin(theta_b), np.zeros_like(radial_in)])
    deriv_3 = np.tile([np.cos(theta_b), np.sin(theta_b), 0.0], (samples + 1, 1))
    theta_in = np.linspace(theta_b, theta_a, samples + 1)
    points_4 = np.column_stack([inner * np.cos(theta_in), inner * np.sin(theta_in), np.zeros_like(theta_in)])
    deriv_4 = np.column_stack([-inner * np.sin(theta_in), inner * np.cos(theta_in), np.zeros_like(theta_in)])
    return sum(
        [
            integrate_segment(points_1, deriv_1, beta, radial_out),
            integrate_segment(points_2, deriv_2, beta, theta_out),
            integrate_segment(points_3, deriv_3, beta, radial_in),
            integrate_segment(points_4, deriv_4, beta, theta_in),
        ]
    )


def curl_z_finite_difference(point: np.ndarray, beta: float, spacing: float) -> float:
    shift_x = np.asarray([spacing, 0.0, 0.0])
    shift_y = np.asarray([0.0, spacing, 0.0])
    d_a_y_dx = (
        retarded_acceleration(point + shift_x, beta)[1]
        - retarded_acceleration(point - shift_x, beta)[1]
    ) / (2.0 * spacing)
    d_a_x_dy = (
        retarded_acceleration(point + shift_y, beta)[0]
        - retarded_acceleration(point - shift_y, beta)[0]
    ) / (2.0 * spacing)
    return float(d_a_y_dx - d_a_x_dy)


def analytic_curl_z(point: np.ndarray, beta: float) -> float:
    radius = float(np.linalg.norm(point))
    cosine = float(point[0] / radius)
    sine = float(point[1] / radius)
    q = 1.0 + 2.0 * beta * cosine + beta**2
    return beta * sine * (1.0 - beta * cosine - 2.0 * beta**2) / (radius**3 * q**2.5)


def curl_check() -> dict:
    betas = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
    samples_ladder = [64, 128, 256, 512]
    inner, outer = 3.0, 5.0
    theta_a, theta_b = 0.35, 1.4
    beta_runs = []
    for beta in betas:
        loop_ladder = []
        for samples in samples_ladder:
            circle = circle_work(beta, 4.0, samples)
            generic = generic_loop_work(beta, inner, outer, theta_a, theta_b, samples)
            loop_ladder.append(
                {
                    "samples_per_segment": samples,
                    "circle_work": circle,
                    "generic_loop_work": generic,
                    "generic_work_over_beta_GM_per_inner_radius": generic * inner / beta,
                }
            )
        beta_runs.append({"beta_g": beta, "quadrature_ladder": loop_ladder})

    point = np.asarray([2.4, 1.8, 0.0])
    curl_spacings = [0.12, 0.06, 0.03, 0.015, 0.0075]
    curl_reference = analytic_curl_z(point, betas[1])
    curl_ladder = []
    for spacing in curl_spacings:
        value = curl_z_finite_difference(point, betas[1], spacing)
        curl_ladder.append(
            {
                "grid_spacing": spacing,
                "curl_z": value,
                "relative_error": abs(value / curl_reference - 1.0),
            }
        )
    fine_generic = np.asarray(
        [run["quadrature_ladder"][-1]["generic_loop_work"] for run in beta_runs]
    )
    fine_scaled = np.asarray(
        [run["quadrature_ladder"][-1]["generic_work_over_beta_GM_per_inner_radius"] for run in beta_runs]
    )
    fine_circles = np.asarray(
        [run["quadrature_ladder"][-1]["circle_work"] for run in beta_runs]
    )
    linear_fit = np.polyfit(np.asarray(betas), fine_generic, 1)
    relative_quadrature_shift = max(
        abs(
            run["quadrature_ladder"][-1]["generic_loop_work"]
            / run["quadrature_ladder"][-2]["generic_loop_work"]
            - 1.0
        )
        for run in beta_runs
    )
    small_beta_scaled_spread = abs(fine_scaled[-1] / fine_scaled[-2] - 1.0)
    tolerances = {
        "circle_work_absolute": 1.0e-12,
        "generic_work_absolute_floor": 1.0e-5,
        "fine_quadrature_relative_shift": 2.0e-5,
        "small_beta_scaled_work_relative_shift": 3.0e-2,
        "fine_curl_relative_error": 5.0e-4,
    }
    passed = bool(
        np.max(np.abs(fine_circles)) <= tolerances["circle_work_absolute"]
        and np.min(np.abs(fine_generic)) >= tolerances["generic_work_absolute_floor"]
        and relative_quadrature_shift <= tolerances["fine_quadrature_relative_shift"]
        and small_beta_scaled_spread <= tolerances["small_beta_scaled_work_relative_shift"]
        and curl_ladder[-1]["relative_error"] <= tolerances["fine_curl_relative_error"]
        and abs(curl_reference) > 0.0
    )
    return {
        "id": "retarded_direction_curl",
        "claim_ids": [
            "moving-source-retarded-direction-curl",
            "retarded-wake-retarded-direction-conservative",
        ],
        "method": "finite-difference curl plus trapezoidal line integrals on constant-radius and radius-angle loops",
        "parameters": {
            "beta_g_ladder": betas,
            "circle_radius": 4.0,
            "generic_loop": {"inner_radius": inner, "outer_radius": outer, "theta_a": theta_a, "theta_b": theta_b},
        },
        "tolerances": tolerances,
        "beta_ladder": beta_runs,
        "curl_ladder": curl_ladder,
        "fine_measurements": {
            "analytic_curl_z_at_beta_0p1": curl_reference,
            "generic_work_linear_fit_slope": float(linear_fit[0]),
            "generic_work_linear_fit_intercept": float(linear_fit[1]),
            "maximum_fine_circle_work_absolute": float(np.max(np.abs(fine_circles))),
            "maximum_fine_quadrature_relative_shift": float(relative_quadrature_shift),
            "small_beta_scaled_work_relative_shift": float(small_beta_scaled_spread),
        },
        "predeclared_pass": "curl is nonzero, constant-radius circle work vanishes, and generic radius-angle loop work converges and scales as beta_g GM/R",
        "predeclared_kill": "curl or generic loop work converges to zero, circle work remains nonzero, or beta_g scaling fails",
        "kill_condition_triggered": not passed,
        "passed": passed,
        "verdict": "PASS" if passed else "KILL",
        "claim_assessments": {
            "moving-source-retarded-direction-curl": "supports",
            "retarded-wake-retarded-direction-conservative": "kills",
        },
    }


def contrast_check() -> dict:
    betas = [0.2, 0.1, 0.05, 0.025, 0.0125]
    ladder = []
    for beta in betas:
        g_lead_over_g0 = 1.0 / (1.0 + beta)
        g_trail_over_g0 = 1.0 / (1.0 - beta)
        contrast = g_trail_over_g0 - g_lead_over_g0
        ladder.append(
            {
                "beta_g": beta,
                "g_lead_over_g0": g_lead_over_g0,
                "g_trail_over_g0": g_trail_over_g0,
                "contrast_over_g0": contrast,
                "contrast_over_beta_g": contrast / beta,
                "relative_error_from_2beta": abs(contrast / (2.0 * beta) - 1.0),
                "relative_error_from_4beta": abs(contrast / (4.0 * beta) - 1.0),
            }
        )
    fine = ladder[-1]
    observed_orders = [
        float(
            np.log(
                ladder[index]["relative_error_from_2beta"]
                / ladder[index + 1]["relative_error_from_2beta"]
            )
            / np.log(2.0)
        )
        for index in range(len(ladder) - 1)
    ]
    tolerances = {
        "fine_contrast_over_beta_absolute_from_2": 5.0e-4,
        "fine_contrast_over_beta_minimum_distance_from_4": 1.9,
        "minimum_last_pair_asymptotic_order": 1.9,
    }
    passed = bool(
        abs(fine["contrast_over_beta_g"] - 2.0)
        <= tolerances["fine_contrast_over_beta_absolute_from_2"]
        and abs(fine["contrast_over_beta_g"] - 4.0)
        >= tolerances["fine_contrast_over_beta_minimum_distance_from_4"]
        and observed_orders[-1] >= tolerances["minimum_last_pair_asymptotic_order"]
    )
    return {
        "id": "two_beta_not_four_beta",
        "claim_ids": [
            "moving-source-2beta-not-4beta",
            "retarded-wake-4beta-contrast",
            "retarded-wake-rescues-parent",
        ],
        "method": "direct evaluation of the radial derivative of Phi on the leading and trailing axes",
        "parameters": {"beta_g_ladder": betas, "force_rule": "a = -gradient(Phi_lag)"},
        "tolerances": tolerances,
        "ladder": ladder,
        "fine_measurements": {"observed_asymptotic_orders": observed_orders},
        "predeclared_pass": "(g_trail-g_lead)/g0 approaches 2 beta_g and remains separated from 4 beta_g",
        "predeclared_kill": "the declared potential-gradient rule approaches 4 beta_g instead",
        "kill_condition_triggered": not passed,
        "passed": passed,
        "verdict": "PASS" if passed else "KILL",
        "claim_assessments": {
            "moving-source-2beta-not-4beta": "supports",
            "retarded-wake-4beta-contrast": "kills",
            "retarded-wake-rescues-parent": "kills; this negative control supplies no rescue and does not retest the parent debts",
        },
    }


def run_checks() -> dict:
    checks = [legendre_check(), poisson_check(), curl_check(), contrast_check()]
    all_claim_ids = sorted({claim for check in checks for claim in check["claim_ids"]})
    return {
        "schema_version": 1,
        "sim": SLUG,
        "task": TASK_ID,
        "run_date": RUN_DATE,
        "model": {
            "potential": "Phi_lag = -GM/[R sqrt(1 + 2 beta_g cos(theta) + beta_g^2)]",
            "lag": "Delta(R) = -w R/c_g",
            "units": "GM = 1",
            "scope": "negative control for the refuted delayed-center shortcut; not the live Branch C gate",
        },
        "claim_ids": all_claim_ids,
        "claim_id_count": len(all_claim_ids),
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
        "overall_verdict": "PASS" if all(check["passed"] for check in checks) else "KILL",
    }


def canonical_bytes(result: dict) -> bytes:
    return (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="execute the complete four-check fixture twice and require byte-identical JSON",
    )
    parser.add_argument("--no-write", action="store_true", help="run checks without updating tracked artifacts")
    args = parser.parse_args()

    result = run_checks()
    payload = canonical_bytes(result)
    if args.check_determinism:
        rerun = canonical_bytes(run_checks())
        if payload != rerun:
            raise RuntimeError("determinism check failed: in-process rerun was not byte-identical")

    if not args.no_write:
        RESULT_PATH.parent.mkdir(exist_ok=True)
        RUN_PATH.parent.mkdir(exist_ok=True)
        RESULT_PATH.write_bytes(payload)
        RUN_PATH.write_bytes(payload)

    for check in result["checks"]:
        print(f"{check['verdict']:4s}  {check['id']}")
    if args.check_determinism:
        print("PASS  byte-identical in-process rerun")
    if not args.no_write:
        print(f"wrote {RESULT_PATH.relative_to(Path.cwd())}")
        print(f"wrote {RUN_PATH.relative_to(Path.cwd())}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
