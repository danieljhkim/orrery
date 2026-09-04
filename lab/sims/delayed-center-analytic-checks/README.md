# Delayed-center analytic checks

This deterministic Python fixture catalogs the numerical checks behind
principia's `theory/moving-source-field-consistency/`. It is a new sim rather
than an in-place evolution of `retarded-scarcity-wake`: the existing entry is
an interactive visualization, while this entry is a machine-readable numeric
fixture with explicit ladders and gates. Both remain in the
`retarded-scarcity-wake` family.

Run the authoritative fixture with:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11169 \
  uv run lab/sims/delayed-center-analytic-checks/main.py --check-determinism
```

The command executes all four checks twice, requires byte-identical canonical
JSON, and writes the same payload to `assets/results.json` and the dated
`runs/2026-09-04.json` record.

## Scope

This is a negative control for the already-refuted delayed-center shortcut.
It does not test, constrain, or rescue the live Branch C gate, which would
require an explicit boost-violating substrate equation and tracer coupling.
No principia files are modified by this fixture.

The task text says “eight” claim ids but enumerates seven analytic ids. Its
scope note explicitly names the eighth, `retarded-wake-rescues-parent`, so the
claim map below includes that refuted-wall row while making clear that the
fixture does not independently retest the parent's other debts.

## Predeclared checks and results

All tolerances below were fixed before the tracked run. The full-precision
ladders and tolerance fields in `assets/results.json` are authoritative.

1. **Legendre-not-Laplace — PASS.** Gauss-Legendre orders 16, 32, 64, and 128
   project modes `n=0..6` at `beta_g=0.3`. The fine coefficients reproduce
   `(-beta_g)^n`; every `n>=1` mode exceeds the `1e-5` presence floor, the
   measured maximum coefficient error is `8.88e-14`, and the normalized
   coefficients extracted from the potential are unchanged at radii 2, 5,
   and 11 (zero spread at stored precision). The inferred offset
   has slope `beta_g` to `1e-12`, exposing its linear growth with radius.
   **Kill condition:** all `n>=1` coefficients vanish.
2. **Exterior Poisson dipole — PASS.** A seven-point Cartesian Laplacian on
   spacings 0.4, 0.2, 0.1, 0.05, and 0.025 samples the exterior profile at radius 4
   and `beta_g=0.2`. The fine profile must have relative L2 error below
   `5e-4`, the projected dipole below `5e-4` relative error, and the last
   refinement order above 1.8. Its effective density is
   `rho_eff = laplacian(Phi)/(4 pi G)` and its dipole-moment slope is compared
   with `-2 beta_g M/3`, demonstrating linear divergence with outer radius.
   The tracked run reaches profile relative error `3.20e-4`, dipole relative
   error `3.21e-5`, and observed last-pair order `2.00003`.
   **Kill condition:** the profile converges to zero or does not converge.
3. **Retarded-direction curl — PASS.** Finite-difference curl spacings 0.12,
   0.06, 0.03, 0.015, and 0.0075 are paired with 64, 128, 256, and 512-point line
   integrals. Constant-radius circles must return work below `1e-12`; a polar
   loop spanning radii 3 to 5 and angles 0.35 to 1.4 must remain nonzero,
   quadrature-converged to `2e-5`, and approach constant
   `W/(beta_g GM/R)` across
   `beta_g={0.2,0.1,0.05,0.025,0.0125,0.00625}` to 3%.
   The tracked fine circle work is below `2.78e-17`, the maximum quadrature
   shift is `4.37e-6`, and the last two scaled works differ by 2.13%.
   **Kill condition:** curl or generic work vanishes, circle work persists, or
   beta scaling fails.
4. **2-beta-not-4-beta — PASS.** Direct potential-gradient contrasts on
   `beta_g={0.2,0.1,0.05,0.025,0.0125}` must approach coefficient 2 within
   `5e-4`, stay at least 1.9 away from coefficient 4, and show the expected
   second-order asymptotic correction. **Kill condition:** the declared
   `a=-gradient(Phi)` rule approaches `4 beta_g`.
   At the finest beta, the measured coefficient is `2.00031255`; its
   asymptotic error converges at order `2.00068`.

## Claim map

| Check | Principia claim id | Effect on row as worded |
| --- | --- | --- |
| Legendre | `moving-source-legendre-not-laplace` | Supports: nonzero radius-independent `n>=1` terms are Laplace-forbidden. |
| Legendre | `retarded-wake-delayed-center-dipole` | Supports the delayed-center toy's internal dipole statement, while identifying its non-vacuum pathology. |
| Poisson | `moving-source-poisson-dipole-divergence` | Supports: the exterior effective dipolar density is nonzero and its integrated dipole grows linearly. |
| Curl/work | `moving-source-retarded-direction-curl` | Supports: curl and generic loop work are nonzero and scale at first order in `beta_g`. |
| Curl/work | `retarded-wake-retarded-direction-conservative` | Kills: the alternate retarded-direction rule is not conservative. |
| Contrast | `moving-source-2beta-not-4beta` | Supports: the declared potential-gradient rule tends to `2 beta_g`. |
| Contrast | `retarded-wake-4beta-contrast` | Kills: `4 beta_g` is not reproduced by the declared rule. |
| Scope guard | `retarded-wake-rescues-parent` | Kills as worded: this pathological negative control supplies no rescue; it does not independently retest the parent ledger. |

## Provenance

Implemented for Orbit task `ORB-11169`, with theory context
`../../principia/theory/moving-source-field-consistency/` and almanac context
`15-discussions/26-08/from-galactic-motion-to-the-moving-gravity-medium-problem.md`.
