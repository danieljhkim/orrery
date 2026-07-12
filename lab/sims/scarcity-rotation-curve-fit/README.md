# Scarcity rotation-curve fit

## Current question: Yukawa shape control (ORB-10167)

Can the frozen 5–15 kpc constant-drift protocol distinguish scarcity's 1/r-tail
saturation from the exponential saturation of the canonical Sanders-form Yukawa
force? The current catalog entry runs:

```bash
uv run lab/sims/scarcity-rotation-curve-fit/main.py
```

The Yukawa competitor applies the point-source force factor
`1 + alpha (1 + r/xi) exp(-r/xi)` to the same spherical baryonic surrogate and
fits `(M, alpha, xi, drift)`. The baseline lands at `alpha = -0.679` and
`xi = 4.64 kpc`, close to the analytic `(-0.68, 5.25 kpc)` expectation. Its
fit RMSE is `2.91 km/s` versus scarcity's `2.70 km/s`; the held-out RMSE values
are `5.52` and `5.26 km/s`, respectively, and both have positive mean residuals
(`+5.11` and `+4.86 km/s`).

The point estimate prefers scarcity, with
`delta AIC = AIC_scarcity - AIC_Yukawa = -369.6`, and all 27 profile variants
do too (`[-479.7, -138.9]`). The 200-resample bootstrap interval is
`[-503.9, +1051.9]`, however, so the declared verdict is **shape-degenerate**:
this band does not robustly choose exponential saturation over a 1/r tail.

The potential as requested has three free physical parameters `(M, alpha, xi)`,
not the same two as scarcity's `(M, beta)`. With the shared drift, AIC therefore
counts four Yukawa parameters versus three scarcity parameters. Hiding the mass
normalization or fixing it from the scarcity fit would make the apparent
equal-complexity comparison data-dependent. The other principal limitation is
that this is a point-source multiplier on a spherical enclosed-mass surrogate,
not a Yukawa convolution of a thin disk.

## Radial asymmetric drift (ORB-10083)

Does a physically motivated radial asymmetric-drift nuisance remove the coherent outer residual,
or does the selected gravity curve still carry it? That extension runs:

```bash
uv run lab/sims/scarcity-rotation-curve-fit/radial_drift.py
```

This preserves ORB-10082's constant-drift capture in `assets/results.json` and `assets/fit.png`,
then writes `assets/radial-drift-results.json` and `assets/radial-drift-fit.png`. Both protocols use
the same 5–15 kpc fit band, untouched 15–18.75 kpc band, seed 42, 200 bootstrap resamples, and 27
mass-profile variants.

The replacement nuisance is the axisymmetric radial Jeans surrogate

```text
v_c² - <v_phi>² = sigma_R(R)² [R/h_nu + 2R/h_sigma - 1/2]
sigma_R(R) = sigma_R0 exp[-(R-R0)/h_sigma]
```

with fixed `R0=8.25 kpc`, tracer scale `h_nu=2.6 kpc`, and epicycle anisotropy
`sigma_phi²/sigma_R²=1/2`. Each gravity model independently fits the same two nuisance parameters
`sigma_R0` and `h_sigma` with identical bounds. These are a smooth physical surrogate, not a
population measurement: the delivered mixed sample has no per-bin dispersions or tracer-density
profile with which to fix them.

### Radial-drift result

| Metric (5–15 kpc) | Scarcity | NFW+baryons | Newtonian baryons |
|---|---:|---:|---:|
| Baryonic mass | 1.717e11 Msun | 1.664e10 Msun | 1.590e11 Msun |
| Shape / halo | beta = 5.57 kpc | M200 = 6.595e11 Msun; c = 40.0 | none |
| Drift parameters | sigma_R0 = 80.0 km/s; h_sigma = 16.09 kpc | 45.88 km/s; 30.0 kpc | 55.59 km/s; 5.17 kpc |
| RMSE | 2.41 km/s | 3.73 km/s | 3.25 km/s |
| chi-square / dof | 2018.4 / 16 | 4245.4 / 15 | 3420.9 / 17 |
| Held-out RMSE | 2.29 km/s | 5.79 km/s | 7.66 km/s |
| Held-out mean data − model | +0.47 km/s | −5.07 km/s | +7.32 km/s |

For scarcity, the radial nuisance reduces the constant-drift held-out mean residual from
`+4.86` to `+0.47 km/s` and RMSE from `5.26` to `2.29 km/s`; it is carrying most of that model's
former coherent outer residual. It does not erase gravity-model dependence: Newtonian baryons
still underpredict by `+7.32 km/s`, while NFW overpredicts by `−5.07 km/s` and is slightly worse
than its constant-drift result. No gravity parameter was tuned outside the declared refit.

The improvement is not a clean physical identification. Scarcity's `sigma_R0` sits at its
`80 km/s` upper bound in the baseline and every bootstrap percentile, and its held-out mean
bootstrap interval is `[-2.26,+3.95] km/s`. NFW also pins concentration and `h_sigma` to their
upper bounds. All fits still have very large reduced chi-square against the statistical-only
errors. The result therefore says a radial population nuisance can absorb scarcity's coherent
outer residual, not that this particular drift profile has been measured.

## Frozen constant-drift question (ORB-10082)

Does scarcity remain preferred on Tycho's 5–15 kpc Gaia DR3 median-`v_phi` curve when compared
with a standard NFW+baryons model whose baryonic scale, halo mass, and concentration are all
free, after AIC penalizes each model's actual parameter count?

The protocol keeps ORB-10077's 5–15 kpc fit band, untouched 15–18.75 kpc consistency band,
seed 42, 200-resample bootstrap, 27 profile variants, and common 3–10 km/s drift nuisance.
Newtonian baryons has two counted parameters, scarcity three, and NFW+baryons four. A standard-
halo preference is called stable only when the baseline, profile sweep, and bootstrap interval
agree.

## Reproduce the frozen constant-drift baseline

From the Orrery root, with Astrolabe's delivered parquet present in the sibling checkout:

```bash
uv run lab/sims/scarcity-rotation-curve-fit/main.py
```

That run is offline with respect to data. It reads
`../astrolabe/data/processed/derived/mw_rotation_curve.parquet` and its JSON lineage sidecar,
uses seed 42 for 200 bootstrap resamples, and rewrites `assets/results.json` plus
`assets/fit.png` deterministically.

## Apparatus

The mass profile is the same spherical enclosed-mass surrogate used by the imported
`rotation-curve-distributed-mass` demo: an exponential disk plus Hernquist bulge. The baseline
geometry is disk scale 2.6 kpc, bulge fraction 0.2, and bulge scale 0.7 kpc. Geometry is fixed in
the baseline fit and swept over 27 reasonable variants afterward.

```text
v_N^2(r) = G_local M F(r) / r
q(r) = exp[-beta integral_r^Rmax F(u)/u^2 du]
v_S^2(r) = v_N^2(r) q(r) / q(R0),  R0 = 8.25 kpc
```

The `q(r)/q(R0)` ratio carries forward the original demo's local calibration of Newton's
constant. `beta = 0` is exactly the Newtonian control. Each model fits one common baryonic mass
scale and the same bounded drift nuisance; scarcity alone adds non-negative `beta`.

The NFW model adds a spherical halo with free `M200` and concentration `c`, where `R200`
encloses 200 times the critical density for `H0 = 70 km/s/Mpc`. Its baryonic mass scale remains
free. Bounds are 1e9–3e11 Msun for baryons, 1e10–3e13 Msun for `M200`, `1 <= c <= 40`, and the
same 3–10 km/s drift used by every model.

## Result

| Metric (5–15 kpc) | Scarcity | NFW+baryons | Newtonian baryons |
|---|---:|---:|---:|
| Baryonic mass | 1.204e11 Msun | 3.136e10 Msun | 1.224e11 Msun |
| Shape / halo | beta = 5.25 kpc | M200 = 5.314e11 Msun; c = 37.30 | none |
| Drift nuisance | 3.0 km/s | 10.0 km/s | 3.0 km/s |
| RMSE | 2.70 km/s | 3.66 km/s | 18.40 km/s |
| chi-square / dof | 2458.8 / 17 | 4120.2 / 16 | 102877.0 / 18 |
| AIC | 2464.8 | 4128.2 | 102881.0 |

For the standard-halo test, `delta AIC = AIC_scarcity - AIC_NFW = -1663.4`; all 27 profile
variants also prefer scarcity, spanning -2293.1 to -847.1. However, the 200-resample bootstrap
interval is **[-4129.3, +1102.2]**, which crosses zero. The honest verdict is therefore
**inconclusive**, not a scarcity win. NFW's drift is pinned at its 10 km/s upper bound and its
high concentration is weakly identified over this short radial band.

The untouched 15–18.75 kpc band has NFW RMSE 4.99 km/s versus scarcity's 5.26 km/s and the
nested control's 31.16 km/s. NFW overpredicts by 4.26 km/s on average; scarcity underpredicts by
4.86 km/s. The held-out comparison slightly favors NFW even though the baseline in-band AIC
favors scarcity.

## What the apparatus does not establish

- The absolute scarcity fit fails against the quoted statistical errors: reduced chi-square is
  144.6. Those errors omit dominant distance, selection, and radially varying asymmetric-drift
  systematics, but the failure must not be hidden behind the 2.70 km/s RMSE.
- The drift nuisance lands on its 3 km/s lower bound and is weakly identified by bootstrap. A
  constant drift term is only a controlled bias treatment, not a physical population model.
- NFW's drift lands on the opposite 10 km/s bound in every bootstrap percentile, while its
  concentration clusters near the high end of the allowed range. The fit does not cleanly
  identify a conventional halo over 5–15 kpc.
- The disk force is a spherical surrogate, not an exact thin-disk potential. The fitted mass is
  therefore an apparatus parameter until checked against an independently constrained baryonic
  model.
- AIC here is a descriptive comparison under the delivered statistical-error likelihood; the
  missing correlated systematics mean its enormous magnitude is not a calibrated probability
  that the scarcity idea is true.
- This result says the added radial shape term supplies something the nested control lacks. It
  does not show that the phenomenological scarcity equation is a law of nature.

## Numerical checks and follow-up

Changing lattice step from 0.02 to 0.005 kpc and outer radius from 50 to 200 kpc moves the fitted
prediction by at most 0.00014 km/s. Profile sweeps keep scarcity RMSE within 2.55–2.77 km/s and
`beta` within 4.05–6.20 kpc.

The current 5–15 kpc band already discriminates this apparatus from its nested control, so this
experiment does **not** request Tycho's proposed 20–25 kpc/APOGEE follow-up. The more immediate
work is independent baryonic-profile and asymmetric-drift validation. Kepler reconciliation is
tracked separately in ORB-10080.
