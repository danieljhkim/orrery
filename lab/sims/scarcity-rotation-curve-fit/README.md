# Scarcity rotation-curve fit

## Question

Does one radial scarcity normalization improve the 5–15 kpc shape of Tycho's Gaia DR3
median-`v_phi` curve over the nested Newtonian-baryons control, when both fits use the same
3–10 km/s asymmetric-drift nuisance?

The protocol was declared before the fit: use 5–15 kpc for fitting, hold 15–18.75 kpc out for
consistency, and call the relative result decisive only if `|delta AIC| >= 10` with the same sign
across all 27 mass-profile variants.

## Reproduce

From the Orrery root, with Astrolabe's delivered parquet present in the sibling checkout:

```bash
uv run lab/sims/scarcity-rotation-curve-fit/main.py
```

The run is offline with respect to data. It reads
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

## Result

| Metric (5–15 kpc) | Scarcity | Newtonian baryons |
|---|---:|---:|
| Mass | 1.204e11 Msun | 1.224e11 Msun |
| Scarcity beta | 5.25 kpc | 0 (nested control) |
| Drift nuisance | 3.0 km/s | 3.0 km/s |
| RMSE | 2.70 km/s | 18.40 km/s |
| chi-square / dof | 2458.8 / 17 | 102877.0 / 18 |
| AIC | 2464.8 | 102881.0 |

`delta AIC = AIC_scarcity - AIC_Newtonian = -100416.2`; the 200-resample bootstrap interval is
[-134303, -59098]. Across the 27 mass-profile variants it ranges from -158602 to -51749, so the
relative preference is decisive and profile-stable under the predeclared rule.

The held-out 15–18.75 kpc band has scarcity RMSE 5.26 km/s versus 31.16 km/s for Newtonian
baryons. Scarcity underpredicts the held-out points by 4.86 km/s on average.

## What the apparatus does not establish

- The absolute scarcity fit fails against the quoted statistical errors: reduced chi-square is
  144.6. Those errors omit dominant distance, selection, and radially varying asymmetric-drift
  systematics, but the failure must not be hidden behind the 2.70 km/s RMSE.
- The drift nuisance lands on its 3 km/s lower bound and is weakly identified by bootstrap. A
  constant drift term is only a controlled bias treatment, not a physical population model.
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
