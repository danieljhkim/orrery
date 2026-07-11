# Scarcity rotation-curve fit

## Question

Does scarcity remain preferred on Tycho's 5–15 kpc Gaia DR3 median-`v_phi` curve when compared
with a standard NFW+baryons model whose baryonic scale, halo mass, and concentration are all
free, after AIC penalizes each model's actual parameter count?

The protocol keeps ORB-10077's 5–15 kpc fit band, untouched 15–18.75 kpc consistency band,
seed 42, 200-resample bootstrap, 27 profile variants, and common 3–10 km/s drift nuisance.
Newtonian baryons has two counted parameters, scarcity three, and NFW+baryons four. A standard-
halo preference is called stable only when the baseline, profile sweep, and bootstrap interval
agree.

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
