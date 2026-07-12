# SPARC scarcity universality test

This apparatus confronts the Milky-Way-fitted fixed scarcity length with the
full size range of the quality-cut SPARC external-galaxy sample. It consumes
Astrolabe's ORB-10168 catalog and runs offline:

```bash
uv run lab/sims/sparc-scarcity-universality/main.py
```

The model reconstructs the standard SPARC baryonic curve with disk
mass-to-light ratio 0.5, bulge ratio 0.7, and signed component velocities. It
fits one positive baryonic normalization per galaxy and no drift term. The
global model counts `N_gal + 1` parameters; the per-galaxy-beta model counts
`2 N_gal`. The additive normalization of the scarcity integral is absorbed by
the per-galaxy amplitude, avoiding an arbitrary external-galaxy calibration
radius.

Run the sim to refresh `assets/results.json` and `assets/diagnostics.png`.
The result section below is filled from that deterministic capture.

## Result

The quality and inclination cuts retain 149 galaxies and 3150 rotation-curve
points. The one-global-length fit lands at `beta = 5.526 kpc`, close to the
Milky Way value of 5.25 kpc, but that numerical coincidence does not survive
the universality diagnostics:

- Allowing `beta_i` lowers AIC by 51,723 even after paying for 148 additional
  shape parameters. The fitted distribution has median 3.07 kpc and 16th–84th
  percentiles 0–8.08 kpc; the constant-beta heterogeneity test gives
  `chi2 = 48,362` for 121 degrees of freedom (`p` underflows to zero).
- `beta_i` correlates positively with disk scale length
  (`Spearman rho = 0.343`, `p = 1.86e-5`). It also correlates with
  characteristic baryonic acceleration (`rho = 0.401`, `p = 4.10e-7`), so
  those covarying galaxy properties are not cleanly separable here.
- The global fit's residual strength correlates with both size
  (`rho = -0.348`, `p = 1.35e-5`) and acceleration (`rho = -0.252`,
  `p = 0.00192`). A single length therefore leaves organized residuals.
- The cheap MOND control fits one global `a0 = 1.406e-10 m/s^2` with the same
  149 normalizations. Its AIC is 29,198, versus 106,619 for global scarcity.

The predeclared kill condition is met: fitted scarcity lengths track galaxy
size at high confidence and are decisively inconsistent with a constant.
Within this apparatus, the fixed-beta scarcity form is refuted as a universal
law despite the global optimum lying near the Milky Way fit.

## Limitations

- `F(r) = r v_bar^2 / max(r v_bar^2)` is the same spherical enclosed-mass
  surrogate used by the Milky Way apparatus, applied to SPARC component
  curves. A thin-disk field is not literally a spherical enclosed mass.
- Published statistical velocity errors do not include all correlated
  distance, inclination, or mass-to-light systematics; AIC is descriptive
  under this likelihood. Both scarcity fits have unacceptable absolute
  chi-square, so none is an adequate noise-level description.
- The one normalization scales the whole published baryonic template. It is a
  deliberately uniform nuisance policy, not an independent gas/disk/bulge
  population fit.
