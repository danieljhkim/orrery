# Solar-system n-body simulator

The browser entry remains the imported interactive nine-body canvas simulator. ORB-10093 adds
an independent precision-export lane that follows Astrolabe's load-bearing baseline exchange
contract.

## Fitted scarcity-gradient falsifier (ORB-10097)

From the Orrery root, with the same local ephemerides, baseline export, and Astrolabe
`derived/<planet>_newtonian_residuals` datasets present:

```bash
uv run lab/sims/solar-system-nbody/integrate_scarcity.py
```

The experiment concurrently integrates pure Newtonian and multiplicative-scarcity systems on
ORB-10093's exact initial conditions, frame, 366 epochs, and DOP853 tolerances. For each planet,
the multiplicative reading scales the full heliocentric Newtonian acceleration by

```text
q(r_gal)/q(R0) = exp[(beta F(R0)/R0²) (r_gal-R0)]
```

at AU scale. It imports `beta=5.25 kpc`, `R0=8.25 kpc`, and the ORB-10077/10082 profile
(`disk scale=2.6 kpc`, `bulge fraction=0.2`, `bulge scale=0.7 kpc`) unchanged. No parameter is
fitted. The primary fixed ICRF axis is galactocentric outward, opposite the conventional Galactic
center direction at RA `266.4051°`, Dec `−28.936175°`. Six cardinal ICRF axes and `R0=8.122 kpc`
provide sensitivity checks. The local exponential drops terms below `1e-15` in `ln(q)` across
Neptune's orbit.

| Planet | Multiplicative RMS (AU) | Measured omission floor (AU) | Ratio | Primary result |
|---|---:|---:|---:|---|
| Mercury | 3.45e-8 | 1.56e-5 | 0.0022 | below |
| Venus | 8.14e-9 | 3.54e-6 | 0.0023 | below |
| Earth | 9.18e-9 | 2.49e-6 | 0.0037 | below |
| Mars | 2.67e-8 | 8.40e-7 | 0.0318 | below |
| Jupiter | 3.06e-8 | 1.84e-7 | 0.167 | below |
| Saturn | 3.17e-8 | 4.25e-8 | 0.744 | below |
| Uranus | 7.72e-9 | 4.08e-9 | 1.895 | **above** |
| Neptune | 1.92e-9 | 1.75e-8 | 0.110 | below |

Uranus is also above its maximum-residual floor (`2.84×`) in the primary orientation. Its RMS
signature spans `2.71e-9–1.23e-8 AU` across orientations, so the above-floor outcome is
orientation-dependent; every other planet remains below its RMS floor throughout the cardinal
axis sweep. Changing `R0` to `8.122 kpc` raises signatures by about 2.5% without changing the
primary classifications.

The screened reading is the explicit zero control: `q=1` leaves the Newtonian equations
unchanged and produces exactly zero scarcity-minus-Newtonian displacement. Halving the maximum
step from two to one day changes any signature coordinate by at most `1.60e-12 AU`. The
concurrent Newtonian trajectory agrees with the frozen ORB-10093 export within `4.47e-10 AU` per
coordinate. The result tests the multiplicative interpretation only; choosing between that and
local screening belongs to Kepler's theory reconciliation.

## Reproduce the baseline

From the Orrery root, with Astrolabe's eight local `ephemeris/*_2016_2026` datasets present:

```bash
uv run lab/sims/solar-system-nbody/export_baseline.py
```

This rewrites `baseline/newtonian_2016_2026.parquet` and `baseline/summary.json`. The combined
parquet contains all eight planetary-system targets on the exact 366-epoch JD TDB grid, in AU
and AU/day. The integration is inertial; positions and velocities become heliocentric ICRF only
when the Sun's instantaneous state is subtracted at output.

The script uses JPL DE440 planetary-system GMs and the first Horizons epoch as its matched
initial state. SciPy DOP853 runs with `rtol=1e-12`, `atol=1e-14`, and a predeclared two-day
maximum step. Relativistic corrections, asteroids, solar oblateness/mass loss, and individually
resolved moons are omitted deliberately. `summary.json` reports the resulting residuals without
tuning them away.

## Untuned result

The exact Astrolabe residual validator reports Mercury RMS `1.5623e-5 AU` and maximum
`3.5245e-5 AU`; the other planets are smaller, down to Uranus RMS `4.08e-9 AU`. Relative energy
drift is `7.28e-14`, and halving the maximum integrator step from two days to one changes any
exported coordinate by at most `2.23e-9 AU`. The residuals are therefore dominated by the
baseline's omitted physics rather than the chosen numerical step. No parameter was tuned after
seeing them.
