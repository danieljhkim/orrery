# Solar-system n-body simulator

The browser entry remains the imported interactive nine-body canvas simulator. ORB-10093 adds
an independent precision-export lane that follows Astrolabe's load-bearing baseline exchange
contract.

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
