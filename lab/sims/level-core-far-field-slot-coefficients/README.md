# Level-core far-field shell stability

ORB-11171 evolves the ORB-11041 fixture in place because the model, G1--G5
definitions, and measured slot question are unchanged; git history preserves
the earlier rung. It imports ORB-10938's evolution module and checks the frozen
stencil at runtime against SHA-256
`aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461`.
Bernoulli speed is imposed nowhere.

Run the authoritative experiment with:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11171 \
  uv run lab/sims/level-core-far-field-slot-coefficients/main.py \
  --check-determinism
```

The `47^3 -> 63^3`, half-width-26, `T=480` feasibility choice improves core
sampling on the corresponding ORB-11041 rungs while preserving all three
winds. The shell ladder is `{3,5,8,11,14,17,20,22,24}`: every old shell is an
exact subset and both added shells are wind dominated at all winds. The command
writes byte-identical canonical JSON to `assets/results.json` and
`runs/2026-09-04-seed-42.json`.

## Gate verdicts

All five unchanged gates pass: settlement/trough control, radial falloff,
lattice-unit slot extraction, per-wind scaling, and the Galilean null. A pass
means the predeclared measurement completed under its resolution rules; it does
not mean every candidate field has a PPN scaling regime. Each gate records an
explicit `PASS`/`KILL` verdict in the result.

## Predeclared answers

1. **Shell stability:** boundary-sensitive. The extended dipole exponents are
   `-1.00`, `-1.13`, and `-1.16`; only the highest-wind value remains within
   ORB-11041's quoted adjacent-rung error. The middle-wind alpha2 lattice proxy
   is `73.1 +/- 14.9`, consistent only after combining its new apparatus error
   with ORB-11041's `52.2 +/- 13.2`. The drift is reported as a boundary
   artifact/instability, not a physics result.
2. **Alpha1 regime:** absent on the converged extended ladder at every wind.
3. **Half-order hazard:** slot-order. The wind-axis dipole of `|w_hat|^2` is the
   ORB-11040 symmetry channel containing `2 u_GP dot delta-v`; it retains a
   converged slot-matching tail at all three winds. A decomposition-dependent
   direct residual is also recorded, but is not substituted for that declared
   arbiter.
4. **Lowest-wind anisotropy:** still has no scaling regime.

These are dimensionless lattice-unit coefficients only. They are neither
physical alpha values nor phenomenological constraints. No principia file is
modified.
