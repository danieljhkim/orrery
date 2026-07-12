# Lattice photon propagation

## Declared test

This experiment tests the literal static-lattice photon rule from ORB-10156:
when a central well depletes addressable points and a signal advances one
point-to-point hop per tick, does the mechanic itself produce defocusing and a
transit-time advance, or can finite hops rescue the observed signs?

Three observables are declared before the run:

1. the signed exit angle of a ray launched parallel to and above the well;
2. the fitted power of the effective-index perturbation and direct comparison
   of `1/r` and `1/r²` candidates;
3. the tick count to a fixed exit plane relative to an undepleted lattice.

## Microscopic apparatus

`n(r)` is the local **linear density of addressable lattice points along a
ray**, normalized to one far from the well. The central deficit is 0.12 inside
a finite core and `0.12 (r_core/r)²` outside. A signal advances exactly one
local neighbour spacing `1/n(r)` per tick. Its direction follows the local
Huygens normal, integrated with a symmetric kick-drift-kick step whose drift is
one full hop — there is no continuum-sized fractional drift hidden in the
propagation rule.

The primary core spans 32 far-field hops, the ray passes at three core radii,
and the exit planes are at ±15 core radii. Geometrically similar runs with
cores of 16 and 64 hops test the effect of finite lattice spacing. The profile
is also fit over 2–12 core radii. The rule uses linear point density because
nearest-neighbour spacing is the photon-facing quantity; interpreting a
two-dimensional areal density would replace `n` by its square root in the weak
field, halving amplitudes without changing any sign or the `1/r²` power.

Run:

```text
uv run lab/sims/lattice-photon-propagation/main.py
```

The command rewrites `assets/results.json` and `assets/propagation.png`.
`results.json` is the authoritative full-precision report and the run aborts
if the radial-law or sign invariants fail.

## Scope boundary

This is the static standing-density branch described by ORB-10158. It does not
model a moving medium. A free-fall lattice flow with a universal local wave
speed is an acoustic-metric completion with an additional velocity field, not
a discrete correction to the one-hop static-density rule tested here.

## Result

The literal static rule confirms all three signs in the analytic derivation.
At impact parameter `3 r_core`, the primary 32-hop-core ray exits deflected
**away** by `0.0405826 rad` (`2.32521°`) and displaced outward by
`0.611518 r_core`. It reaches the exit plane `3.08111` ticks early, an advance
of `0.0962846` tick per core radius relative to the undepleted lattice.

The fitted effective-index perturbation is

```text
1 - n(r) = 0.12 (r_core/r)^1.999999999999997
```

over `2–12 r_core`. The relative RMSE is `6.17×10⁻¹⁵`; the direct `1/r²`
candidate has relative RMSE `3.85×10⁻¹⁵`, versus `0.4225` for `1/r`. Thus the
profile is the local per-shell `1/r²` law, not a PPN-like `1/r` potential.

The 16-, 32-, and 64-hop core runs give away-deflection angles of
`2.3252146°`, `2.3252120°`, and `2.3252111°`; their normalized advances are
`0.09628477`, `0.09628464`, and `0.09628456` tick per core radius. Finite hops
converge without a sign reversal. For the declared static-density mechanic,
the discrete apparatus therefore supports defocusing and a Shapiro-like
**advance**, not the observed focusing and delay. Theory-ledger reconciliation
belongs to kepler.
