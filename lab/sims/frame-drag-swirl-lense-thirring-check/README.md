# Frame-drag swirl Lense–Thirring shape check

This deterministic Python fixture checks the imported
`frame-drag-swirl` visualization as it exists. It is a separate catalog entry
because the original is an interactive, hand-drawn 2-D field with no numeric
output, while this fixture records explicit ladders and predeclared gates.

Run the authoritative fixture with:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11172 \
  uv run lab/sims/frame-drag-swirl-lense-thirring-check/main.py --check-determinism
```

The command runs every check twice, requires byte-identical canonical JSON,
and writes the same payload to `results.json` and `runs/2026-09-04.json`.

## Source law, quoted rather than re-derived

The fixture transcribes this complete swirl law verbatim from
`../frame-drag-swirl/index.html`, lines 35–39:

```js
function flow(r){
  if(prof==='flat')return u0*Math.min(r/80,1);
  if(prof==='solid')return u0*r/RMAXV*1.6;
  return u0*80/Math.max(r,24);
}
```

The visualization turns that tangential flow speed into angular velocity at
lines 74–75, also quoted verbatim:

```js
    var om=flow(r)/r;
    phases[i]=phases[i].map(function(ph){return ph+om*1.0;});
```

`main.py` checks both source fragments at runtime before measuring its literal
Python transcription. The UI default is the `flat` profile (line 11), with
`u0=1.2` and `GM=225` set at line 34. The field is imposed directly; there is
no evolution equation that makes a rotating source generate the swirl.

## Comparator and predeclared verdicts

The equatorial comparator requested by ORB-11172 is
`Omega_LT = 2 G J / (c^2 r^3)`: falloff `r^-3`, linearity in source angular
momentum `J`, and reversal of `Omega` when `J` reverses. The toy is only a 2-D
equatorial picture and defines no polar dependence, so it cannot test the
three-dimensional polar/vector structure of the Lense–Thirring field.

All gates and tolerances below are fixed in `main.py`; full-precision ladders
are in `results.json`.

1. **Radial falloff — KILL.** Nested 5, 9, 17, and 33-point log-radius
   ladders span `r=80..320`. The default flat profile measures exponent
   `-1.0 ± 7.11e-15`, not `-3`; the predeclared pass requires
   `-3` within the estimated ladder error. The optional solid and whirlpool
   profiles measure approximately `0` and `-2`, so none of the three selectable
   laws has the Lense–Thirring falloff.
2. **J-scaling surrogate — PASS, with a scope caveat.** At fixed `r=160`, the
   amplitude ladder `u0={0.15,0.3,0.6,1.2,2.4}` measures exponent `1` within
   its ladder error. The predeclared pass is linearity in `u0`. This is only a
   shape result: the toy defines no physical `J`, so `u0` is a free spin
   surrogate rather than a derived source angular momentum.
3. **Spin sign — PASS, algebraic only.** At fixed radius, signed samples
   `u0={-1.2,0,1.2}` satisfy `Omega(-u0)=-Omega(u0)` and `Omega(0)=0`. The
   predeclared kill is any failure of that odd sign structure. The browser
   slider itself exposes only nonnegative `u0`; signed evaluation extends the
   quoted algebra without changing it.

The separate `GM={56.25,112.5,225,450,900}` diagnostic measures exponent
zero: `GM` affects tracer gravity in the browser, but never appears in
`flow(r)`. Thus the toy has the right linear and sign algebra only after its
swirl amplitude is supplied by hand. It has no rotating-source mechanic and
does **not** reproduce the full Lense–Thirring shape.

## Scope and observational context

No magnitude in physical units is claimed. The browser's dimensionless
coordinates and `u0` have no lattice-to-physical normalization, which remains
open family debt. Consequently this fixture does not compare its amplitude to
an experiment; it reports dimensionless shape tests only.

For the physical comparators—Gravity Probe B's frame-dragging drift and the
LAGEOS/LARES measurement—see the sibling principia study
`principia/studies/moving-source-gravity-bounds.md`. This task does not modify
principia. It also does not substitute for a rotating-source level-core
apparatus; spin GEM remains outside the PPN-reduction fixture's scope.

## Provenance

Implemented for Orbit task `ORB-11172`, cross-linked with
`../frame-drag-swirl/`, and kept in the `gravity-as-scarcity` family.
