---
title: "Gravity as the gradient of scarce space"
status: exploratory
families: [gravity-as-scarcity]
almanac: 15-discussions/26-07/gravity-as-scarcity-of-space.md
created: 2026-07-09
updated: 2026-07-09
---

# Gravity as the gradient of scarce space

**The idea.** Gravity is not a force law but the gradient of a stored scalar — "scarcity of
space" — computed on a lattice by counting points in spherical shells around masses. A mass
depletes a finite budget of space; particles roll down the scarcity gradient. Built up over a
discussion thread (see `almanac` link) that repeatedly found the toy model rediscovering real
physics: Gauss's law from budget-division bookkeeping, Yukawa-like cutoffs from budget
exhaustion, MOND-flavored behavior, and frame dragging from adding swirl.

**Current verdict (from the discussion):** *right shape, wrong magnitude* — a respectable
place for a counting argument to land. The model reproduces functional forms, not measured
values.

## Evidence ledger

| Claim | Status | Evidence |
|---|---|---|
| Accumulated per-shell dilution Σ 1/count(k) ∝ 1/r, so "gravity = gradient of scarcity" reproduces Newton's 1/r² | supported | by construction in [scarcity-grid-weight-black-hole](../sims/scarcity-grid-weight-black-hole/), [scarcity-capped-cumulative-field](../sims/scarcity-capped-cumulative-field/); note: storing 1/r² and differentiating gives 1/r³ — the stored field must be potential-like (correction #1 in the thread) |
| Dividing a fixed budget across shells yields inverse-square from pure geometry (Gauss's law analog) | supported | [scarcity-shell-depletion-field](../sims/scarcity-shell-depletion-field/) |
| Subtractive budget gives a hard cutoff radius r ≈ (3T/4π)^⅓ beyond which gravity dies (Yukawa-cartoon) | supported | [scarcity-capped-cumulative-field](../sims/scarcity-capped-cumulative-field/), [scarcity-star-well-headroom](../sims/scarcity-star-well-headroom/) — as a property of the *model*; no claim it matches nature |
| An extended (distributed) mass under the model bends rotation curves toward flat (dark-matter/MOND-adjacent) | mixed | [rotation-curve-distributed-mass](../sims/rotation-curve-distributed-mass/) shows the qualitative shape; magnitudes not fit to any galaxy data — needs a study note on measured rotation curves before this can be called supported |
| Adding swirl to the scarcity field reproduces frame dragging | mixed | [frame-drag-swirl](../sims/frame-drag-swirl/) — right shape; real frame dragging (Lense–Thirring) has specific magnitude/falloff this toy hasn't been checked against (`conjecture — to verify`: study note needed) |
| The scarcity picture is equivalent to weak-field GR's "gradient of time-flow rate" heuristic | untested | asserted by analogy in the thread; needs a worked comparison |

## Open questions

- Can the lattice model be normalized once (one constant) and then match *two* independent
  observables? That would upgrade "right shape" materially.
- Where does the model *diverge* from Newton/GR at accessible scales? A falsifying sim is
  worth more than another confirming one.

## Related

Reference n-body baseline: [solar-system-nbody](../sims/solar-system-nbody/) (conventional
Newtonian leapfrog — the control the field models are compared against).
