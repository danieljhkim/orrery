---
title: "The vortex electron vs Bell: local hidden variables to the ceiling"
status: refuted
families: [bell-tests]
almanac: 15-discussions/26-07/vortex-electron-vs-bell.md
created: 2026-07-09
updated: 2026-07-09
---

# The vortex electron vs Bell

**The idea.** The electron rides its own ripples (a pilot-wave picture); spin outcomes come
from a locally carried axis meeting a detector "current" via tilt/friction dynamics. The
thread rebuilt — largely by independent reinvention — the historical sequence of local
hidden-variable models (de Broglie, EPR, Belinfante, Bell's own contextual single-spin model,
Hestenes, angle-targeted bias schemes), each repair fixing the objection that killed the last.

**Why it's refuted.** Every variant converged on the same wall: the two-particle CHSH bound
|S| ≤ 2, which is agnostic to shape, mechanism, and response rule. QM predicts and experiment
confirms |S| up to 2√2 ≈ 2.83 (measured ~2.4–2.7 in the 2015 loophole-free tests — see
[studies/chsh-bounds](../studies/chsh-bounds.md)). The controlling variable is the *relative*
detector angle, chosen at spacelike separation — a quantity that exists at no single location,
which no locally carried answer sheet can respond to. Single-particle contextuality survives;
two-particle locality does not.

This doc stays per the house rule: **a refuted branch is a result.** It is the most
instructive artifact in the repo — it shows *why* locality can't be rescued this way.

## Evidence ledger

| Claim | Status | Evidence |
|---|---|---|
| The friction-vortex response rule (outcome = sign(axis · current), deterministic or stochastic) reproduces single-particle spin statistics | supported | [vortex-bell](../lab/sims/vortex-bell/) `vortex_bell.py` — contextual single-spin behavior works, echoing Bell's own 1964 single-spin model |
| The same rule reproduces two-particle Bell correlations | refuted | [vortex-bell](../lab/sims/vortex-bell/): optimized over all detector settings, \|S\| saturates at exactly 2.000; E(22.5°) = −0.75 vs QM's −0.924; corkscrew and tilt-enumeration variants (`corkscrew_bell.py`, `tilt_enumeration.py`) hit the same ceiling |
| Nature itself respects the \|S\| ≤ 2 ceiling (so the model could still be right) | refuted | loophole-free experiments measure \|S\| ≈ 2.4–2.7 — [studies/chsh-bounds](../studies/chsh-bounds.md) |
| A tabletop analog (ping-pong balls with hidden tilt axes) demonstrates the classical ceiling empirically | supported | [pingpong-bell](../lab/sims/pingpong-bell/): protocol + analysis give \|S\| = 1.49 ± 0.17 on sample data — within the classical bound, as any local mechanism must be (protocol: almanac `20-projects/pingpong-bell/`) |

## What survives

- The pilot-wave *interference* story for a single particle is not what Bell kills; Bohmian
  mechanics survives by paying with explicit nonlocality.
- The [swirl-photon](swirl-photon.md) correspondence stands independently of this refutation.
