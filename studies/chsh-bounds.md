---
title: "CHSH: the classical bound, the quantum maximum, and the loophole-free measurements"
status: active
created: 2026-07-09
updated: 2026-07-09
---

# CHSH bounds

The facts the `bell-tests` family leans on.

## The statistic

S = E(a,b) − E(a,b′) + E(a′,b) + E(a′,b′), with E the correlation of ±1-valued outcomes at
detector angles chosen per wing. (`lib/py/orrery` exposes `chsh_S` in exactly this form.)

## The three numbers

| Value | What | Source |
|---|---|---|
| \|S\| ≤ 2 | ceiling for **any** local hidden-variable model, regardless of mechanism | J. S. Bell, *Physics Physique Fizika* **1**, 195 (1964); J. F. Clauser, M. A. Horne, A. Shimony, R. A. Holt, *Phys. Rev. Lett.* **23**, 880 (1969) |
| \|S\| ≤ 2√2 ≈ 2.828 | quantum-mechanical maximum (Tsirelson's bound) | B. S. Cirel'son, *Lett. Math. Phys.* **4**, 93 (1980) |
| \|S\| ≈ 2.4–2.7 | measured, with detection + locality loopholes closed simultaneously | 2015 loophole-free tests: Hensen et al. (Delft, NV centers, 1.3 km), *Nature* **526**, 682; Giustina et al. (Vienna), *Phys. Rev. Lett.* **115**, 250401; Shalm et al. (NIST), *Phys. Rev. Lett.* **115**, 250402 |

Context: the 2022 Nobel Prize in Physics (Clauser, Aspect, Zeilinger) recognized this
experimental lineage.

## What hangs on this

- [theory/vortex-electron](../theory/vortex-electron.md) is refuted by the gap between rows 1
  and 3: its best |S| is exactly 2.000 ([lab/sims/vortex-bell](../lab/sims/vortex-bell/)), and nature
  measurably exceeds 2.
- [lab/sims/pingpong-bell](../lab/sims/pingpong-bell/) demonstrates row 1 with a tabletop mechanism
  (sample data: |S| = 1.49 ± 0.17).
