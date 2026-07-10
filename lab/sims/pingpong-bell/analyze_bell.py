"""Analyze a ping-pong Bell-test tally sheet: correlations E and CHSH S.

Part of the pingpong-bell project (20-projects/pingpong-bell/).
See protocol-and-analysis.md for the tally format:

    trial,angle_A,angle_B,outcome_A,outcome_B
    1,0,22.5,1,-1

outcome_*: 1 (UP), -1 (DOWN), 0 (null — ball missed both bins; kept for the
null-rate report, excluded from correlations).

Usage:
    python3 analyze_bell.py tally.csv
    python3 analyze_bell.py tally.csv --plot
"""

import csv
import itertools
import sys

import numpy as np


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["angle_A"]), float(r["angle_B"]),
                         int(r["outcome_A"]), int(r["outcome_B"])))
    if not rows:
        sys.exit("empty tally")
    return rows


def analyze(rows):
    angles_A = sorted({r[0] for r in rows})
    angles_B = sorted({r[1] for r in rows})
    total = len(rows)
    nulls = sum(1 for r in rows if r[2] == 0 or r[3] == 0)

    print(f"trials: {total}   nulls: {nulls} ({100*nulls/total:.1f}%)"
          + ("   <-- WARNING: >5% nulls, detection loophole is material"
             if nulls / total > 0.05 else ""))
    print(f"settings A: {angles_A}   B: {angles_B}\n")

    E, SE, N = {}, {}, {}
    for a in angles_A:
        for b in angles_B:
            prods = [rA * rB for aa, bb, rA, rB in rows
                     if aa == a and bb == b and rA != 0 and rB != 0]
            n = len(prods)
            N[a, b] = n
            if n == 0:
                print(f"  E({a:g}, {b:g}) = ---            (n=0)")
                continue
            e = float(np.mean(prods))
            se = float(np.std(prods, ddof=1) / np.sqrt(n)) if n > 1 else 1.0
            E[a, b], SE[a, b] = e, se
            print(f"  E({a:g}, {b:g}) = {e:+.3f} ± {se:.3f}  (n={n})")

    if len(angles_A) != 2 or len(angles_B) != 2 or len(E) < 4:
        print("\nneed exactly 2 settings per side with data in all 4 cells "
              "for CHSH; stopping at correlations.")
        return E

    (a1, a2), (b1, b2) = angles_A, angles_B
    best = None
    for signs in itertools.product([1, -1], repeat=4):
        if signs.count(-1) != 1:          # CHSH: exactly one minus term
            continue
        S = sum(s * E[p] for s, p in zip(
            signs, [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]))
        if best is None or abs(S) > abs(best[0]):
            best = (S, signs)
    S, signs = best
    sig = float(np.sqrt(sum(SE[p] ** 2 for p in
                            [(a1, b1), (a1, b2), (a2, b1), (a2, b2)])))

    terms = ["E(a1,b1)", "E(a1,b2)", "E(a2,b1)", "E(a2,b2)"]
    expr = " ".join(("+ " if s > 0 else "- ") + t
                    for s, t in zip(signs, terms)).lstrip("+ ")
    print(f"\nCHSH  S = {expr}")
    print(f"      |S| = {abs(S):.3f} ± {sig:.3f}")
    print(f"      classical (local hidden-variable) bound: 2")
    print(f"      quantum maximum: 2*sqrt(2) = {2*np.sqrt(2):.3f}")

    excess = (abs(S) - 2) / sig if sig > 0 else 0.0
    if abs(S) <= 2:
        print("      verdict: within the classical ceiling, as predicted "
              "for this rig.")
    elif excess < 2:
        print(f"      verdict: above 2 but only {excess:.1f} sigma — noise. "
              "Collect more trials.")
    else:
        print(f"      verdict: {excess:.1f} sigma above 2. NOT quantum "
              "ping-pong: a loophole is open. Work the checklist in "
              "experiment-design.md.")
    return E


def plot(rows, E):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = sorted(E)
    rel = [abs(a - b) for a, b in pairs]
    vals = [E[p] for p in pairs]
    th = np.linspace(0, 180, 361)
    plt.figure(figsize=(7, 4.5))
    plt.plot(th, -np.cos(np.deg2rad(th)), label="quantum: -cos(theta)")
    plt.plot(th, -1 + 2 * th / 180, "--", label="deterministic LHV: sawtooth")
    plt.plot(rel, vals, "o", ms=9, label="this rig")
    plt.xlabel("relative angle (deg)")
    plt.ylabel("correlation E")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("bell_correlations.png", dpi=150)
    print("\nplot saved: bell_correlations.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = load(sys.argv[1])
    E = analyze(rows)
    if "--plot" in sys.argv and E:
        plot(rows, E)
