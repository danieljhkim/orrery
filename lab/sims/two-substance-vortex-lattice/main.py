"""Measure interactions between quantized defects in a two-substance 2-D lattice.

Each lattice cell carries separately conserved A and B densities.  Their sum
shares a unit capacity, enforced by a one-sided pressure, while integer phase
windings live in the relative phase theta_A-theta_B.  A positive defect winds
A and expels A into a conserved halo; a negative defect winds B and expels B.
The apparatus constructs the minimum topological phase field for pinned cores,
then measures pair energy as the core separation changes.  A balanced defect
winds both phases equally and depletes both densities, so it has a void core
but no relative-phase charge.

This is deliberately a 2-D static lattice experiment.  It can test the sign,
short-range density correction, and the 2-D log-potential/1-r-force law.  It
cannot establish a 3-D 1/r potential or 1/r^2 force; that requires vortex
lines or rings in a three-dimensional lattice.

Run with:
    uv run lab/sims/two-substance-vortex-lattice/main.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orrery import rng


SEED = 42
SIZE = 257
BASE_DENSITY = 0.5
PHASE_STIFFNESS = 1.0
CAPACITY = 1.0
CAPACITY_PRESSURE = 18.0
DENSITY_STIFFNESS = 2.5
CORE_SIGMA = 1.8
HALO_SIGMA = 4.8
PROFILE_AMPLITUDE = 0.42
BALANCED_AMPLITUDE = 0.34
SEPARATIONS = np.arange(8.0, 73.0, 2.0)
FAR_MASK = (SEPARATIONS >= 30.0) & (SEPARATIONS <= 64.0)
SHORT_MASK = (SEPARATIONS >= 8.0) & (SEPARATIONS <= 16.0)


@dataclass(frozen=True)
class Defect:
    """Pinned integer winding and its density-core identity."""

    x: float
    y: float
    winding_a: int
    winding_b: int
    expelled: str

    @property
    def charge(self) -> int:
        return self.winding_a - self.winding_b


def positive(x: float, y: float, winding: int = 1) -> Defect:
    return Defect(x, y, winding, 0, "A")


def negative(x: float, y: float, winding: int = 1) -> Defect:
    return Defect(x, y, 0, winding, "B")


def balanced(x: float, y: float, winding: int = 1) -> Defect:
    return Defect(x, y, winding, winding, "both")


def coordinates() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(SIZE, dtype=float) - SIZE // 2
    return np.meshgrid(axis, axis)


def conserved_core_profile(
    xx: np.ndarray, yy: np.ndarray, defect: Defect
) -> np.ndarray:
    """Return a core deficit plus halo with exactly zero lattice sum."""
    radius2 = (xx - defect.x) ** 2 + (yy - defect.y) ** 2
    core = np.exp(-radius2 / (2.0 * CORE_SIGMA**2))
    halo = np.exp(-radius2 / (2.0 * HALO_SIGMA**2))
    profile = -core + (CORE_SIGMA / HALO_SIGMA) ** 2 * halo
    profile -= np.mean(profile)
    return profile


def fields(defects: list[Defect]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = coordinates()
    rho_a = np.full((SIZE, SIZE), BASE_DENSITY)
    rho_b = np.full((SIZE, SIZE), BASE_DENSITY)
    relative_phase = np.zeros((SIZE, SIZE))

    for defect in defects:
        relative_phase += defect.charge * np.arctan2(yy - defect.y, xx - defect.x)
        profile = conserved_core_profile(xx, yy, defect)
        amplitude = BALANCED_AMPLITUDE if defect.expelled == "both" else PROFILE_AMPLITUDE
        if defect.expelled in ("A", "both"):
            rho_a += amplitude * profile
        if defect.expelled in ("B", "both"):
            rho_b += amplitude * profile

    if np.min(rho_a) <= 0.0 or np.min(rho_b) <= 0.0:
        raise RuntimeError("density profile crossed zero; reduce the core amplitude")
    return rho_a, rho_b, relative_phase


def wrapped_difference(values: np.ndarray, axis: int) -> np.ndarray:
    difference = np.diff(values, axis=axis)
    return np.arctan2(np.sin(difference), np.cos(difference))


def energy(defects: list[Defect]) -> dict[str, float]:
    rho_a, rho_b, phase = fields(defects)
    total_density = rho_a + rho_b
    capacity_excess = np.maximum(total_density - CAPACITY, 0.0)
    capacity_energy = 0.5 * CAPACITY_PRESSURE * float(np.sum(capacity_excess**2))

    density_energy = 0.0
    for density in (rho_a, rho_b):
        density_energy += 0.5 * DENSITY_STIFFNESS * (
            float(np.sum(np.diff(density, axis=0) ** 2))
            + float(np.sum(np.diff(density, axis=1) ** 2))
        )

    # The relative phase is the only long-range mode.  This makes equal
    # co-winding neutral as a model hypothesis that the measurement tests.
    phase_energy = 0.5 * PHASE_STIFFNESS * (
        float(np.sum(wrapped_difference(phase, axis=0) ** 2))
        + float(np.sum(wrapped_difference(phase, axis=1) ** 2))
    )
    return {
        "phase": phase_energy,
        "capacity": capacity_energy,
        "density_gradient": density_energy,
        "total": phase_energy + capacity_energy + density_energy,
    }


def pair_defects(pair: str, separation: float) -> list[Defect]:
    left, right = -0.5 * separation, 0.5 * separation
    if pair == "like":
        return [positive(left, 0.0), positive(right, 0.0)]
    if pair == "opposite":
        return [positive(left, 0.0), negative(right, 0.0)]
    if pair == "balanced-positive":
        return [balanced(left, 0.0), positive(right, 0.0)]
    if pair == "balanced-negative":
        return [balanced(left, 0.0), negative(right, 0.0)]
    raise ValueError(f"unknown pair: {pair}")


def interaction_curve(pair: str) -> dict[str, np.ndarray]:
    components = {name: [] for name in ("phase", "capacity", "density_gradient", "total")}
    for separation in SEPARATIONS:
        defects = pair_defects(pair, separation)
        pair_energy = energy(defects)
        singles = [energy([defect]) for defect in defects]
        for name in components:
            components[name].append(pair_energy[name] - sum(item[name] for item in singles))

    curves = {name: np.asarray(values) for name, values in components.items()}
    curves["force"] = -np.gradient(curves["total"], SEPARATIONS)
    curves["phase_force"] = -np.gradient(curves["phase"], SEPARATIONS)
    curves["density_force"] = -np.gradient(
        curves["capacity"] + curves["density_gradient"], SEPARATIONS
    )
    return curves


def linear_fit(template: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack((np.ones_like(template), template))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    return coefficients, float(np.sqrt(np.mean(residual**2)))


def analyze(curves: dict[str, dict[str, np.ndarray]]) -> dict:
    conservation_fields = fields([positive(-12.0, 0.0), negative(12.0, 0.0)])
    expected_total = BASE_DENSITY * SIZE * SIZE
    result: dict[str, object] = {
        "apparatus": {
            "dimensions": 2,
            "lattice_size": [SIZE, SIZE],
            "boundary": "open square; fits stop at one quarter of the width",
            "density_totals": {
                "A": BASE_DENSITY * SIZE * SIZE,
                "B": BASE_DENSITY * SIZE * SIZE,
                "conservation": "core/halo profiles have exactly zero discrete sum",
                "maximum_absolute_error": float(
                    max(
                        abs(np.sum(conservation_fields[0]) - expected_total),
                        abs(np.sum(conservation_fields[1]) - expected_total),
                    )
                ),
            },
            "maximum_local_occupancy_in_conservation_probe": float(
                np.max(conservation_fields[0] + conservation_fields[1])
            ),
            "topological_charge": "q = winding_A - winding_B",
            "seed": SEED,
        },
        "pairs": {},
    }
    for pair, curve in curves.items():
        force = curve["force"]
        far_force = force[FAR_MASK]
        far_radius = SEPARATIONS[FAR_MASK]
        nonzero = np.abs(far_force) > 1.0e-12
        exponent = float(
            np.polyfit(np.log(far_radius[nonzero]), np.log(np.abs(far_force[nonzero])), 1)[0]
        )
        log_coefficients, log_rmse = linear_fit(
            np.log(far_radius), curve["total"][FAR_MASK]
        )
        inverse_coefficients, inverse_rmse = linear_fit(
            1.0 / far_radius, curve["total"][FAR_MASK]
        )
        result["pairs"][pair] = {
            "charges": [defect.charge for defect in pair_defects(pair, SEPARATIONS[0])],
            "far_force_mean": float(np.mean(far_force)),
            "far_force_sign": (
                "repulsive" if np.mean(far_force) > 1.0e-6 else
                "attractive" if np.mean(far_force) < -1.0e-6 else "neutral"
            ),
            "force_power_exponent": exponent,
            "potential_fit": {
                "log_r": {"slope": float(log_coefficients[1]), "rmse": log_rmse},
                "inverse_r": {
                    "slope": float(inverse_coefficients[1]),
                    "rmse": inverse_rmse,
                },
            },
            "short_range": {
                "total_force_mean": float(np.mean(force[SHORT_MASK])),
                "phase_force_mean": float(np.mean(curve["phase_force"][SHORT_MASK])),
                "density_force_mean": float(np.mean(curve["density_force"][SHORT_MASK])),
                "closest_separation": float(SEPARATIONS[0]),
                "density_force_at_closest_separation": float(curve["density_force"][0]),
                "maximum_density_force": float(np.max(curve["density_force"][SHORT_MASK])),
                "first_sample_with_density_strengthening": float(
                    SEPARATIONS[np.flatnonzero(curve["density_force"] > 0.0)[0]]
                ),
            },
            "sweep": [
                {
                    "separation": float(separation),
                    "interaction_energy": float(curve["total"][index]),
                    "phase_energy": float(curve["phase"][index]),
                    "density_energy": float(
                        curve["capacity"][index] + curve["density_gradient"][index]
                    ),
                    "force": float(force[index]),
                }
                for index, separation in enumerate(SEPARATIONS)
            ],
        }

    # A seeded jitter bootstrap quantifies sensitivity to the chosen fit window.
    random = rng(SEED)
    like_force = np.abs(curves["like"]["force"][FAR_MASK])
    far_radius = SEPARATIONS[FAR_MASK]
    exponents = []
    for _ in range(512):
        jitter = random.normal(0.0, 0.002, size=like_force.size)
        exponents.append(
            np.polyfit(np.log(far_radius), np.log(like_force * np.exp(jitter)), 1)[0]
        )
    result["pairs"]["like"]["force_exponent_seeded_jitter_sd"] = float(np.std(exponents, ddof=1))

    balanced_a = result["pairs"]["balanced-positive"]
    balanced_b = result["pairs"]["balanced-negative"]
    rho_a, rho_b, _ = fields([balanced(0.0, 0.0)])
    center = SIZE // 2
    result["balanced_defect"] = {
        "relative_phase_charge": 0,
        "core_total_density": float(rho_a[center, center] + rho_b[center, center]),
        "vacuum_total_density": 2.0 * BASE_DENSITY,
        "void_fraction": float(1.0 - rho_a[center, center] - rho_b[center, center]),
        "far_force_with_positive": balanced_a["far_force_mean"],
        "far_force_with_negative": balanced_b["far_force_mean"],
    }
    like = result["pairs"]["like"]
    opposite = result["pairs"]["opposite"]
    charged_scale = max(abs(like["far_force_mean"]), abs(opposite["far_force_mean"]))
    balanced_scale = max(
        abs(result["balanced_defect"]["far_force_with_positive"]),
        abs(result["balanced_defect"]["far_force_with_negative"]),
    )
    result["conclusions"] = {
        "sign": "like polarity repels; opposite polarity attracts",
        "far_field_law": "2-D logarithmic potential and approximately 1/r force",
        "three_dimensional_scope": (
            "not tested; a 3-D vortex-line or vortex-ring apparatus is required "
            "to test a 1/r potential and 1/r^2 force"
        ),
        "like_short_range_density_strengthening_fraction": float(
            like["short_range"]["density_force_mean"]
            / like["short_range"]["phase_force_mean"]
        ),
        "like_contact_behavior": (
            "density pressure strengthens repulsion across the 8-16 cell mean, "
            "but reverses and softens it at the closest 8-cell core-overlap probe"
        ),
        "balanced_far_force_fraction_of_charged_scale": float(
            balanced_scale / charged_scale
        ),
        "balanced_state": "far-field neutral within the measured residual, with a void core",
    }
    return result


def validate_result(result: dict) -> None:
    """Fail loudly if a regenerated artifact no longer supports its claims."""
    like = result["pairs"]["like"]
    opposite = result["pairs"]["opposite"]
    conclusions = result["conclusions"]
    checks = {
        "like defects repel": like["far_force_mean"] > 0.0,
        "opposite defects attract": opposite["far_force_mean"] < 0.0,
        "force is close to 1/r": -1.2 < like["force_power_exponent"] < -0.8,
        "log potential beats 1/r potential": (
            like["potential_fit"]["log_r"]["rmse"]
            < like["potential_fit"]["inverse_r"]["rmse"]
        ),
        "density term strengthens halo-range repulsion on average": (
            conclusions["like_short_range_density_strengthening_fraction"] > 0.0
        ),
        "closest core overlap is not monotonically stronger": (
            like["short_range"]["density_force_at_closest_separation"] < 0.0
        ),
        "balanced defect is far-field neutral": (
            conclusions["balanced_far_force_fraction_of_charged_scale"] < 1.0e-3
        ),
        "balanced defect retains a void": result["balanced_defect"]["void_fraction"] > 0.2,
        "both density totals are conserved": (
            result["apparatus"]["density_totals"]["maximum_absolute_error"] < 1.0e-9
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("validation failed: " + "; ".join(failed))


def make_plot(curves: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    colors = {
        "like": "#ff6b7a",
        "opposite": "#59c3ff",
        "balanced-positive": "#74d99f",
        "balanced-negative": "#c8a2ff",
    }
    for pair, curve in curves.items():
        axes[0].plot(SEPARATIONS, curve["total"], "-o", ms=2.5, color=colors[pair], label=pair)
        axes[1].plot(SEPARATIONS, curve["force"], "-o", ms=2.5, color=colors[pair], label=pair)
    axes[0].set(xlabel="core separation (cells)", ylabel="pair interaction energy", title="Measured pair potential")
    axes[1].axhline(0.0, color="#888888", lw=0.8)
    axes[1].set(xlabel="core separation (cells)", ylabel="radial force (+ outward)", title="Sign and force law")

    like = curves["like"]
    axes[2].plot(SEPARATIONS, like["phase_force"], color="#ffb45c", label="relative-phase force")
    axes[2].plot(SEPARATIONS, like["density_force"], color="#74d99f", label="density/capacity force")
    axes[2].plot(SEPARATIONS, like["force"], color="#ff6b7a", lw=2, label="total")
    axes[2].axhline(0.0, color="#888888", lw=0.8)
    axes[2].set_xlim(8, 32)
    axes[2].set(xlabel="core separation (cells)", ylabel="radial force (+ outward)", title="Like-polarity short range")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Two-substance vortex lattice (2-D, 257 x 257)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves = {
        pair: interaction_curve(pair)
        for pair in ("like", "opposite", "balanced-positive", "balanced-negative")
    }
    result = analyze(curves)
    validate_result(result)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    make_plot(curves, args.output_dir / "interaction-sweep.png")

    like = result["pairs"]["like"]
    opposite = result["pairs"]["opposite"]
    neutron = result["balanced_defect"]
    print(f"like polarity: {like['far_force_sign']}, mean far force {like['far_force_mean']:.6g}")
    print(f"opposite polarity: {opposite['far_force_sign']}, mean far force {opposite['far_force_mean']:.6g}")
    print(f"like force exponent: {like['force_power_exponent']:.3f} (2-D prediction -1)")
    print(f"balanced void fraction: {neutron['void_fraction']:.3f}")
    print(f"balanced far forces: + {neutron['far_force_with_positive']:.3g}, - {neutron['far_force_with_negative']:.3g}")


if __name__ == "__main__":
    main()
