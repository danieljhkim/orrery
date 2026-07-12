"""Measure two-source superposition on a finite shared counting lattice.

Question: when a compact source is embedded in an extended depletion well,
does the gradient attributable to that source grow as inverse headroom,
remain independent of the background, or screen as the shared count is
exhausted?

The apparatus first solves the five-point lattice Poisson equation for an
extended disk and a one-cell compact source.  The resulting exposures drive
literal finite counting: every cell starts with C occupied anonymous slots,
and Poisson-distributed depletion attempts choose slots uniformly.  Conditional
on an exposure H, the exact survivor count is Binomial(C, exp(-H)).  A paired
background/combined trial then removes compact-source hits only from the
background survivors.  This implements a shared finite budget without putting
any continuum superposition factor into the source coupling.

The compact field is the radial gradient of paired stored scarcity.  Its signed
projection onto the isolated-source gradient is swept from zero background to
99.5% ambient depletion.  The run records exact counting expectations, seeded
finite-count measurements, model residuals, saturation behavior, and two
resolution checks.

Run with:
    uv run lab/sims/lattice-two-source-superposition/main.py
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
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from orrery import rng


SEED = 42
PRIMARY_SIZE = 129
CHECK_SIZES = (97, 161)
CELL_CAPACITY = 50_000
REPLICATES = 32
BACKGROUND_RADIUS_FRACTION = 0.31
COMPACT_PEAK_EXPOSURE = 0.10
AMBIENT_DEPLETIONS = np.array(
    [0.0, 0.01, 0.03, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.96, 0.985, 0.995]
)
MEASURE_RADIUS_FRACTIONS = (4.0 / PRIMARY_SIZE, 18.0 / PRIMARY_SIZE)


@dataclass(frozen=True)
class LatticeFields:
    size: int
    radius: np.ndarray
    background_exposure: np.ndarray
    compact_exposure: np.ndarray
    fit_mask: np.ndarray


def poisson_field(source: np.ndarray) -> np.ndarray:
    """Solve 4h-neighbours = source with a zero-exposure outer boundary."""
    size = source.shape[0]
    width = size - 2
    points = width * width
    main = np.full(points, 4.0)
    side = np.full(points - 1, -1.0)
    side[np.arange(1, points) % width == 0] = 0.0
    vertical = np.full(points - width, -1.0)
    operator = diags(
        (vertical, side, main, side, vertical),
        (-width, -1, 0, 1, width),
        format="csc",
    )
    field = np.zeros_like(source, dtype=float)
    field[1:-1, 1:-1] = spsolve(operator, source[1:-1, 1:-1].ravel()).reshape(
        width, width
    )
    return field


def build_fields(size: int) -> LatticeFields:
    coordinate = np.arange(size) - size // 2
    xx, yy = np.meshgrid(coordinate, coordinate)
    radius = np.hypot(xx, yy)

    background_source = (
        radius <= BACKGROUND_RADIUS_FRACTION * size
    ).astype(float)
    compact_source = np.zeros((size, size))
    compact_source[size // 2, size // 2] = 1.0
    background = poisson_field(background_source)
    compact = poisson_field(compact_source)
    background /= background[size // 2, size // 2]
    compact *= COMPACT_PEAK_EXPOSURE / compact[size // 2, size // 2]

    low = max(2, int(round(MEASURE_RADIUS_FRACTIONS[0] * size)))
    high = int(round(MEASURE_RADIUS_FRACTIONS[1] * size))
    fit_mask = (np.arange(size // 2) >= low) & (np.arange(size // 2) <= high)
    return LatticeFields(size, radius, background, compact, fit_mask)


def radial_profile(values: np.ndarray, radius: np.ndarray) -> np.ndarray:
    bins = np.floor(radius).astype(int)
    count = np.bincount(bins.ravel())
    total = np.bincount(bins.ravel(), weights=values.ravel())
    profile = total / np.maximum(count, 1)
    return profile[: values.shape[0] // 2]


def field_amplitude(profile: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    gradient = np.gradient(profile)
    reference_gradient = np.gradient(reference)
    return float(
        np.dot(gradient[mask], reference_gradient[mask])
        / np.dot(reference_gradient[mask], reference_gradient[mask])
    )


def run_size(size: int, with_counts: bool) -> tuple[dict, dict[str, np.ndarray]]:
    fields = build_fields(size)
    isolated_map = -np.expm1(-fields.compact_exposure)
    isolated_profile = radial_profile(isolated_map, fields.radius)
    random = rng(SEED + size)
    expected_amplitudes = []
    measured_amplitudes = []
    measured_standard_errors = []
    delivered_fractions = []
    profiles = []

    for depletion in AMBIENT_DEPLETIONS:
        background_scale = 0.0 if depletion == 0.0 else -np.log1p(-depletion)
        survival_background = np.exp(-background_scale * fields.background_exposure)
        compact_hit_probability = isolated_map
        expected_increment = survival_background * compact_hit_probability
        expected_profile = radial_profile(expected_increment, fields.radius)
        expected_amplitudes.append(
            field_amplitude(expected_profile, isolated_profile, fields.fit_mask)
        )
        delivered_fractions.append(
            float(expected_increment.sum() / isolated_map.sum())
        )
        profiles.append(expected_profile)

        if with_counts:
            replicate_amplitudes = []
            for _ in range(REPLICATES):
                background_survivors = random.binomial(
                    CELL_CAPACITY, survival_background
                )
                compact_removals = random.binomial(
                    background_survivors, compact_hit_probability
                )
                measured_profile = radial_profile(
                    compact_removals / CELL_CAPACITY, fields.radius
                )
                replicate_amplitudes.append(
                    field_amplitude(
                        measured_profile, isolated_profile, fields.fit_mask
                    )
                )
            measured_amplitudes.append(float(np.mean(replicate_amplitudes)))
            measured_standard_errors.append(
                float(np.std(replicate_amplitudes, ddof=1) / np.sqrt(REPLICATES))
            )

    amplitude = np.asarray(expected_amplitudes)
    headroom = 1.0 - AMBIENT_DEPLETIONS
    fit = (AMBIENT_DEPLETIONS >= 0.01) & (AMBIENT_DEPLETIONS <= 0.90)
    alpha = float(
        np.dot(np.log(headroom[fit]), np.log(amplitude[fit]))
        / np.dot(np.log(headroom[fit]), np.log(headroom[fit]))
    )
    candidates = {
        "independence": np.ones_like(headroom),
        "screening-headroom": headroom,
        "multiplicative-enhancement": 1.0 / headroom,
        "fitted-headroom-power": headroom**alpha,
    }
    comparison = {}
    compare = AMBIENT_DEPLETIONS <= 0.90
    for name, prediction in candidates.items():
        comparison[name] = {
            "log_rmse_through_90_percent": float(
                np.sqrt(np.mean((np.log(amplitude[compare]) - np.log(prediction[compare])) ** 2))
            )
        }

    result = {
        "size": size,
        "fit_radius_cells": [
            int(np.flatnonzero(fields.fit_mask)[0]),
            int(np.flatnonzero(fields.fit_mask)[-1]),
        ],
        "fitted_modulation": {
            "form": "A(D) = (1-D)^alpha, constrained to A(0)=1",
            "alpha": alpha,
            "fit_depletion_range": [0.01, 0.90],
        },
        "candidate_comparison": comparison,
        "sweep": [],
    }
    for index, depletion in enumerate(AMBIENT_DEPLETIONS):
        row = {
            "ambient_depletion": float(depletion),
            "ambient_headroom": float(headroom[index]),
            "expected_field_amplitude": float(amplitude[index]),
            "field_over_headroom": float(amplitude[index] / headroom[index]),
            "delivered_compact_sink_fraction": delivered_fractions[index],
        }
        if with_counts:
            row.update(
                measured_field_amplitude=measured_amplitudes[index],
                measured_standard_error=measured_standard_errors[index],
                measurement_z_from_expectation=float(
                    (measured_amplitudes[index] - amplitude[index])
                    / measured_standard_errors[index]
                ),
            )
        result["sweep"].append(row)
    profile_data = {
        "radius": np.arange(len(isolated_profile)),
        "isolated": isolated_profile,
        "increments": np.asarray(profiles),
    }
    return result, profile_data


def make_plot(result: dict, profile_data: dict[str, np.ndarray], output: Path) -> None:
    sweep = result["primary"]["sweep"]
    depletion = np.array([row["ambient_depletion"] for row in sweep])
    headroom = 1.0 - depletion
    expected = np.array([row["expected_field_amplitude"] for row in sweep])
    measured = np.array([row["measured_field_amplitude"] for row in sweep])
    error = np.array([row["measured_standard_error"] for row in sweep])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(headroom, expected, "-o", color="#59c3ff", label="exact counting expectation")
    axes[0].errorbar(headroom, measured, yerr=error, fmt=".", color="#171717", label="seeded finite counts")
    axes[0].loglog(headroom, headroom, "--", color="#74d99f", label="screening: 1-D")
    axes[0].loglog(headroom, np.ones_like(headroom), ":", color="#ffb45c", label="independence")
    axes[0].loglog(headroom, 1.0 / headroom, "-.", color="#ff6b7a", label="enhancement: 1/(1-D)")
    axes[0].invert_xaxis()
    axes[0].set(xlabel="ambient headroom 1-D (exhaustion to the right)", ylabel="compact field / isolated field", title="Background modulation")

    radii = profile_data["radius"]
    for target, color in ((0.0, "#ffffff"), (0.8, "#59c3ff"), (0.96, "#ffb45c"), (0.995, "#ff6b7a")):
        index = int(np.argmin(np.abs(AMBIENT_DEPLETIONS - target)))
        axes[1].plot(radii, profile_data["increments"][index], color=color, label=f"D={AMBIENT_DEPLETIONS[index]:.3f}")
    axes[1].set_xlim(0, 28)
    axes[1].set(xlabel="radius from compact source (cells)", ylabel="paired compact scarcity", title="Saturation reshapes the local field")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Shared-budget counting lattice: embedded source is screened")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary, profile_data = run_size(PRIMARY_SIZE, with_counts=True)
    resolution = []
    for size in CHECK_SIZES:
        check, _ = run_size(size, with_counts=False)
        resolution.append(
            {
                "size": size,
                "alpha": check["fitted_modulation"]["alpha"],
                "max_absolute_amplitude_difference_from_primary": float(
                    np.max(
                        np.abs(
                            np.array([row["expected_field_amplitude"] for row in check["sweep"]])
                            - np.array([row["expected_field_amplitude"] for row in primary["sweep"]])
                        )
                    )
                ),
            }
        )

    result = {
        "schema_version": 1,
        "seed": SEED,
        "question": "Does ambient depletion enhance, leave unchanged, or screen an embedded compact source's stored-scarcity gradient?",
        "microscopic_rule": "Each cell has a finite set of anonymous occupied slots; background hits act first and compact-source hits can clear only surviving slots.",
        "apparatus": {
            "primary_size": PRIMARY_SIZE,
            "cell_capacity": CELL_CAPACITY,
            "replicates": REPLICATES,
            "background_source": f"uniform disk of radius {BACKGROUND_RADIUS_FRACTION} times lattice width",
            "compact_source": "one central lattice cell",
            "compact_peak_exposure": COMPACT_PEAK_EXPOSURE,
            "boundary": "zero exposure (full reservoir) on outermost cells",
            "ambient_depletions": AMBIENT_DEPLETIONS.tolist(),
        },
        "primary": primary,
        "resolution_checks": resolution,
        "verdict": {
            "apparatus_level": "screening, followed by saturation; no multiplicative enhancement",
            "scope": "This verdict follows from anonymous fixed-coupling depletion attempts on a shared finite count. A microscopic rule that changes attempt rate with occupancy would be a different model.",
        },
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    make_plot(result, profile_data, args.output_dir / "superposition.png")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
