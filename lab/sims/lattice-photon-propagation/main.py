"""Propagate one-hop-per-tick rays across a centrally depleted lattice.

Question: if a static counting lattice has fewer points near a central well,
does literal nearest-neighbour propagation bend a signal toward or away from
the well, and does it produce a delay or an advance?

The microscopic declaration is deliberately narrow.  ``n(r)`` is the local
linear density of addressable points along any ray.  A signal advances to one
neighbour per tick, so its physical hop length is ``1/n`` and its effective
refractive index is ``n``.  The central deficit is constant inside a finite
core and falls as 1/r^2 outside it.  Rays follow the local Huygens normal and
are integrated in full, finite hops.  Three geometrically similar lattices
test whether hop discreteness changes the sign.

Run with:
    uv run lab/sims/lattice-photon-propagation/main.py
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


DEFICIT_AT_CORE = 0.12
PRIMARY_CORE_HOPS = 32
CHECK_CORE_HOPS = (16, 64)
DOMAIN_CORE_RADII = 15.0
IMPACT_CORE_RADII = 3.0
PROFILE_RADII = np.geomspace(2.0, 12.0, 160)
PLOT_IMPACTS = (2.3, 3.0, 4.0, 5.5)


@dataclass(frozen=True)
class RayTrace:
    core_hops: int
    impact: float
    position: np.ndarray
    direction: np.ndarray
    ticks: np.ndarray

    @property
    def deflection_radians(self) -> float:
        return float(np.arctan2(self.direction[-1, 1], self.direction[-1, 0]))

    @property
    def transit_ticks(self) -> float:
        return float(self.ticks[-1])


def point_density(radius: np.ndarray | float, core: float) -> np.ndarray | float:
    """Linear point density, normalized to one far from the well."""
    radius_array = np.asarray(radius)
    scaled = np.maximum(radius_array, core) / core
    density = 1.0 - DEFICIT_AT_CORE / scaled**2
    return float(density) if density.ndim == 0 else density


def grad_log_density(position: np.ndarray, core: float) -> np.ndarray:
    """Gradient of log(n), zero in the declared flat finite core."""
    radius_squared = float(np.dot(position, position))
    if radius_squared <= core**2:
        return np.zeros(2)
    amplitude = DEFICIT_AT_CORE * core**2
    density = 1.0 - amplitude / radius_squared
    return (2.0 * amplitude / (density * radius_squared**2)) * position


def curvature(position: np.ndarray, direction: np.ndarray, core: float) -> np.ndarray:
    """Huygens/geometric-optics curvature d(direction)/d(arclength)."""
    gradient = grad_log_density(position, core)
    return gradient - direction * float(np.dot(direction, gradient))


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def trace_ray(core_hops: int, impact_core_radii: float) -> RayTrace:
    """Advance a ray by one local neighbour spacing per tick."""
    core = float(core_hops)
    x_limit = DOMAIN_CORE_RADII * core
    position = np.array([-x_limit, impact_core_radii * core], dtype=float)
    direction = np.array([1.0, 0.0])
    positions = [position.copy()]
    directions = [direction.copy()]
    ticks = [0.0]

    max_ticks = int(3 * x_limit)
    for _ in range(max_ticks):
        density = point_density(np.linalg.norm(position), core)
        hop = 1.0 / density

        # Symmetric kick-drift-kick: the drift is one literal local hop.
        half_direction = unit(direction + 0.5 * hop * curvature(position, direction, core))
        candidate = position + hop * half_direction
        next_direction = unit(
            half_direction
            + 0.5 * hop * curvature(candidate, half_direction, core)
        )

        if candidate[0] >= x_limit:
            fraction = (x_limit - position[0]) / (candidate[0] - position[0])
            position = position + fraction * (candidate - position)
            direction = unit(direction + fraction * (next_direction - direction))
            positions.append(position.copy())
            directions.append(direction.copy())
            ticks.append(ticks[-1] + float(fraction))
            break

        position = candidate
        direction = next_direction
        positions.append(position.copy())
        directions.append(direction.copy())
        ticks.append(ticks[-1] + 1.0)
    else:
        raise RuntimeError("ray did not cross the exit plane")

    return RayTrace(
        core_hops=core_hops,
        impact=impact_core_radii * core,
        position=np.asarray(positions),
        direction=np.asarray(directions),
        ticks=np.asarray(ticks),
    )


def fit_radial_law(core: float) -> dict:
    radii = PROFILE_RADII * core
    perturbation = 1.0 - point_density(radii, core)
    log_radius = np.log(radii / core)
    log_perturbation = np.log(perturbation)
    design = np.column_stack((np.ones_like(log_radius), -log_radius))
    intercept, exponent = np.linalg.lstsq(design, log_perturbation, rcond=None)[0]
    fitted = np.exp(intercept) * (radii / core) ** (-exponent)

    candidates = {}
    for name, power in (("inverse_r", 1.0), ("inverse_r_squared", 2.0)):
        shape = (radii / core) ** (-power)
        amplitude = float(np.dot(shape, perturbation) / np.dot(shape, shape))
        prediction = amplitude * shape
        candidates[name] = {
            "power": power,
            "amplitude_at_core": amplitude,
            "relative_rmse": float(
                np.sqrt(np.mean((prediction - perturbation) ** 2))
                / np.mean(perturbation)
            ),
        }

    return {
        "fit_range_core_radii": [float(PROFILE_RADII[0]), float(PROFILE_RADII[-1])],
        "fitted_power": float(exponent),
        "fitted_amplitude_at_core": float(np.exp(intercept)),
        "relative_rmse": float(
            np.sqrt(np.mean((fitted - perturbation) ** 2)) / np.mean(perturbation)
        ),
        "candidates": candidates,
    }


def summarize_trace(trace: RayTrace) -> dict:
    core = float(trace.core_hops)
    baseline_ticks = 2.0 * DOMAIN_CORE_RADII * core
    final_y_shift = float(trace.position[-1, 1] - trace.impact)
    deflection = trace.deflection_radians
    transit_anomaly = trace.transit_ticks - baseline_ticks
    return {
        "core_radius_hops": trace.core_hops,
        "impact_core_radii": trace.impact / core,
        "deflection_radians": deflection,
        "deflection_degrees": float(np.degrees(deflection)),
        "exit_transverse_shift_hops": final_y_shift,
        "exit_transverse_shift_core_radii": final_y_shift / core,
        "transit_ticks": trace.transit_ticks,
        "undepleted_baseline_ticks": baseline_ticks,
        "transit_anomaly_ticks": transit_anomaly,
        "transit_anomaly_per_core_radius": transit_anomaly / core,
        "deflection_sign": "away" if deflection > 0 else "toward",
        "transit_sign": "advance" if transit_anomaly < 0 else "delay",
    }


def validate(result: dict) -> None:
    radial = result["radial_law"]
    traces = [result["primary"], *result["resolution_checks"]]
    if abs(radial["fitted_power"] - 2.0) > 1e-10:
        raise AssertionError("radial fit did not recover the declared 1/r^2 law")
    if not (
        radial["candidates"]["inverse_r_squared"]["relative_rmse"]
        < radial["candidates"]["inverse_r"]["relative_rmse"] * 1e-8
    ):
        raise AssertionError("1/r^2 candidate did not decisively beat 1/r")
    if any(row["deflection_sign"] != "away" for row in traces):
        raise AssertionError("finite-hop ray changed the deflection sign")
    if any(row["transit_sign"] != "advance" for row in traces):
        raise AssertionError("finite-hop ray changed the transit-time sign")
    primary_angle = result["primary"]["deflection_radians"]
    if any(abs(row["deflection_radians"] - primary_angle) > 2e-4 for row in result["resolution_checks"]):
        raise AssertionError("deflection did not converge with lattice resolution")


def make_plot(result: dict, rays: list[RayTrace], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

    radius = PROFILE_RADII
    perturbation = 1.0 - point_density(radius, 1.0)
    axes[0].loglog(radius, perturbation, color="#59c3ff", linewidth=2, label="measured 1-n")
    axes[0].loglog(radius, DEFICIT_AT_CORE / radius, "--", color="#ffb45c", label="1/r candidate")
    axes[0].loglog(radius, DEFICIT_AT_CORE / radius**2, ":", color="#74d99f", linewidth=2, label="1/r^2 candidate")
    axes[0].set(xlabel="radius / core radius", ylabel="effective-index deficit 1-n", title="Radial law")
    axes[0].legend(fontsize=8)

    primary_core = float(PRIMARY_CORE_HOPS)
    circle = plt.Circle((0, 0), 1.0, color="#ff6b7a", alpha=0.18)
    axes[1].add_patch(circle)
    for ray in rays:
        scaled = ray.position / primary_core
        axes[1].plot(scaled[:, 0], scaled[:, 1], linewidth=1.5, label=f"b={ray.impact / primary_core:.1f} core")
    axes[1].axhline(0, color="#777777", linewidth=0.6)
    axes[1].set(xlim=(-DOMAIN_CORE_RADII, DOMAIN_CORE_RADII), xlabel="x / core radius", ylabel="y / core radius", title="Full-hop rays defocus")
    axes[1].legend(fontsize=8)

    convergence = sorted([result["primary"], *result["resolution_checks"]], key=lambda row: row["core_radius_hops"])
    sizes = np.array([row["core_radius_hops"] for row in convergence])
    angles = np.array([row["deflection_degrees"] for row in convergence])
    advances = -np.array([row["transit_anomaly_per_core_radius"] for row in convergence])
    axes[2].plot(sizes, angles, "-o", color="#59c3ff", label="away deflection (degrees)")
    axes[2].plot(sizes, advances, "-s", color="#ffb45c", label="advance / core radius")
    axes[2].set(xlabel="core radius (far-field hops)", title="Finite-hop convergence")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
    fig.suptitle("Static depleted lattice: one-hop-per-tick photon propagation")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary_trace = trace_ray(PRIMARY_CORE_HOPS, IMPACT_CORE_RADII)
    check_traces = [trace_ray(size, IMPACT_CORE_RADII) for size in CHECK_CORE_HOPS]
    plot_rays = [trace_ray(PRIMARY_CORE_HOPS, impact) for impact in PLOT_IMPACTS]
    result = {
        "schema_version": 1,
        "question": "Does literal one-hop-per-tick propagation on a static depleted lattice deflect toward or away and produce delay or advance?",
        "microscopic_rule": "n(r) is local linear point density; one adjacent-point hop per tick has physical length 1/n(r), and the Huygens normal is advanced in full local hops.",
        "apparatus": {
            "index_profile": "n=1-0.12 inside the core; n=1-0.12*(core/r)^2 outside",
            "deficit_at_core": DEFICIT_AT_CORE,
            "domain_half_width_core_radii": DOMAIN_CORE_RADII,
            "impact_core_radii": IMPACT_CORE_RADII,
            "primary_core_radius_hops": PRIMARY_CORE_HOPS,
            "resolution_core_radius_hops": list(CHECK_CORE_HOPS),
            "ray_integrator": "symmetric kick-drift-kick; each drift is one local neighbour spacing",
        },
        "radial_law": fit_radial_law(float(PRIMARY_CORE_HOPS)),
        "primary": summarize_trace(primary_trace),
        "resolution_checks": [summarize_trace(trace) for trace in check_traces],
        "verdict": {
            "deflection": "away from the depleted well",
            "radial_law": "effective-index perturbation follows 1/r^2, not 1/r",
            "transit_time": "advance relative to the undepleted one-hop lattice",
            "discrete_check": "the signs persist when the core spans 16, 32, and 64 far-field hops",
            "scope": "static density depletion only; a flowing-lattice/acoustic-metric completion is a different propagation rule",
        },
    }
    validate(result)

    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    make_plot(result, plot_rays, args.output_dir / "propagation.png")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
