"""Measure rays moving at constant local speed through an inward-flowing lattice.

The ray Hamiltonian is H(x,k) = v(x).k + c_local(x)|k|.  Thus the group
velocity is the lattice velocity plus a unit-speed signal in the comoving
frame.  The primary imposed v=sqrt(r_s/r) profile is a controlled
Gullstrand--Painleve surrogate, not an emergent flow of the counting rules.
Secondary arms transplant the exponents measured by ORB-10162.  A standing
1/r^2 density deficit can additionally raise the comoving signal speed and
provides the wrong-signed static-index contaminant.

Run with:
    uv run lab/sims/flowing-lattice-photon-propagation/main.py
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
from scipy.integrate import solve_ivp


SCHWARZSCHILD_RADIUS = 1.0
IMPACT = 20.0
DOMAIN_HALF_WIDTH = 400.0
MAX_STEPS = (1.6, 0.8, 0.4)
DEFICIT_AMPLITUDES = (0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12)
PLOT_AMPLITUDE = 0.04


@dataclass(frozen=True)
class Arm:
    name: str
    exponent: float
    exponent_uncertainty: float
    origin: str


ARMS = (
    Arm("imposed-gp-surrogate", 0.5, 0.0, "controlled v proportional to r^-1/2 surrogate"),
    Arm("measured-central-only", 2.00062, 0.00002, "ORB-10162 central-only measured profile"),
    Arm("measured-equal-per-shell", 0.99454, 0.00036, "ORB-10162 equal-per-shell measured profile"),
)


@dataclass(frozen=True)
class Trace:
    position: np.ndarray
    momentum: np.ndarray
    transit_time: float
    evaluations: int

    @property
    def deflection(self) -> float:
        return float(np.arctan2(self.momentum[-1, 1], self.momentum[-1, 0]))


def flow_and_jacobian(position: np.ndarray, exponent: float, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    """Inward radial flow and its Cartesian Jacobian.

    All arms have the same speed at the impact radius.  The secondary arms
    therefore compare the measured ORB-10162 *shapes* without pretending its
    apparatus-specific velocity normalization supplies a gravitational scale.
    """
    if not enabled:
        return np.zeros(2), np.zeros((2, 2))
    radius = float(np.linalg.norm(position))
    speed_at_impact = np.sqrt(SCHWARZSCHILD_RADIUS / IMPACT)
    coefficient = speed_at_impact * IMPACT**exponent
    factor = -coefficient * radius ** (-(exponent + 1.0))
    flow = factor * position
    jacobian = factor * (
        np.eye(2) - (exponent + 1.0) * np.outer(position, position) / radius**2
    )
    return flow, jacobian


def local_speed_and_gradient(position: np.ndarray, deficit: float) -> tuple[float, np.ndarray]:
    """Comoving hop speed for the static arm's 1/r^2 point deficit."""
    radius = float(np.linalg.norm(position))
    density = 1.0 - deficit * (SCHWARZSCHILD_RADIUS / radius) ** 2
    speed = 1.0 / density
    radial_derivative = -2.0 * deficit * SCHWARZSCHILD_RADIUS**2 / (radius**3 * density**2)
    return speed, radial_derivative * position / radius


def trace_ray(exponent: float, deficit: float, flow_enabled: bool, max_step: float) -> Trace:
    initial = np.array([-DOMAIN_HALF_WIDTH, IMPACT, 1.0, 0.0])

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        position = state[:2]
        momentum = state[2:]
        wave_number = float(np.linalg.norm(momentum))
        flow, jacobian = flow_and_jacobian(position, exponent, flow_enabled)
        speed, speed_gradient = local_speed_and_gradient(position, deficit)
        group_velocity = flow + speed * momentum / wave_number
        momentum_rate = -(jacobian.T @ momentum) - wave_number * speed_gradient
        return np.concatenate((group_velocity, momentum_rate))

    def exit_plane(_time: float, state: np.ndarray) -> float:
        return float(state[0] - DOMAIN_HALF_WIDTH)

    exit_plane.terminal = True
    exit_plane.direction = 1
    solution = solve_ivp(
        rhs,
        (0.0, 4.0 * DOMAIN_HALF_WIDTH),
        initial,
        method="DOP853",
        events=exit_plane,
        rtol=2e-11,
        atol=2e-13,
        max_step=max_step,
    )
    if not solution.success or len(solution.t_events[0]) != 1:
        raise RuntimeError(f"ray failed to reach exit plane: {solution.message}")
    return Trace(
        position=solution.y[:2].T,
        momentum=solution.y[2:].T,
        transit_time=float(solution.t_events[0][0]),
        evaluations=solution.nfev,
    )


def convergence_measurement(exponent: float, deficit: float, flow_enabled: bool) -> tuple[dict, Trace]:
    traces = [trace_ray(exponent, deficit, flow_enabled, step) for step in MAX_STEPS]
    finest = traces[-1]
    deflections = np.array([trace.deflection for trace in traces])
    times = np.array([trace.transit_time for trace in traces])
    baseline = 2.0 * DOMAIN_HALF_WIDTH
    return {
        "deflection_radians": finest.deflection,
        "deflection_degrees": float(np.degrees(finest.deflection)),
        "deflection_sign": "toward" if finest.deflection < 0.0 else "away",
        "deflection_numerical_uncertainty_radians": float(np.max(np.abs(deflections[:-1] - deflections[-1]))),
        "transit_time": finest.transit_time,
        "flat_transit_time": baseline,
        "transit_anomaly": finest.transit_time - baseline,
        "transit_sign": "delay" if finest.transit_time > baseline else "advance",
        "transit_numerical_uncertainty": float(np.max(np.abs(times[:-1] - times[-1]))),
        "convergence_max_steps": list(MAX_STEPS),
        "convergence_deflections": deflections.tolist(),
        "convergence_transit_times": times.tolist(),
        "finest_rhs_evaluations": finest.evaluations,
    }, finest


def gamma_one_expectation() -> dict:
    radius = np.hypot(DOMAIN_HALF_WIDTH, IMPACT)
    separation = 2.0 * DOMAIN_HALF_WIDTH
    return {
        "deflection_radians_leading_order": 2.0 * SCHWARZSCHILD_RADIUS / IMPACT,
        "transit_delay_leading_order": SCHWARZSCHILD_RADIUS
        * np.log((2.0 * radius + separation) / (2.0 * radius - separation)),
        "formula": "alpha=(1+gamma) r_s/b with gamma=1; delay=r_s ln((r1+r2+R)/(r1+r2-R))",
    }


def build_result() -> tuple[dict, dict[str, Trace]]:
    expectation = gamma_one_expectation()
    traces: dict[str, Trace] = {}
    arms = {}
    for arm in ARMS:
        measurement, trace = convergence_measurement(arm.exponent, 0.0, True)
        traces[arm.name] = trace
        measurement.update(
            {
                "profile_exponent": arm.exponent,
                "profile_exponent_uncertainty": arm.exponent_uncertainty,
                "profile_origin": arm.origin,
                "deflection_over_gamma_one_expectation": abs(measurement["deflection_radians"])
                / expectation["deflection_radians_leading_order"],
                "delay_over_gamma_one_expectation": measurement["transit_anomaly"]
                / expectation["transit_delay_leading_order"],
            }
        )
        arms[arm.name] = measurement

    flow_only = arms[ARMS[0].name]
    contaminant = []
    for amplitude in DEFICIT_AMPLITUDES:
        index_only, _ = convergence_measurement(ARMS[0].exponent, amplitude, False)
        combined, combined_trace = convergence_measurement(ARMS[0].exponent, amplitude, True)
        if amplitude == PLOT_AMPLITUDE:
            traces["combined"] = combined_trace
        flow_angle = flow_only["deflection_radians"]
        index_angle = index_only["deflection_radians"]
        contaminant.append(
            {
                "standing_deficit_amplitude_at_r_s": amplitude,
                "flow_only_deflection_radians": flow_angle,
                "index_only_deflection_radians": index_angle,
                "combined_deflection_radians": combined["deflection_radians"],
                "flow_drag_to_index_gradient_ratio": abs(flow_angle / index_angle),
                "nonlinear_interaction_radians": combined["deflection_radians"] - flow_angle - index_angle,
                "index_only_deflection_uncertainty_radians": index_only["deflection_numerical_uncertainty_radians"],
                "combined_deflection_uncertainty_radians": combined["deflection_numerical_uncertainty_radians"],
                "combined_transit_anomaly": combined["transit_anomaly"],
                "combined_transit_uncertainty": combined["transit_numerical_uncertainty"],
            }
        )

    return {
        "schema_version": 1,
        "question": "Does a constant-comoving-speed ray advected by inward lattice flow recover toward-deflection and delay, and how strongly does the static deficit contaminate it?",
        "apparatus": {
            "ray_hamiltonian": "H=v(x).k+c_local(x)|k|",
            "group_velocity": "dx/dt=v+c_local*k/|k|",
            "schwarzschild_radius": SCHWARZSCHILD_RADIUS,
            "impact_parameter": IMPACT,
            "domain_half_width": DOMAIN_HALF_WIDTH,
            "primary_flow": "inward |v|=sqrt(r_s/r), imposed GP surrogate",
            "secondary_normalization": "same speed as primary at the impact radius; measured exponents transplanted from ORB-10162",
            "standing_deficit": "n(r)=1-A(r_s/r)^2 and c_local=1/n(r)",
        },
        "surrogate_status": "ORB-10162 measured natural central-only and equal-per-shell rules near r^-2 and r^-1, not r^-1/2; the primary is a controlled conditional surrogate.",
        "gamma_one_expectation": expectation,
        "flow_arms": arms,
        "contaminant_sweep": contaminant,
        "measurement_summary": {
            "primary_deflection": flow_only["deflection_sign"],
            "primary_transit": flow_only["transit_sign"],
            "scope": "apparatus measurements only; theory-ledger judgment belongs to kepler",
        },
    }, traces


def validate(result: dict) -> None:
    primary = result["flow_arms"]["imposed-gp-surrogate"]
    if primary["deflection_sign"] != "toward" or primary["transit_sign"] != "delay":
        raise AssertionError("primary GP surrogate did not produce the declared signs")
    if not 0.75 < primary["deflection_over_gamma_one_expectation"] < 1.35:
        raise AssertionError("primary deflection is inconsistent with the gamma=1 scale")
    if not 0.6 < primary["delay_over_gamma_one_expectation"] < 1.5:
        raise AssertionError("primary delay is inconsistent with the gamma=1 scale")
    if set(result["flow_arms"]) != {arm.name for arm in ARMS}:
        raise AssertionError("measured-profile secondary arms are missing")
    ratios = np.array([row["flow_drag_to_index_gradient_ratio"] for row in result["contaminant_sweep"]])
    if not np.all(np.diff(ratios) < 0.0):
        raise AssertionError("contaminant ratio did not decrease with deficit amplitude")
    if any(row["index_only_deflection_radians"] <= 0.0 for row in result["contaminant_sweep"]):
        raise AssertionError("static-index contaminant did not deflect away")
    if primary["deflection_numerical_uncertainty_radians"] > 2e-8 or primary["transit_numerical_uncertainty"] > 2e-7:
        raise AssertionError("primary resolution uncertainty is too large")


def make_plot(result: dict, traces: dict[str, Trace], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    colors = {"imposed-gp-surrogate": "#59c3ff", "measured-central-only": "#ffb45c", "measured-equal-per-shell": "#74d99f"}
    for name, trace in traces.items():
        if name == "combined":
            axes[0].plot(trace.position[:, 0], trace.position[:, 1], "--", color="#ff6b7a", label=f"GP + A={PLOT_AMPLITUDE:g}")
        else:
            axes[0].plot(trace.position[:, 0], trace.position[:, 1], color=colors[name], label=name)
    axes[0].axhline(IMPACT, color="#777777", linewidth=0.7)
    axes[0].set(xlabel="x / r_s", ylabel="y / r_s", title="Advected ray paths")
    axes[0].legend(fontsize=7)

    names = list(result["flow_arms"])
    measured = [abs(result["flow_arms"][name]["deflection_radians"]) for name in names]
    axes[1].bar(range(len(names)), measured, color=[colors[name] for name in names])
    axes[1].axhline(result["gamma_one_expectation"]["deflection_radians_leading_order"], color="#ffffff", linestyle="--", label="gamma=1 leading order")
    axes[1].set_xticks(range(len(names)), ["r^-1/2", "r^-2.00062", "r^-0.99454"])
    axes[1].set(ylabel="toward deflection / rad", title="Flow-profile arms")
    axes[1].legend(fontsize=8)

    sweep = result["contaminant_sweep"]
    amplitudes = [row["standing_deficit_amplitude_at_r_s"] for row in sweep]
    ratios = [row["flow_drag_to_index_gradient_ratio"] for row in sweep]
    axes[2].loglog(amplitudes, ratios, "-o", color="#ff6b7a")
    axes[2].axhline(1.0, color="#ffffff", linestyle="--", linewidth=0.8)
    axes[2].set(xlabel="standing-deficit amplitude A", ylabel="|flow deflection| / |index deflection|", title="Contaminant ratio")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
    fig.suptitle("Flowing-lattice photon propagation")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result, traces = build_result()
    validate(result)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    make_plot(result, traces, args.output_dir / "propagation.png")

    primary = result["flow_arms"]["imposed-gp-surrogate"]
    print(f"primary deflection: {primary['deflection_radians']:.9f} rad ({primary['deflection_sign']})")
    print(f"gamma=1 deflection ratio: {primary['deflection_over_gamma_one_expectation']:.6f}")
    print(f"primary transit anomaly: {primary['transit_anomaly']:.9f} ({primary['transit_sign']})")
    print(f"gamma=1 delay ratio: {primary['delay_over_gamma_one_expectation']:.6f}")


if __name__ == "__main__":
    main()
