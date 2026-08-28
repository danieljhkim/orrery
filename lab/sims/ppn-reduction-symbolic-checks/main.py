"""Verify the exact algebra behind the settled-flow PPN reduction.

The deterministic SymPy fixture checks the reservoir-frame PG transform,
all-order flatness of pure wind, exact steady-flow diagonalization and its
static Schwarzschild endpoint, and the two-piece obstruction to removing the
shift.  A double expansion then exposes a generic, curvature-carrying
``sqrt(epsilon) * epsilon_w`` wake term.  That half-order term survives both
the temporal shift and a spatial coordinate choice, so the predeclared
super-PPN hazard fires unless the realized wake obeys extra cancellations.

Usage:
    uv run lab/sims/ppn-reduction-symbolic-checks/main.py
    uv run lab/sims/ppn-reduction-symbolic-checks/main.py --check-determinism
"""

from __future__ import annotations

import argparse
import json
from math import factorial
from pathlib import Path

import sympy as sp

TASK_ID = "ORB-11040"
SLUG = "ppn-reduction-symbolic-checks"
RESULT_PATH = Path(__file__).parent / "assets" / "results.json"


def acoustic_metric(flow: sp.Matrix, c: sp.Symbol) -> sp.Matrix:
    """Return ds^2 = -c^2 dt^2 + |dx - flow dt|^2."""
    metric = sp.eye(4)
    metric[0, 0] = -c**2 + flow.dot(flow)
    for index in range(3):
        metric[0, index + 1] = metric[index + 1, 0] = -flow[index]
    return metric


def matrix_is_zero(matrix: sp.Matrix) -> bool:
    return all(sp.simplify(value) == 0 for value in matrix)


def vector_is_zero(vector: sp.Matrix) -> bool:
    return all(sp.simplify(value) == 0 for value in vector)


def expression_list(values: sp.Matrix | list[sp.Expr]) -> list[str]:
    return [sp.sstr(sp.factor(value)) for value in values]


def gradient(value: sp.Expr, coordinates: tuple[sp.Symbol, ...]) -> sp.Matrix:
    return sp.Matrix([sp.diff(value, coordinate) for coordinate in coordinates])


def curl(vector: sp.Matrix, coordinates: tuple[sp.Symbol, ...]) -> sp.Matrix:
    x, y, z = coordinates
    return sp.Matrix(
        [
            sp.diff(vector[2], y) - sp.diff(vector[1], z),
            sp.diff(vector[0], z) - sp.diff(vector[2], x),
            sp.diff(vector[1], x) - sp.diff(vector[0], y),
        ]
    )


def riemann_tensor(
    metric: sp.Matrix, coordinates: tuple[sp.Symbol, ...]
) -> list[list[list[list[sp.Expr]]]]:
    """Return R^rho_{ sigma mu nu} for a four-dimensional metric."""
    dimension = len(coordinates)
    inverse = sp.simplify(metric.inv())
    christoffel = [
        [
            [
                sp.simplify(
                    sum(
                        inverse[rho, delta]
                        * (
                            sp.diff(metric[delta, nu], coordinates[mu])
                            + sp.diff(metric[delta, mu], coordinates[nu])
                            - sp.diff(metric[mu, nu], coordinates[delta])
                        )
                        / 2
                        for delta in range(dimension)
                    )
                )
                for nu in range(dimension)
            ]
            for mu in range(dimension)
        ]
        for rho in range(dimension)
    ]
    return [
        [
            [
                [
                    sp.simplify(
                        sp.diff(christoffel[rho][nu][sigma], coordinates[mu])
                        - sp.diff(christoffel[rho][mu][sigma], coordinates[nu])
                        + sum(
                            christoffel[rho][mu][eta]
                            * christoffel[eta][nu][sigma]
                            - christoffel[rho][nu][eta]
                            * christoffel[eta][mu][sigma]
                            for eta in range(dimension)
                        )
                    )
                    for nu in range(dimension)
                ]
                for mu in range(dimension)
            ]
            for sigma in range(dimension)
        ]
        for rho in range(dimension)
    ]


def series_coefficient(
    value: sp.Expr,
    first: sp.Symbol,
    first_power: int,
    second: sp.Symbol,
    second_power: int,
) -> sp.Expr:
    derivative = sp.diff(value, first, first_power, second, second_power)
    return sp.simplify(
        derivative.subs({first: 0, second: 0})
        / (factorial(first_power) * factorial(second_power))
    )


def statement_zero_gate() -> dict:
    c = sp.symbols("c", nonzero=True)
    velocity = sp.Matrix(sp.symbols("v_x v_y v_z"))
    wind = sp.Matrix(sp.symbols("U_x U_y U_z"))
    disturbance = sp.Matrix(sp.symbols("u_x u_y u_z"))

    # x_reservoir = x_mass + U t, hence dx_mass = dx_reservoir - U dt.
    jacobian = sp.eye(4)
    for index in range(3):
        jacobian[index + 1, 0] = -wind[index]
    transformed = sp.simplify(jacobian.T * acoustic_metric(velocity, c) * jacobian)
    target = acoustic_metric(velocity + wind, c)
    in_disturbance_variables = transformed.subs(
        {velocity[index]: disturbance[index] - wind[index] for index in range(3)}
    ).applyfunc(sp.simplify)
    wind_derivatives_are_zero = all(
        matrix_is_zero(in_disturbance_variables.diff(component))
        for component in wind
    )
    passed = matrix_is_zero(transformed - target) and matrix_is_zero(
        in_disturbance_variables - acoustic_metric(disturbance, c)
    )
    passed = passed and wind_derivatives_are_zero
    return {
        "name": "statement_0_reservoir_pg_form",
        "passed": passed,
        "details": {
            "transformed_metric_equals_pg_of_u": matrix_is_zero(transformed - target),
            "metric_after_v_equals_u_minus_U": sp.sstr(in_disturbance_variables),
            "explicit_wind_derivative_at_fixed_u_is_zero": wind_derivatives_are_zero,
        },
    }


def pure_wind_flatness_gate() -> dict:
    t, x, y, z = sp.symbols("t x y z")
    c = sp.symbols("c", nonzero=True)
    wind = sp.Matrix(sp.symbols("U_x U_y U_z"))
    metric = acoustic_metric(-wind, c)
    tensor = riemann_tensor(metric, (t, x, y, z))
    components = [
        tensor[rho][sigma][mu][nu]
        for rho in range(4)
        for sigma in range(4)
        for mu in range(4)
        for nu in range(4)
    ]

    # The same result is the pullback of Minkowski under x_reservoir=x_mass+Ut.
    jacobian = sp.eye(4)
    for index in range(3):
        jacobian[index + 1, 0] = wind[index]
    minkowski = sp.diag(-c**2, 1, 1, 1)
    pullback = sp.simplify(jacobian.T * minkowski * jacobian)
    nonzero_components = [value for value in components if sp.simplify(value) != 0]
    passed = not nonzero_components and matrix_is_zero(pullback - metric)
    return {
        "name": "pure_wind_full_riemann_flatness",
        "passed": passed,
        "details": {
            "riemann_components_checked": len(components),
            "nonzero_components": len(nonzero_components),
            "exact_minkowski_pullback": matrix_is_zero(pullback - metric),
            "orders_in_U": "all",
        },
    }


def diagonalization_gate() -> dict:
    c, epsilon, radius, theta = sp.symbols(
        "c epsilon r theta", positive=True
    )
    flow = sp.Matrix(sp.symbols("u_x u_y u_z"))
    speed_squared = flow.dot(flow)
    denominator = c**2 - speed_squared
    jacobian = sp.eye(4)
    for index in range(3):
        jacobian[0, index + 1] = -flow[index] / denominator
    diagonalized = sp.simplify(jacobian.T * acoustic_metric(flow, c) * jacobian)
    target = sp.zeros(4)
    target[0, 0] = -denominator
    target[1:4, 1:4] = sp.eye(3) + flow * flow.T / denominator

    # Static GP in spherical PG coordinates: u_r/c=-sqrt(epsilon).
    radial_flow = -c * sp.sqrt(epsilon)
    pg_spherical = sp.diag(0, 1, radius**2, radius**2 * sp.sin(theta) ** 2)
    pg_spherical[0, 0] = -c**2 * (1 - epsilon)
    pg_spherical[0, 1] = pg_spherical[1, 0] = -radial_flow
    radial_jacobian = sp.eye(4)
    radial_jacobian[0, 1] = -radial_flow / (c**2 * (1 - epsilon))
    static_diagonal = sp.simplify(
        radial_jacobian.T * pg_spherical * radial_jacobian
    )
    schwarzschild = sp.diag(
        -c**2 * (1 - epsilon),
        1 / (1 - epsilon),
        radius**2,
        radius**2 * sp.sin(theta) ** 2,
    )
    identity_passed = matrix_is_zero(diagonalized - target)
    static_passed = matrix_is_zero(static_diagonal - schwarzschild)
    return {
        "name": "steady_diagonalization_and_static_schwarzschild",
        "passed": identity_passed and static_passed,
        "details": {
            "cross_terms_vanish": all(
                sp.simplify(diagonalized[0, index]) == 0 for index in range(1, 4)
            ),
            "exact_spatial_stretch": identity_passed,
            "gp_lapse_squared": sp.sstr(-static_diagonal[0, 0] / c**2),
            "gp_g_rr": sp.sstr(static_diagonal[1, 1]),
            "exact_schwarzschild": static_passed,
        },
    }


def obstruction_gate() -> dict:
    x, y, z, c = sp.symbols("x y z c", nonzero=True)
    coordinates = (x, y, z)
    functions = sp.Matrix(
        [sp.Function(name)(x, y, z) for name in ("u_x", "u_y", "u_z")]
    )
    speed_squared = functions.dot(functions)
    denominator = c**2 - speed_squared
    left = curl(functions / denominator, coordinates)
    vorticity_piece = curl(functions, coordinates) / denominator
    anisotropy_piece = gradient(speed_squared, coordinates).cross(functions) / (
        denominator**2
    )
    identity_residual = sp.simplify(left - vorticity_piece - anisotropy_piece)

    radius_squared = x**2 + y**2 + z**2
    radial_profile = sp.Function("F")(radius_squared)
    static_gp = radial_profile * sp.Matrix([x, y, z])
    static_speed_squared = static_gp.dot(static_gp)
    static_vorticity = sp.simplify(curl(static_gp, coordinates))
    static_anisotropy_numerator = sp.simplify(
        gradient(static_speed_squared, coordinates).cross(static_gp)
    )

    anisotropic = sp.Matrix([x, 2 * y, 3 * z])
    anisotropic_speed_squared = anisotropic.dot(anisotropic)
    anisotropic_numerator = sp.simplify(
        gradient(anisotropic_speed_squared, coordinates).cross(anisotropic)
    )
    expected_anisotropic_numerator = sp.Matrix([-12 * y * z, 12 * x * z, -4 * x * y])
    passed = vector_is_zero(identity_residual)
    passed = passed and vector_is_zero(static_vorticity)
    passed = passed and vector_is_zero(static_anisotropy_numerator)
    passed = passed and vector_is_zero(
        anisotropic_numerator - expected_anisotropic_numerator
    )
    passed = passed and not vector_is_zero(anisotropic_numerator)
    return {
        "name": "two_piece_g0i_obstruction",
        "passed": passed,
        "details": {
            "vector_identity_residual": expression_list(identity_residual),
            "static_gp_vorticity_piece": expression_list(static_vorticity),
            "static_gp_anisotropy_piece_numerator": expression_list(
                static_anisotropy_numerator
            ),
            "anisotropic_test_field": ["x", "2*y", "3*z"],
            "anisotropic_piece_numerator": expression_list(anisotropic_numerator),
            "anisotropic_piece_nonzero": not vector_is_zero(anisotropic_numerator),
        },
    }


def half_order_gate() -> dict:
    p, wind_order = sp.symbols("p epsilon_w")
    a = sp.Matrix(sp.symbols("a_x a_y a_z"))
    b = sp.Matrix(sp.symbols("b_x b_y b_z"))
    flow = p * a + wind_order * b
    speed_squared = flow.dot(flow)
    flat = sp.diag(-1, 1, 1, 1)
    pg_perturbation = acoustic_metric(flow, sp.Integer(1)) - flat

    diagonal = sp.zeros(4)
    diagonal[0, 0] = -(1 - speed_squared)
    diagonal[1:4, 1:4] = sp.eye(3) + flow * flow.T / (1 - speed_squared)
    diagonal_perturbation = diagonal - flat
    temporal_generator = -flow / (1 - speed_squared)

    dot_ab = a.dot(b)
    expected_spatial_cross = a * b.T + b * a.T
    raw_checks = [
        series_coefficient(pg_perturbation[0, 0], p, 2, wind_order, 0)
        - a.dot(a),
        series_coefficient(pg_perturbation[0, 0], p, 1, wind_order, 1)
        - 2 * dot_ab,
    ]
    raw_checks.extend(
        series_coefficient(pg_perturbation[0, index + 1], p, 1, wind_order, 0)
        + a[index]
        for index in range(3)
    )
    raw_checks.extend(
        series_coefficient(pg_perturbation[0, index + 1], p, 0, wind_order, 1)
        + b[index]
        for index in range(3)
    )
    diagonal_checks = [
        series_coefficient(diagonal_perturbation[0, 0], p, 1, wind_order, 1)
        - 2 * dot_ab
    ]
    diagonal_checks.extend(
        series_coefficient(
            diagonal_perturbation[row + 1, column + 1],
            p,
            1,
            wind_order,
            1,
        )
        - expected_spatial_cross[row, column]
        for row in range(3)
        for column in range(3)
    )
    diagonal_checks.extend(
        series_coefficient(
            diagonal_perturbation[row, column], p, 2, wind_order, 1
        )
        for row in range(4)
        for column in range(4)
    )
    expected_generator_correction = -(a.dot(a) * b + 2 * dot_ab * a)
    generator_correction = sp.Matrix(
        [
            series_coefficient(value, p, 2, wind_order, 1)
            for value in temporal_generator
        ]
    )

    # A radial potential base flow and an anisotropic potential wake remove h_0i
    # through O(p*w), yet their diagonal half-order metric has nonzero linearized
    # R_0i0j.  No temporal or spatial coordinate choice can erase curvature.
    x, y, z = sp.symbols("x y z")
    coordinates = (x, y, z)
    witness_a = sp.Matrix([x, y, z])
    witness_b = sp.Matrix([x, 2 * y, 3 * z])
    witness_h00 = 2 * witness_a.dot(witness_b)
    witness_spatial = witness_a * witness_b.T + witness_b * witness_a.T
    curvature_witness = sp.Matrix(
        3,
        3,
        lambda row, column: -sp.diff(
            witness_h00, coordinates[row], coordinates[column]
        )
        / 2,
    )
    expected_curvature = sp.diag(-2, -4, -6)
    witness_passed = matrix_is_zero(curvature_witness - expected_curvature)
    hazard_survives = witness_passed and not matrix_is_zero(curvature_witness)

    classifications = [
        {
            "slot": "h_00",
            "order": "epsilon",
            "term": "a.a",
            "classification": "slot-surviving",
            "reason": "static Schwarzschild lapse",
        },
        {
            "slot": "h_00",
            "order": "sqrt(epsilon)*epsilon_w",
            "term": "2*a.b",
            "classification": "slot-surviving",
            "reason": "stationary temporal shift leaves g_00 unchanged; generic wake contraction is nonzero",
        },
        {
            "slot": "h_0i",
            "order": "sqrt(epsilon)",
            "term": "-a_i",
            "classification": "gauge-removable",
            "reason": "static radial GP part is removed by the temporal lambda shift",
        },
        {
            "slot": "h_0i",
            "order": "epsilon_w",
            "term": "-b_i longitudinal part",
            "classification": "gauge-removable",
            "reason": "the integrable part is absorbed by lambda",
        },
        {
            "slot": "h_0i",
            "order": "epsilon_w",
            "term": "-b_i transverse part",
            "classification": "slot-surviving",
            "reason": "curl obstruction prevents a scalar lambda",
        },
        {
            "slot": "h_0i",
            "order": "epsilon*epsilon_w",
            "term": "-[a^2*b_i + 2*(a.b)*a_i] longitudinal part",
            "classification": "gauge-removable",
            "reason": "nonlinear term in grad(lambda)=-u/(1-u^2)",
        },
        {
            "slot": "h_0i",
            "order": "epsilon*epsilon_w",
            "term": "-[a^2*b_i + 2*(a.b)*a_i] transverse part",
            "classification": "slot-surviving",
            "reason": "the two-piece obstruction is the integrability condition",
        },
        {
            "slot": "h_ij",
            "order": "epsilon",
            "term": "a_i*a_j",
            "classification": "slot-surviving",
            "reason": "static Schwarzschild spatial curvature",
        },
        {
            "slot": "h_ij",
            "order": "sqrt(epsilon)*epsilon_w",
            "term": "a_i*b_j + b_i*a_j",
            "classification": "slot-surviving",
            "reason": "generic part has nonzero linearized curvature and is not a spatial pure gauge",
        },
    ]
    passed = all(sp.simplify(value) == 0 for value in raw_checks)
    passed = passed and all(sp.simplify(value) == 0 for value in diagonal_checks)
    passed = passed and vector_is_zero(
        generator_correction - expected_generator_correction
    )
    passed = passed and hazard_survives
    return {
        "name": "half_order_bookkeeping",
        "passed": passed,
        "details": {
            "small_parameters": {
                "p": "sqrt(epsilon)",
                "epsilon_w": "U/c",
                "u_over_c": "p*a + epsilon_w*b",
            },
            "classifications_through_epsilon_epsilon_w": classifications,
            "direct_metric_coefficients_at_epsilon_epsilon_w": {
                "h_00": "0",
                "h_0i": ["0", "0", "0"],
                "h_ij": [["0"] * 3 for _ in range(3)],
                "note": "epsilon*epsilon_w enters the nonlinear temporal generator and its obstruction, not a new monomial of the PG metric for delta-v=O(epsilon_w)",
            },
            "temporal_generator_epsilon_epsilon_w": expression_list(
                generator_correction
            ),
            "curvature_witness": {
                "a": ["x", "y", "z"],
                "b": ["x", "2*y", "3*z"],
                "both_leading_flows_are_potential": True,
                "R_0i0j_at_sqrt_epsilon_epsilon_w": [
                    expression_list(curvature_witness.row(index)) for index in range(3)
                ],
                "nonzero": hazard_survives,
            },
            "hazard_verdict": "KILL: a generic wake has observable sqrt(epsilon)*epsilon_w curvature; survival requires extra cancellations in the realized wake",
            "super_ppn_kill": hazard_survives,
        },
    }


def run_checks() -> dict:
    gates = [
        statement_zero_gate(),
        pure_wind_flatness_gate(),
        diagonalization_gate(),
        obstruction_gate(),
        half_order_gate(),
    ]
    return {
        "schema_version": 1,
        "sim": SLUG,
        "task": TASK_ID,
        "engine": f"sympy-{sp.__version__}",
        "all_passed": all(gate["passed"] for gate in gates),
        "hazard_verdict": gates[-1]["details"]["hazard_verdict"],
        "gates": gates,
    }


def encode(result: dict) -> bytes:
    return (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="execute every symbolic gate twice and require byte-identical JSON",
    )
    arguments = parser.parse_args()

    result = run_checks()
    payload = encode(result)
    if arguments.check_determinism:
        repeated = encode(run_checks())
        if payload != repeated:
            raise SystemExit("determinism gate failed: repeated JSON bytes differ")

    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_bytes(payload)
    for gate in result["gates"]:
        print(f"{'PASS' if gate['passed'] else 'FAIL'}  {gate['name']}")
    print(result["hazard_verdict"])
    if arguments.check_determinism:
        print("PASS  byte-identical in-process rerun")
    print(f"wrote {RESULT_PATH.relative_to(Path.cwd())}")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
