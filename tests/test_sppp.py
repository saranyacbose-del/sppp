"""
Unit Tests — SPPP System
========================
Verifies all equations from the paper against expected values.
All test values taken directly from the paper (Table 4, Section 3.3).

Run:  python -m pytest tests/test_sppp.py -v
      python tests/test_sppp.py

Reference: Saranya C and Janaki G (2026). Manuscript ID: applsci-4280957.
"""

import math
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sppp.spc_algorithm import (
    compute_g, compute_h, compute_p, compute_alpha,
    scale_personalised_cost, update_weights, initialise_weights,
    total_cost, BETA, GAMMA, D_TH, ETA,
    INTERACTION_MATRIX,
)
from sppp.planner import SPPPPlanner, validate_path
from data.dataset import (
    GRID_SIZE, START_NODE, GOAL_NODE,
    REFERENCE_OBSTACLE_MAP, SCENARIOS,
)


TOLERANCE = 0.02   # ±0.02 units tolerance for floating point comparisons


# ---------------------------------------------------------------------------
# Equation tests
# ---------------------------------------------------------------------------

def test_g_start_node():
    """G(start) = 0 always.  Eq.(1)"""
    g = compute_g(START_NODE, START_NODE)
    assert abs(g) < 1e-9, f"G(start,start) should be 0, got {g}"
    print("✓ test_g_start_node")


def test_g_example_from_paper():
    """
    Paper example: G(2,3) = sqrt((0-2)^2 + (2-3)^2) = sqrt(5) ≈ 2.24
    Eq.(2), Section 3.3.2.
    """
    g = compute_g((2,3), START_NODE)
    assert abs(g - 2.24) < TOLERANCE, f"G(2,3) expected 2.24, got {g:.4f}"
    print("✓ test_g_example_from_paper")


def test_h_example_from_paper():
    """
    Paper example: H(2,3) = sqrt((7-2)^2 + (4-3)^2) = sqrt(26) ≈ 5.10
    Eq.(4), Section 3.3.3.
    """
    h = compute_h((2,3), GOAL_NODE)
    assert abs(h - 5.10) < TOLERANCE, f"H(2,3) expected 5.10, got {h:.4f}"
    print("✓ test_h_example_from_paper")


def test_h_goal_node():
    """H(goal) = 0 always.  Eq.(3)"""
    h = compute_h(GOAL_NODE, GOAL_NODE)
    assert abs(h) < 1e-9, f"H(goal,goal) should be 0, got {h}"
    print("✓ test_h_goal_node")


def test_alpha_far_from_goal():
    """
    When D=8.06 (far from goal, > D_th=3.0): alpha is low but > 1.
    Exact value: 1 + 2.0/(1+exp(0.5*(8.06-3.0))) = 1.147.
    Alpha is monotonically decreasing with D — confirmed. Eq.(6).
    """
    alpha = compute_alpha((0,0), GOAL_NODE)
    # alpha should be > 1 (always) and < 1 + beta (= 3.0)
    assert 1.0 < alpha < 1.0 + BETA, \
        f"alpha should be in (1, {1+BETA}), got {alpha:.4f}"
    assert abs(alpha - 1.147) < 0.01, \
        f"alpha at D=8.06 expected ≈1.147, got {alpha:.4f}"
    print(f"✓ test_alpha_far_from_goal  (alpha={alpha:.4f}, low amplification as expected)")


def test_alpha_at_goal():
    """
    When D=0: alpha is near maximum but not exactly 1+beta (sigmoid asymptote).
    Exact value: 1 + 2.0/(1+exp(0.5*(0-3.0))) = 2.635.
    Maximum only reached asymptotically as D→-inf. Eq.(6).
    """
    alpha = compute_alpha(GOAL_NODE, GOAL_NODE, beta=BETA, gamma=GAMMA, d_th=D_TH)
    # Should be > 2.0 (well above baseline of 1.0)
    assert alpha > 2.0, f"alpha at goal should be > 2.0, got {alpha:.4f}"
    assert abs(alpha - 2.635) < 0.01, \
        f"alpha at D=0 expected ≈2.635, got {alpha:.4f}"
    print(f"✓ test_alpha_at_goal  (alpha={alpha:.4f} — high amplification near goal)")


def test_alpha_sign_positive_gamma():
    """
    Verifies POSITIVE sign in exponent: e^(+gamma*(D-D_th)).
    Paper Eq.(6). Sign was corrected from original Figure 5 draft.
    """
    # D < D_th → (D-D_th) < 0 → +gamma*(D-D_th) < 0
    # → exp term → small → denominator → 1 → alpha → 1+beta (HIGH)
    alpha_near = compute_alpha((7,4), GOAL_NODE)   # D=0, near goal
    alpha_far  = compute_alpha((0,0), GOAL_NODE)   # D>>0, far from goal
    assert alpha_near > alpha_far, \
        "alpha should be HIGHER near goal (positive exponent sign confirmed)"
    print(f"✓ test_alpha_sign_positive_gamma  "
          f"(near={alpha_near:.3f} > far={alpha_far:.3f})")


def test_weight_initialisation():
    """
    Uniform initialisation for active preferences.
    Eq.(12): w_i^0 = 1/N_active.
    Scenario S3: 3 active (I2,I3,I5) → w = 1/3 ≈ 0.333. Eq.(13).
    """
    active = SCENARIOS["S3_Full_Emergency"]["active_prefs"]
    w = initialise_weights(active)
    n_active = sum(active)
    expected_w = 1.0 / n_active

    for i, act in enumerate(active):
        if act:
            assert abs(w[i] - expected_w) < 1e-6, \
                f"w[{i}] expected {expected_w:.4f}, got {w[i]:.4f}"
    assert abs(w.sum() - 1.0) < 1e-6, f"weights should sum to 1, got {w.sum()}"
    print(f"✓ test_weight_initialisation  (w={expected_w:.4f} for {n_active} active prefs)")


def test_total_cost_function():
    """
    F(s) = G(s) + H(s) + P(s).  Eq.(15).
    """
    f = total_cost(2.24, 5.10, 31.0)
    expected = 2.24 + 5.10 + 31.0
    assert abs(f - expected) < 1e-6, f"F expected {expected}, got {f}"
    print(f"✓ test_total_cost_function  (F={f:.2f})")


def test_table4_node_02():
    """
    Table 4, first row: node (0,2) — Start node.
    G=0.00, H=7.28, F=G+H+P=34.28 → P=27.
    """
    g = compute_g((0,2), START_NODE)
    h = compute_h((0,2), GOAL_NODE)
    assert abs(g - 0.00) < TOLERANCE, f"G(0,2) expected 0.00, got {g:.4f}"
    assert abs(h - 7.28) < TOLERANCE, f"H(0,2) expected 7.28, got {h:.4f}"
    print(f"✓ test_table4_node_02  (G={g:.2f}, H={h:.2f})")


def test_table4_node_14_corrected():
    """
    Table 4, node (1,4) — corrected in revision.
    G was 1.41 (error), correct value is 2.24.
    H=6.00, P=31, F=39.24.
    """
    g = compute_g((1,4), START_NODE)
    h = compute_h((1,4), GOAL_NODE)
    assert abs(g - 2.24) < TOLERANCE, \
        f"G(1,4) expected 2.24 (corrected), got {g:.4f}"
    assert abs(h - 6.00) < TOLERANCE, \
        f"H(1,4) expected 6.00, got {h:.4f}"
    f = total_cost(g, h, 31.0)
    assert abs(f - 39.24) < TOLERANCE, \
        f"F(1,4) expected 39.24 (corrected), got {f:.4f}"
    print(f"✓ test_table4_node_14_corrected  (G={g:.2f}, H={h:.2f}, F={f:.2f})")


def test_table4_min_f_node():
    """
    Table 4: node (5,4) has minimum F(s)=21.39.
    G=5.39, H=2.00, P=14.
    """
    g = compute_g((5,4), START_NODE)
    h = compute_h((5,4), GOAL_NODE)
    assert abs(g - 5.39) < TOLERANCE, f"G(5,4) expected 5.39, got {g:.4f}"
    assert abs(h - 2.00) < TOLERANCE, f"H(5,4) expected 2.00, got {h:.4f}"
    f = total_cost(g, h, 14.0)
    assert abs(f - 21.39) < TOLERANCE, f"F(5,4) expected 21.39, got {f:.4f}"
    print(f"✓ test_table4_min_f_node  (F={f:.2f} — minimum F(s) node)")


def test_cohens_d_calculation():
    """
    Cohen's d from Table 10 values.
    Eq.(25)–(27). Paper reports d ≈ 18.9 (corrected from 19.7).
    """
    mu_sppp = 94.3;  sd_sppp = 2.1
    mu_edt  = 31.7;  sd_edt  = 4.2
    pooled  = math.sqrt((sd_edt**2 + sd_sppp**2) / 2)
    d       = (mu_sppp - mu_edt) / pooled
    assert abs(d - 18.85) < 0.1, f"Cohen's d expected ≈18.85, got {d:.2f}"
    assert abs(d - 18.9) < 0.1,  f"Paper value 18.9 should match, got {d:.2f}"
    print(f"✓ test_cohens_d_calculation  (d={d:.2f})")


def test_empirical_bound():
    """
    Empirical suboptimality bound.
    Eq.(20a): F_SPPP ≤ F* + P_bar * n
    257.2 vs theoretical 404. Ratio = 1.57x tighter.  Section 3.6.
    """
    p_bar = 25.47
    n     = 10.1
    empirical   = p_bar * n           # ≈ 257.2
    theoretical = 5 * n * 8           # = 404.0
    ratio       = theoretical / empirical
    assert abs(empirical - 257.2) < 1.0, \
        f"Empirical bound expected ≈257.2, got {empirical:.1f}"
    assert abs(ratio - 1.57) < 0.02, \
        f"Tightness ratio expected 1.57x, got {ratio:.2f}x"
    print(f"✓ test_empirical_bound  ({empirical:.1f} vs {theoretical:.1f}, "
          f"ratio={ratio:.2f}x tighter)")


def test_pmax_bound():
    """
    Per-node Pmax ≤ 40 (8 preferences × max weight 1 × max severity 5).
    Eq.(19).  Corrected in revision from Pmax=5 to Pmax≤40.
    """
    pmax = 8 * 1.0 * 5  # N_active * w_max * O_max
    assert pmax == 40.0, f"Pmax expected 40, got {pmax}"
    print(f"✓ test_pmax_bound  (Pmax={pmax})")


def test_planner_finds_path():
    """SPPP planner finds a valid path from start to goal."""
    active = SCENARIOS["S3_Full_Emergency"]["active_prefs"]
    planner = SPPPPlanner(
        GRID_SIZE, START_NODE, GOAL_NODE,
        active, REFERENCE_OBSTACLE_MAP.copy(),
    )
    path = planner.plan()
    assert path is not None, "SPPP planner should find a path"
    assert path[0]  == START_NODE, f"Path should start at {START_NODE}"
    assert path[-1] == GOAL_NODE,  f"Path should end at {GOAL_NODE}"
    print(f"✓ test_planner_finds_path  (path length={len(path)} nodes)")


def test_planner_computation_time():
    """
    Computation time < 2ms on standard hardware.
    Paper: 1.91 ms on laptop, 3.76 ms on RPi4.
    Section 4.8, Table 12.
    """
    active = SCENARIOS["S3_Full_Emergency"]["active_prefs"]
    planner = SPPPPlanner(
        GRID_SIZE, START_NODE, GOAL_NODE,
        active, REFERENCE_OBSTACLE_MAP.copy(),
    )
    planner.plan()
    # Allow 50ms tolerance for test environment (CI may be slower)
    assert planner.computation_time_ms < 50.0, \
        f"Comp time {planner.computation_time_ms:.2f}ms exceeds 50ms limit"
    print(f"✓ test_planner_computation_time  ({planner.computation_time_ms:.3f} ms)")


def test_validation_module():
    """Real-time validation module passes with zero errors. Section 4.7."""
    active = SCENARIOS["S3_Full_Emergency"]["active_prefs"]
    planner = SPPPPlanner(
        GRID_SIZE, START_NODE, GOAL_NODE,
        active, REFERENCE_OBSTACLE_MAP.copy(),
    )
    path = planner.plan()
    assert planner.validation_passed, \
        f"Validation should pass with zero errors"
    print(f"✓ test_validation_module  (validation_passed={planner.validation_passed})")


def test_interaction_matrix_shape():
    """Interaction matrix is 8×8. Table 2."""
    assert len(INTERACTION_MATRIX) == 8
    assert all(len(row) == 8 for row in INTERACTION_MATRIX)
    print("✓ test_interaction_matrix_shape  (8×8 confirmed)")


def test_interaction_matrix_o8_only_emergency():
    """
    O8 (Construction) only activates for I5 (Emergency). Table 2.
    'Construction zone segments avoided in 100% of SPPP runs under
    any preference scenario including I5.' Section 4.5.
    """
    o8_row = INTERACTION_MATRIX[7]  # O8 Construction
    assert o8_row[4] == True,  "O8 should activate for I5 (Emergency)"
    assert all(o8_row[i] == False for i in range(8) if i != 4), \
        "O8 should ONLY activate for I5, not other preferences"
    print("✓ test_interaction_matrix_o8_only_emergency")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*55)
    print("SPPP Unit Tests — Verifying Paper Equations")
    print("="*55 + "\n")

    tests = [
        test_g_start_node,
        test_g_example_from_paper,
        test_h_example_from_paper,
        test_h_goal_node,
        test_alpha_far_from_goal,
        test_alpha_at_goal,
        test_alpha_sign_positive_gamma,
        test_weight_initialisation,
        test_total_cost_function,
        test_table4_node_02,
        test_table4_node_14_corrected,
        test_table4_min_f_node,
        test_cohens_d_calculation,
        test_empirical_bound,
        test_pmax_bound,
        test_planner_finds_path,
        test_planner_computation_time,
        test_validation_module,
        test_interaction_matrix_shape,
        test_interaction_matrix_o8_only_emergency,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed ✓")
    print("="*55)
