"""
SPPP Planner — Semantic Personalised Path Planning
===================================================
Improved A* search engine augmented with the SPC cost term.
Implements Phase 2 and Phase 3 of the SPPP system architecture (Figure 1).

Reference: Saranya C and Janaki G (2026). Manuscript ID: applsci-4280957.
"""

import heapq
import math
import time
import numpy as np
from typing import List, Tuple, Dict, Optional

from sppp.spc_algorithm import (
    compute_g, compute_h, compute_p, compute_alpha,
    scale_personalised_cost, update_weights, initialise_weights,
    total_cost, ETA,
)


# ---------------------------------------------------------------------------
# Validation module  (Section 4.7 — Real-Time Replanning)
# ---------------------------------------------------------------------------

def validate_path(
    path: List[Tuple[int,int]],
    f_values: Dict[Tuple[int,int], float],
    obstacle_map: np.ndarray,
    spike_threshold: float = 4.5,
) -> Tuple[bool, List[str]]:
    """
    Real-time validation and replanning module.
    Checks: path continuity · F(s) cost bound · obstacle spike detection.

    Returns (is_valid, list_of_errors).
    Zero errors expected in nominal operation (confirmed in paper Section 4.7).
    """
    errors = []

    # 1. Path continuity — consecutive nodes must be adjacent (8-connected)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        if abs(r2-r1) > 1 or abs(c2-c1) > 1:
            errors.append(f"Continuity break between {path[i]} and {path[i+1]}")

    # 2. F(s) cost bound — no node should have F = 0 (degenerate path)
    for node in path:
        if node in f_values and f_values[node] < 0:
            errors.append(f"Negative F(s) at node {node}: {f_values[node]:.3f}")

    # 3. Obstacle spike detection
    for node in path:
        row, col = node
        aggregate = obstacle_map[row, col, :].sum()
        if aggregate / 8.0 >= spike_threshold:  # mean severity ≥ spike_threshold
            errors.append(
                f"Obstacle spike at {node}: aggregate={aggregate:.1f}, "
                f"mean={aggregate/8:.2f} ≥ threshold {spike_threshold}"
            )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# SPPP A* Planner
# ---------------------------------------------------------------------------

class SPPPPlanner:
    """
    Semantic Personalised Path Planning system.

    Phases implemented:
      Phase 1 — User input & obstacle map integration
      Phase 2 — SPC-based route map calculation
      Phase 3 — Total cost function computation & optimal path selection

    Parameters
    ----------
    grid_size    : (rows, cols), default (8, 8)
    start        : (row, col) start node
    goal         : (row, col) goal node
    active_prefs : list of 8 booleans — which preferences are active
    obstacle_map : ndarray (rows, cols, 8) — O1..O8 per cell, each 0–5
    eta          : learning rate for weight evolution (default 0.01)
    beta         : adaptive scaling magnitude (default 2.0)
    gamma        : adaptive scaling sensitivity (default 0.5)
    d_th         : adaptive scaling activation threshold (default 3.0)
    """

    def __init__(
        self,
        grid_size: Tuple[int,int],
        start: Tuple[int,int],
        goal: Tuple[int,int],
        active_prefs: List[bool],
        obstacle_map: np.ndarray,
        eta: float = ETA,
        beta: float = 2.0,
        gamma: float = 0.5,
        d_th: float = 3.0,
    ):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal  = goal
        self.active_prefs = active_prefs
        self.obstacle_map = obstacle_map
        self.eta   = eta
        self.beta  = beta
        self.gamma = gamma
        self.d_th  = d_th

        # Phase 1 — initialise weights uniformly  Eq.(12)
        self.weights = initialise_weights(active_prefs)

        # Result storage
        self.path:     List[Tuple[int,int]] = []
        self.f_values: Dict[Tuple[int,int], float] = {}
        self.computation_time_ms: float = 0.0
        self.nodes_expanded: int = 0
        self.validation_passed: bool = False

    # ------------------------------------------------------------------
    def _neighbours(self, node: Tuple[int,int]) -> List[Tuple[int,int]]:
        """8-connected grid neighbours within bounds."""
        r, c = node
        nbrs = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    nbrs.append((nr, nc))
        return nbrs

    # ------------------------------------------------------------------
    def plan(self, replan: bool = False) -> Optional[List[Tuple[int,int]]]:
        """
        Execute the SPPP improved A* search (Phase 2 & 3).

        Returns the optimal personalised path, or None if no path exists.
        Computation time is recorded for benchmarking (Table 7, Table 12).
        """
        t_start = time.perf_counter()

        # Open list: (F(s), node)
        open_list: List[Tuple[float, Tuple[int,int]]] = []
        heapq.heappush(open_list, (0.0, self.start))

        came_from:  Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {self.start: None}
        g_score:    Dict[Tuple[int,int], float] = {self.start: 0.0}
        f_record:   Dict[Tuple[int,int], float] = {}

        closed_set: set = set()
        self.nodes_expanded = 0

        while open_list:
            _, current = heapq.heappop(open_list)

            if current in closed_set:
                continue
            closed_set.add(current)
            self.nodes_expanded += 1

            # Termination: goal reached
            if current == self.goal:
                path = self._reconstruct_path(came_from)
                t_end = time.perf_counter()
                self.computation_time_ms = (t_end - t_start) * 1000.0
                self.path = path
                self.f_values = f_record

                # Real-time validation module
                self.validation_passed, errors = validate_path(
                    path, f_record, self.obstacle_map
                )

                # Dynamic weight evolution after each path (Eq. 10, 11)
                if path:
                    self.weights = update_weights(
                        self.weights, path, self.obstacle_map,
                        self.active_prefs, self.goal, self.eta
                    )
                return path

            for neighbour in self._neighbours(current):
                if neighbour in closed_set:
                    continue

                # Step 2 — Compute G(s)
                g_new = compute_g(neighbour, self.start)

                if g_new >= g_score.get(neighbour, float('inf')):
                    continue

                g_score[neighbour] = g_new

                # Step 2 — Compute H(s)
                h = compute_h(neighbour, self.goal)

                # Step 2 — Compute P(s)  Eq.(5)
                p = compute_p(
                    neighbour, self.obstacle_map,
                    self.weights, self.active_prefs
                )

                # Step 3 — Adaptive scaling  Eq.(6)(7)
                alpha = compute_alpha(
                    neighbour, self.goal,
                    self.beta, self.gamma, self.d_th
                )
                p_scaled = scale_personalised_cost(p, alpha)

                # Step 5 — Total cost  Eq.(15)
                f = total_cost(g_new, h, p_scaled)
                f_record[neighbour] = f

                came_from[neighbour] = current
                heapq.heappush(open_list, (f, neighbour))

        # No path found
        t_end = time.perf_counter()
        self.computation_time_ms = (t_end - t_start) * 1000.0
        return None

    # ------------------------------------------------------------------
    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int,int], Optional[Tuple[int,int]]]
    ) -> List[Tuple[int,int]]:
        """
        Reconstruct path by back-tracking parent pointers from goal to start.
        Section 3.5.
        """
        path = []
        node = self.goal
        while node is not None:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    def trigger_replan(
        self,
        spike_node: Tuple[int,int],
        spike_severity: float = 5.0,
        delta_f_threshold: float = 1.0,
        spike_threshold:   float = 4.5,
    ) -> Optional[List[Tuple[int,int]]]:
        """
        Inject an obstacle spike and conditionally replan.
        Section 4.7 — Real-Time Replanning Performance.

        Replanning is triggered only when BOTH conditions hold:
          (a) spike_severity >= spike_threshold (default 4.5)
          (b) delta_F >= delta_f_threshold (default 1.0 unit)
        """
        if spike_severity < spike_threshold:
            return self.path  # suppressed — below activation threshold

        # Inject spike into obstacle map (all categories at spike node)
        row, col = spike_node
        original = self.obstacle_map[row, col, :].copy()
        self.obstacle_map[row, col, :] = spike_severity

        # Replan
        new_path = self.plan(replan=True)

        # Check delta_F condition
        if new_path and self.path:
            old_cost = sum(self.f_values.get(n, 0) for n in self.path)
            new_cost = sum(self.f_values.get(n, 0) for n in new_path)
            if abs(new_cost - old_cost) < delta_f_threshold:
                # Restore and keep original path
                self.obstacle_map[row, col, :] = original
                return self.path

        return new_path

    # ------------------------------------------------------------------
    def get_cost_components(self) -> List[Dict]:
        """
        Return G(s), H(s), P(s), F(s) for each node along the path.
        Reproduces Table 4 in the paper.
        """
        records = []
        for node in self.path:
            g = compute_g(node, self.start)
            h = compute_h(node, self.goal)
            p = compute_p(node, self.obstacle_map, self.weights, self.active_prefs)
            alpha = compute_alpha(node, self.goal, self.beta, self.gamma, self.d_th)
            p_sc  = scale_personalised_cost(p, alpha)
            f     = total_cost(g, h, p_sc)
            records.append({
                "node": node,
                "G(s)": round(g, 2),
                "H(s)": round(h, 2),
                "P(s)": round(p_sc, 2),
                "F(s)": round(f, 2),
            })
        return records

    # ------------------------------------------------------------------
    def compute_mahc(self) -> float:
        """
        Mean Absolute Heading Change (MAHC) — path smoothness metric.
        Section 3.7 (new metric added in revision).
        Lower = smoother trajectory.
        """
        if len(self.path) < 3:
            return 0.0
        angles = []
        for i in range(1, len(self.path)-1):
            r0,c0 = self.path[i-1]
            r1,c1 = self.path[i]
            r2,c2 = self.path[i+1]
            v1 = (r1-r0, c1-c0)
            v2 = (r2-r1, c2-c1)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 > 0 and mag2 > 0:
                cos_a = max(-1, min(1, dot/(mag1*mag2)))
                angles.append(math.degrees(math.acos(cos_a)))
        return float(np.mean(angles)) if angles else 0.0

    def compute_feasibility(self, kappa_max: float = 1.0) -> float:
        """
        Trajectory feasibility: proportion of segments satisfying κ ≤ κ_max.
        Section 3.7. κ_max = 1.0 rad/node for ground vehicle navigation.
        """
        if len(self.path) < 3:
            return 1.0
        feasible = 0
        total = 0
        for i in range(1, len(self.path)-1):
            r0,c0 = self.path[i-1]
            r1,c1 = self.path[i]
            r2,c2 = self.path[i+1]
            v1 = (r1-r0, c1-c0)
            v2 = (r2-r1, c2-c1)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2+v1[1]**2)
            mag2 = math.sqrt(v2[0]**2+v2[1]**2)
            if mag1 > 0 and mag2 > 0:
                cos_a = max(-1, min(1, dot/(mag1*mag2)))
                kappa = math.acos(cos_a)
                if kappa <= kappa_max:
                    feasible += 1
                total += 1
        return feasible / total if total > 0 else 1.0

    def compute_mean_obstacle_severity(self) -> float:
        """
        Mean obstacle severity along path — dynamic interaction risk metric.
        Section 3.7. Independent of SPC formulation.
        """
        if not self.path:
            return 0.0
        severities = []
        for node in self.path:
            row, col = node
            mean_sev = self.obstacle_map[row, col, :].mean()
            severities.append(mean_sev)
        return float(np.mean(severities))
