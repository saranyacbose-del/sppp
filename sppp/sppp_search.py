"""
Semantic Personalised Path Planning (SPPP) — A* Search Engine
=============================================================
Implements the improved A* search loop integrating the SPC cost function.

    F(s) = G(s) + H(s) + P(s)

The search selects the minimum F(s) node from the open list at each
iteration and reconstructs the optimal personalised path via parent pointers.

Author      : Saranya C, Janaki G
Institution : SRM Institute of Science and Technology, Kattankulathur
"""

import heapq
import time
import math
from typing import Dict, List, Tuple, Optional, Set

from sppp.spc_algorithm import (
    compute_G, compute_H, compute_P, compute_F,
    update_weights, initialise_weights, validate_obstacle_scores,
    euclidean_distance, OBSTACLE_KEYS
)


class SPPPResult:
    """
    Container for SPPP search results.

    Attributes
    ----------
    path : list of tuple
        Ordered list of node coordinates from start to goal.
    total_cost : float
        Total F(s) cost accumulated along path.
    nodes_expanded : int
        Number of nodes expanded during search.
    computation_time_ms : float
        Wall-clock computation time in milliseconds.
    g_values : dict
        G(s) value at each path node.
    h_values : dict
        H(s) value at each path node.
    p_values : dict
        P(s) value at each path node.
    f_values : dict
        F(s) value at each path node.
    weights : dict
        Final preference weights after dynamic evolution.
    success : bool
        Whether a valid path was found.
    """

    def __init__(self):
        self.path = []
        self.total_cost = 0.0
        self.nodes_expanded = 0
        self.computation_time_ms = 0.0
        self.g_values = {}
        self.h_values = {}
        self.p_values = {}
        self.f_values = {}
        self.weights = {}
        self.success = False

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "=" * 60,
            "SPPP Search Result Summary",
            "=" * 60,
            f"Success          : {self.success}",
            f"Path length      : {len(self.path)} nodes",
            f"Total F(s) cost  : {self.total_cost:.2f} units",
            f"Nodes expanded   : {self.nodes_expanded}",
            f"Computation time : {self.computation_time_ms:.3f} ms",
            f"Path             : {self.path}",
            "-" * 60,
            "Cost breakdown per node:",
            f"{'Node':<12} {'G(s)':>8} {'H(s)':>8} "
            f"{'P(s)':>8} {'F(s)':>8} {'P/F %':>8}",
            "-" * 60,
        ]
        for node in self.path:
            g = self.g_values.get(node, 0.0)
            h = self.h_values.get(node, 0.0)
            p = self.p_values.get(node, 0.0)
            f = self.f_values.get(node, 0.0)
            pf = (p / f * 100) if f > 0 else 0.0
            lines.append(
                f"{str(node):<12} {g:>8.2f} {h:>8.2f} "
                f"{p:>8.2f} {f:>8.2f} {pf:>7.1f}%"
            )
        lines.append("=" * 60)
        mean_p = (
            sum(self.p_values.get(n, 0) for n in self.path) / len(self.path)
            if self.path else 0
        )
        mean_f = (
            sum(self.f_values.get(n, 0) for n in self.path) / len(self.path)
            if self.path else 0
        )
        mean_pf = (mean_p / mean_f * 100) if mean_f > 0 else 0
        lines.append(f"Mean P(s) per node : {mean_p:.2f}")
        lines.append(f"Mean F(s) per node : {mean_f:.2f}")
        lines.append(f"Mean P(s)/F(s)     : {mean_pf:.1f}%")
        lines.append("=" * 60)
        return "\n".join(lines)


class SPPPSearch:
    """
    Semantic Personalised Path Planning (SPPP) A* Search Engine.

    Implements the three-phase SPPP system:
    - Phase 1: User input and obstacle map integration
    - Phase 2: SPC-based route map calculation
    - Phase 3: Total cost optimisation and path selection

    Parameters
    ----------
    grid_size : tuple of int
        Grid dimensions (rows, cols). Default: (8, 8).
    start : tuple of int
        Start node coordinates (row, col). Default: (0, 2).
    goal : tuple of int
        Goal node coordinates (row, col). Default: (7, 4).
    obstacle_map : dict
        Per-node obstacle severity scores {(row,col): {O1:val,...,O8:val}}.
    active_prefs : list of str
        Active user preference keys (e.g. ['I2', 'I3', 'I5']).

    Examples
    --------
    >>> import numpy as np
    >>> obs_map = {(r,c): {k: np.random.uniform(0,5) for k in
    ...            ['O1','O2','O3','O4','O5','O6','O7','O8']}
    ...            for r in range(8) for c in range(8)}
    >>> planner = SPPPSearch(
    ...     grid_size=(8,8), start=(0,2), goal=(7,4),
    ...     obstacle_map=obs_map, active_prefs=['I2','I3','I5']
    ... )
    >>> result = planner.search()
    >>> print(result.summary())
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (8, 8),
                 start: Tuple[int, int] = (0, 2),
                 goal: Tuple[int, int] = (7, 4),
                 obstacle_map: Optional[Dict] = None,
                 active_prefs: Optional[List[str]] = None):

        self.rows, self.cols = grid_size
        self.start = start
        self.goal = goal
        self.obstacle_map = obstacle_map or {}
        self.active_prefs = active_prefs or ['I2', 'I3', 'I5']
        self.weights = initialise_weights(self.active_prefs)

    def _get_neighbours(self,
                        node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid 8-connected grid neighbours.

        Parameters
        ----------
        node : tuple of int
            Current node (row, col).

        Returns
        -------
        list of tuple
            Valid neighbour coordinates within grid bounds.
        """
        r, c = node
        neighbours = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbours.append((nr, nc))
        return neighbours

    def _get_obstacle_scores(self,
                             node: Tuple[int, int]) -> Dict[str, float]:
        """Retrieve and validate obstacle scores for a node."""
        raw = self.obstacle_map.get(node, {k: 0.0 for k in OBSTACLE_KEYS})
        return validate_obstacle_scores(raw)

    def search(self) -> SPPPResult:
        """
        Execute the SPPP A* search.

        Implements the improved A* loop (Phase 3) using SPC cost function.
        Selects minimum F(s) node from open list at each iteration.
        Reconstructs path via parent pointers from goal to start.

        Returns
        -------
        SPPPResult
            Complete search result including path, costs, and timing.

        Notes
        -----
        Real-time replanning is triggered if obstacle severity exceeds 4.5
        at any mid-path node during validation. See validate_path() method.
        """
        t_start = time.perf_counter()
        result = SPPPResult()
        result.weights = dict(self.weights)

        # Open list: (F(s), node)
        open_list = []
        heapq.heappush(open_list, (0.0, self.start))

        # Cost tracking
        g_score = {self.start: 0.0}
        f_score = {self.start: 0.0}
        came_from = {self.start: None}

        # Per-node cost storage
        G_map, H_map, P_map, F_map = {}, {}, {}, {}

        closed_set: Set[Tuple[int, int]] = set()
        nodes_expanded = 0

        while open_list:
            _, current = heapq.heappop(open_list)

            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            # Goal reached
            if current == self.goal:
                path = self._reconstruct_path(came_from, current)
                result.path = path
                result.success = True
                result.nodes_expanded = nodes_expanded
                result.g_values = {n: G_map.get(n, 0.0) for n in path}
                result.h_values = {n: H_map.get(n, 0.0) for n in path}
                result.p_values = {n: P_map.get(n, 0.0) for n in path}
                result.f_values = {n: F_map.get(n, 0.0) for n in path}
                result.total_cost = sum(
                    F_map.get(n, 0.0) for n in path
                )
                # Dynamic weight update after path found
                result.weights = update_weights(
                    self.weights, path,
                    self.obstacle_map, self.active_prefs
                )
                t_end = time.perf_counter()
                result.computation_time_ms = (t_end - t_start) * 1000
                return result

            # Expand neighbours
            for neighbour in self._get_neighbours(current):
                if neighbour in closed_set:
                    continue

                obs = self._get_obstacle_scores(neighbour)
                G = compute_G(neighbour, self.start)
                H = compute_H(neighbour, self.goal)
                P = compute_P(
                    obs, self.weights,
                    self.active_prefs, H
                )
                F = compute_F(G, H, P)

                tentative_g = g_score.get(current, float('inf')) + \
                    euclidean_distance(
                        current[0], current[1],
                        neighbour[0], neighbour[1]
                    )

                if tentative_g < g_score.get(neighbour, float('inf')):
                    g_score[neighbour] = tentative_g
                    f_score[neighbour] = F
                    came_from[neighbour] = current
                    G_map[neighbour] = G
                    H_map[neighbour] = H
                    P_map[neighbour] = P
                    F_map[neighbour] = F
                    heapq.heappush(open_list, (F, neighbour))

        # No path found
        t_end = time.perf_counter()
        result.computation_time_ms = (t_end - t_start) * 1000
        result.nodes_expanded = nodes_expanded
        return result

    def _reconstruct_path(self,
                          came_from: Dict,
                          current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct path by backtracking parent pointers from goal to start.

        Parameters
        ----------
        came_from : dict
            Parent pointer dictionary from A* search.
        current : tuple
            Goal node to backtrack from.

        Returns
        -------
        list of tuple
            Path from start to goal.
        """
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        return path[::-1]

    def validate_path(self,
                      path: List[Tuple[int, int]],
                      spike_severity: float = 5.0,
                      spike_node_idx: Optional[int] = None,
                      severity_threshold: float = 4.5,
                      cost_threshold: float = 1.0) -> Dict:
        """
        Real-time path validation and replanning module.

        Injects an obstacle severity spike at a mid-path node and
        determines whether replanning is required based on:
        1. Spike severity >= severity_threshold (default 4.5)
        2. Cost difference ΔF >= cost_threshold (default 1.0)

        Parameters
        ----------
        path : list of tuple
            Current planned path.
        spike_severity : float
            Severity of injected obstacle spike (0-5). Default: 5.0.
        spike_node_idx : int, optional
            Index of spike injection node. Default: mid-path.
        severity_threshold : float
            Minimum severity to trigger replan check. Default: 4.5.
        cost_threshold : float
            Minimum ΔF to commit replan. Default: 1.0.

        Returns
        -------
        dict
            Validation result with keys:
            - replan_triggered (bool)
            - route_changed (bool)
            - replan_time_ms (float)
            - new_path (list)
            - spike_node (tuple)
            - delta_F (float)
        """
        if spike_node_idx is None:
            spike_node_idx = len(path) // 2

        spike_node = path[spike_node_idx]
        result = {
            'replan_triggered': False,
            'route_changed': False,
            'replan_time_ms': 0.0,
            'new_path': path,
            'spike_node': spike_node,
            'delta_F': 0.0
        }

        # Check severity threshold
        if spike_severity < severity_threshold:
            return result

        result['replan_triggered'] = True

        # Inject spike into obstacle map
        updated_map = {k: dict(v) for k, v in self.obstacle_map.items()}
        if spike_node not in updated_map:
            updated_map[spike_node] = {k: 0.0 for k in OBSTACLE_KEYS}
        for obs_key in OBSTACLE_KEYS:
            updated_map[spike_node][obs_key] = spike_severity

        # Replan with updated map
        t_start = time.perf_counter()
        new_planner = SPPPSearch(
            grid_size=(self.rows, self.cols),
            start=self.start,
            goal=self.goal,
            obstacle_map=updated_map,
            active_prefs=self.active_prefs
        )
        new_result = new_planner.search()
        t_end = time.perf_counter()
        result['replan_time_ms'] = (t_end - t_start) * 1000

        # Compute cost difference
        old_cost = sum(
            compute_F(
                compute_G(n, self.start),
                compute_H(n, self.goal),
                compute_P(
                    validate_obstacle_scores(
                        self.obstacle_map.get(n, {})
                    ),
                    self.weights, self.active_prefs,
                    compute_H(n, self.goal)
                )
            ) for n in path
        )
        new_cost = new_result.total_cost
        delta_F = abs(new_cost - old_cost)
        result['delta_F'] = delta_F

        # Commit replan only if cost difference exceeds threshold
        if delta_F >= cost_threshold and new_result.path != path:
            result['route_changed'] = True
            result['new_path'] = new_result.path

        return result
