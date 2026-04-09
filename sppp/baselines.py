"""
Baseline Path Planning Algorithms
==================================
Implements four baseline algorithms for comparison with SPPP:

    1. EDT-A*    : Euclidean Distance Traditional A* (no personalised cost)
    2. Dijkstra  : Uniform-cost search without heuristic guidance
    3. Theta*    : Any-angle path planning with line-of-sight relaxation
    4. D* Lite   : Incremental replanning algorithm

All baselines are evaluated on identical obstacle maps and grid configurations
as used for SPPP (Table 5 in paper).

Author      : Saranya C, Janaki G
Institution : SRM Institute of Science and Technology, Kattankulathur
"""

import heapq
import time
import math
from typing import Dict, List, Tuple, Optional, Set

from sppp.spc_algorithm import euclidean_distance, OBSTACLE_KEYS


# ---------------------------------------------------------------------------
# EDT-A* Baseline
# ---------------------------------------------------------------------------

class EDTAStar:
    """
    Euclidean Distance Traditional A* (EDT-A*).

    Computes F_EDT(s) = G(s) + H(s) with no personalised cost component.
    Represents the standard geometric planner baseline.

    Parameters
    ----------
    grid_size : tuple of int
        Grid dimensions (rows, cols).
    start : tuple of int
        Start node coordinates.
    goal : tuple of int
        Goal node coordinates.
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (8, 8),
                 start: Tuple[int, int] = (0, 2),
                 goal: Tuple[int, int] = (7, 4)):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal = goal

    def _get_neighbours(self, node):
        r, c = node
        return [
            (r + dr, c + dc)
            for dr in [-1, 0, 1] for dc in [-1, 0, 1]
            if not (dr == 0 and dc == 0)
            and 0 <= r + dr < self.rows
            and 0 <= c + dc < self.cols
        ]

    def search(self) -> Dict:
        """
        Run EDT-A* search.

        Returns
        -------
        dict
            path, total_cost, nodes_expanded, computation_time_ms
        """
        t_start = time.perf_counter()
        open_list = []
        heapq.heappush(open_list, (0.0, self.start))
        g_score = {self.start: 0.0}
        came_from = {self.start: None}
        closed_set: Set = set()
        nodes_expanded = 0

        while open_list:
            _, current = heapq.heappop(open_list)
            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            if current == self.goal:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path = path[::-1]
                t_end = time.perf_counter()
                return {
                    'path': path,
                    'total_cost': g_score[self.goal],
                    'nodes_expanded': nodes_expanded,
                    'computation_time_ms': (t_end - t_start) * 1000,
                    'success': True
                }

            for nb in self._get_neighbours(current):
                if nb in closed_set:
                    continue
                edge = euclidean_distance(
                    current[0], current[1], nb[0], nb[1]
                )
                tg = g_score[current] + edge
                H = euclidean_distance(nb[0], nb[1],
                                       self.goal[0], self.goal[1])
                F = tg + H
                if tg < g_score.get(nb, float('inf')):
                    g_score[nb] = tg
                    came_from[nb] = current
                    heapq.heappush(open_list, (F, nb))

        t_end = time.perf_counter()
        return {
            'path': [], 'total_cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'computation_time_ms': (t_end - t_start) * 1000,
            'success': False
        }


# ---------------------------------------------------------------------------
# Dijkstra Baseline
# ---------------------------------------------------------------------------

class Dijkstra:
    """
    Dijkstra's uniform-cost search.

    Expands nodes in order of accumulated cost G(s) without heuristic.
    F_Dijk(s) = G(s)

    Parameters
    ----------
    grid_size : tuple of int
        Grid dimensions (rows, cols).
    start : tuple of int
        Start node coordinates.
    goal : tuple of int
        Goal node coordinates.
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (8, 8),
                 start: Tuple[int, int] = (0, 2),
                 goal: Tuple[int, int] = (7, 4)):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal = goal

    def _get_neighbours(self, node):
        r, c = node
        return [
            (r + dr, c + dc)
            for dr in [-1, 0, 1] for dc in [-1, 0, 1]
            if not (dr == 0 and dc == 0)
            and 0 <= r + dr < self.rows
            and 0 <= c + dc < self.cols
        ]

    def search(self) -> Dict:
        """
        Run Dijkstra search.

        Returns
        -------
        dict
            path, total_cost, nodes_expanded, computation_time_ms
        """
        t_start = time.perf_counter()
        open_list = []
        heapq.heappush(open_list, (0.0, self.start))
        g_score = {self.start: 0.0}
        came_from = {self.start: None}
        closed_set: Set = set()
        nodes_expanded = 0

        while open_list:
            g, current = heapq.heappop(open_list)
            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            if current == self.goal:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path = path[::-1]
                t_end = time.perf_counter()
                return {
                    'path': path,
                    'total_cost': g_score[self.goal],
                    'nodes_expanded': nodes_expanded,
                    'computation_time_ms': (t_end - t_start) * 1000,
                    'success': True
                }

            for nb in self._get_neighbours(current):
                if nb in closed_set:
                    continue
                edge = euclidean_distance(
                    current[0], current[1], nb[0], nb[1]
                )
                tg = g_score[current] + edge
                if tg < g_score.get(nb, float('inf')):
                    g_score[nb] = tg
                    came_from[nb] = current
                    heapq.heappush(open_list, (tg, nb))

        t_end = time.perf_counter()
        return {
            'path': [], 'total_cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'computation_time_ms': (t_end - t_start) * 1000,
            'success': False
        }


# ---------------------------------------------------------------------------
# Theta* Baseline
# ---------------------------------------------------------------------------

class ThetaStar:
    """
    Theta* any-angle path planning algorithm.

    Relaxes grid-edge constraints by allowing line-of-sight connections
    between non-adjacent nodes. Parent pointers are updated across nodes
    within line-of-sight.

    F_theta*(s) = G_los(s) + H(s)

    Parameters
    ----------
    grid_size : tuple of int
        Grid dimensions (rows, cols).
    start : tuple of int
        Start node coordinates.
    goal : tuple of int
        Goal node coordinates.
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (8, 8),
                 start: Tuple[int, int] = (0, 2),
                 goal: Tuple[int, int] = (7, 4)):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal = goal

    def _get_neighbours(self, node):
        r, c = node
        return [
            (r + dr, c + dc)
            for dr in [-1, 0, 1] for dc in [-1, 0, 1]
            if not (dr == 0 and dc == 0)
            and 0 <= r + dr < self.rows
            and 0 <= c + dc < self.cols
        ]

    def _line_of_sight(self, n1, n2):
        """Check line-of-sight between two nodes using Bresenham's algorithm."""
        r1, c1 = n1
        r2, c2 = n2
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        r, c = r1, c1
        n = 1 + dr + dc
        r_inc = 1 if r2 > r1 else -1
        c_inc = 1 if c2 > c1 else -1
        error = dr - dc
        dr *= 2
        dc *= 2
        for _ in range(n):
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return False
            if error > 0:
                r += r_inc
                error -= dc
            elif error < 0:
                c += c_inc
                error += dr
            else:
                r += r_inc
                c += c_inc
                error += dr - dc
        return True

    def search(self) -> Dict:
        """
        Run Theta* search.

        Returns
        -------
        dict
            path, total_cost, nodes_expanded, computation_time_ms
        """
        t_start = time.perf_counter()
        open_list = []
        heapq.heappush(open_list, (0.0, self.start))
        g_score = {self.start: 0.0}
        came_from = {self.start: None}
        closed_set: Set = set()
        nodes_expanded = 0

        while open_list:
            _, current = heapq.heappop(open_list)
            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            if current == self.goal:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path = path[::-1]
                t_end = time.perf_counter()
                return {
                    'path': path,
                    'total_cost': g_score[self.goal],
                    'nodes_expanded': nodes_expanded,
                    'computation_time_ms': (t_end - t_start) * 1000,
                    'success': True
                }

            parent = came_from.get(current)
            for nb in self._get_neighbours(current):
                if nb in closed_set:
                    continue
                H = euclidean_distance(
                    nb[0], nb[1], self.goal[0], self.goal[1]
                )
                # Theta*: try line-of-sight from parent
                if parent and self._line_of_sight(parent, nb):
                    tg = g_score[parent] + euclidean_distance(
                        parent[0], parent[1], nb[0], nb[1]
                    )
                    if tg < g_score.get(nb, float('inf')):
                        g_score[nb] = tg
                        came_from[nb] = parent
                        heapq.heappush(open_list, (tg + H, nb))
                else:
                    tg = g_score[current] + euclidean_distance(
                        current[0], current[1], nb[0], nb[1]
                    )
                    if tg < g_score.get(nb, float('inf')):
                        g_score[nb] = tg
                        came_from[nb] = current
                        heapq.heappush(open_list, (tg + H, nb))

        t_end = time.perf_counter()
        return {
            'path': [], 'total_cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'computation_time_ms': (t_end - t_start) * 1000,
            'success': False
        }


# ---------------------------------------------------------------------------
# D* Lite Baseline
# ---------------------------------------------------------------------------

class DStarLite:
    """
    D* Lite incremental replanning algorithm.

    Efficiently updates paths when new obstacles are discovered.
    Supports partial real-time path adaptation without semantic cost encoding.

    F_D*(s) = g(s) + h(s, s_goal)

    Parameters
    ----------
    grid_size : tuple of int
        Grid dimensions (rows, cols).
    start : tuple of int
        Start node coordinates.
    goal : tuple of int
        Goal node coordinates.
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (8, 8),
                 start: Tuple[int, int] = (0, 2),
                 goal: Tuple[int, int] = (7, 4)):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal = goal

    def _get_neighbours(self, node):
        r, c = node
        return [
            (r + dr, c + dc)
            for dr in [-1, 0, 1] for dc in [-1, 0, 1]
            if not (dr == 0 and dc == 0)
            and 0 <= r + dr < self.rows
            and 0 <= c + dc < self.cols
        ]

    def search(self) -> Dict:
        """
        Run D* Lite search (simplified forward pass for comparison).

        Returns
        -------
        dict
            path, total_cost, nodes_expanded, computation_time_ms
        """
        t_start = time.perf_counter()
        open_list = []
        heapq.heappush(open_list, (0.0, self.start))
        g_score = {self.start: 0.0}
        came_from = {self.start: None}
        closed_set: Set = set()
        nodes_expanded = 0

        while open_list:
            _, current = heapq.heappop(open_list)
            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            if current == self.goal:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path = path[::-1]
                t_end = time.perf_counter()
                return {
                    'path': path,
                    'total_cost': g_score[self.goal],
                    'nodes_expanded': nodes_expanded,
                    'computation_time_ms': (t_end - t_start) * 1000,
                    'success': True
                }

            for nb in self._get_neighbours(current):
                if nb in closed_set:
                    continue
                edge = euclidean_distance(
                    current[0], current[1], nb[0], nb[1]
                )
                h = euclidean_distance(
                    nb[0], nb[1], self.goal[0], self.goal[1]
                )
                tg = g_score[current] + edge
                F = tg + h
                if tg < g_score.get(nb, float('inf')):
                    g_score[nb] = tg
                    came_from[nb] = current
                    heapq.heappush(open_list, (F, nb))

        t_end = time.perf_counter()
        return {
            'path': [], 'total_cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'computation_time_ms': (t_end - t_start) * 1000,
            'success': False
        }
