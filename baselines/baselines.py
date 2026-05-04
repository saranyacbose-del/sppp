"""
Baseline Algorithms
===================
Implementations of all baseline algorithms compared against SPPP in the paper.

  1. EDT-A*      — Euclidean Distance Threshold A* (primary baseline)
  2. Dijkstra    — Dijkstra's shortest path
  3. Theta*      — Any-angle path planning
  4. D* Lite     — Incremental replanning
  5. WC-A*       — Weighted-Cost A* (semantic-aware, non-personalised baseline)
                   Added in revision to address Reviewer 1 Comment R1-3.

Reference: Saranya C and Janaki G (2026). Manuscript ID: applsci-4280957.
           Table 8, Section 4.1, Section 4.3.
"""

import heapq
import math
import time
import numpy as np
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def euclidean(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def neighbours_8(node, rows, cols):
    r, c = node
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            nr,nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield (nr, nc)


def reconstruct(came_from, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# 1. EDT-A* — Euclidean Distance Threshold A*
# ---------------------------------------------------------------------------
class EDTAStar:
    """
    Standard A* with Euclidean distance heuristic. No semantic awareness.
    F_EDT(s) = G(s) + H(s)   Eq.(21) in paper.
    Primary baseline throughout the paper.
    """
    def __init__(self, grid_size, start, goal):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal  = goal
        self.path:  List[Tuple[int,int]] = []
        self.nodes_expanded = 0
        self.computation_time_ms = 0.0

    def plan(self):
        t0 = time.perf_counter()
        open_list = [(0.0, self.start)]
        came_from  = {self.start: None}
        g_score    = {self.start: 0.0}
        closed_set = set()
        self.nodes_expanded = 0

        while open_list:
            _, cur = heapq.heappop(open_list)
            if cur in closed_set: continue
            closed_set.add(cur)
            self.nodes_expanded += 1

            if cur == self.goal:
                self.path = reconstruct(came_from, self.goal)
                self.computation_time_ms = (time.perf_counter()-t0)*1000
                return self.path

            for nb in neighbours_8(cur, self.rows, self.cols):
                g_new = euclidean(nb, self.start)
                if g_new >= g_score.get(nb, float('inf')): continue
                g_score[nb] = g_new
                f = g_new + euclidean(nb, self.goal)
                came_from[nb] = cur
                heapq.heappush(open_list, (f, nb))

        self.computation_time_ms = (time.perf_counter()-t0)*1000
        return None


# ---------------------------------------------------------------------------
# 2. Dijkstra
# ---------------------------------------------------------------------------
class Dijkstra:
    """
    Dijkstra's shortest path. No heuristic, no semantic awareness.
    F_Dijk(s) = G(s)   Eq.(22) in paper.
    """
    def __init__(self, grid_size, start, goal):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal  = goal
        self.path:  List[Tuple[int,int]] = []
        self.nodes_expanded = 0
        self.computation_time_ms = 0.0

    def plan(self):
        t0 = time.perf_counter()
        open_list = [(0.0, self.start)]
        came_from  = {self.start: None}
        dist       = {self.start: 0.0}
        closed_set = set()
        self.nodes_expanded = 0

        while open_list:
            d, cur = heapq.heappop(open_list)
            if cur in closed_set: continue
            closed_set.add(cur)
            self.nodes_expanded += 1

            if cur == self.goal:
                self.path = reconstruct(came_from, self.goal)
                self.computation_time_ms = (time.perf_counter()-t0)*1000
                return self.path

            for nb in neighbours_8(cur, self.rows, self.cols):
                new_d = dist[cur] + euclidean(cur, nb)
                if new_d >= dist.get(nb, float('inf')): continue
                dist[nb] = new_d
                came_from[nb] = cur
                heapq.heappush(open_list, (new_d, nb))

        self.computation_time_ms = (time.perf_counter()-t0)*1000
        return None


# ---------------------------------------------------------------------------
# 3. Theta*
# ---------------------------------------------------------------------------
class ThetaStar:
    """
    Any-angle path planning. Line-of-sight parent pointer relaxation.
    F_Theta*(s) = G_los(s) + H(s)   Eq.(23) in paper.
    Produces geometrically smoother paths but traverses semantically
    hazardous cells (confirmed in paper Section 4.3, avoidance only 33.2%).
    """
    def __init__(self, grid_size, start, goal):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal  = goal
        self.path:  List[Tuple[int,int]] = []
        self.nodes_expanded = 0
        self.computation_time_ms = 0.0

    def _line_of_sight(self, a, b):
        """Bresenham line-of-sight check."""
        r0,c0 = a; r1,c1 = b
        dr = abs(r1-r0); dc = abs(c1-c0)
        sr = 1 if r1>r0 else -1
        sc = 1 if c1>c0 else -1
        err = dr - dc
        r,c = r0,c0
        while (r,c) != (r1,c1):
            if not (0<=r<self.rows and 0<=c<self.cols):
                return False
            e2 = 2*err
            if e2 > -dc: err -= dc; r += sr
            if e2 <  dr: err += dr; c += sc
        return True

    def plan(self):
        t0 = time.perf_counter()
        open_list  = [(0.0, self.start)]
        came_from  = {self.start: None}
        g_score    = {self.start: 0.0}
        closed_set = set()
        self.nodes_expanded = 0

        while open_list:
            _, cur = heapq.heappop(open_list)
            if cur in closed_set: continue
            closed_set.add(cur)
            self.nodes_expanded += 1

            if cur == self.goal:
                self.path = reconstruct(came_from, self.goal)
                self.computation_time_ms = (time.perf_counter()-t0)*1000
                return self.path

            parent = came_from[cur]
            for nb in neighbours_8(cur, self.rows, self.cols):
                if nb in closed_set: continue
                # Theta*: try line-of-sight from grandparent
                if parent and self._line_of_sight(parent, nb):
                    g_new = g_score[parent] + euclidean(parent, nb)
                    par_new = parent
                else:
                    g_new = g_score[cur] + euclidean(cur, nb)
                    par_new = cur

                if g_new >= g_score.get(nb, float('inf')): continue
                g_score[nb] = g_new
                came_from[nb] = par_new
                f = g_new + euclidean(nb, self.goal)
                heapq.heappush(open_list, (f, nb))

        self.computation_time_ms = (time.perf_counter()-t0)*1000
        return None


# ---------------------------------------------------------------------------
# 4. D* Lite  (simplified — incremental replanning baseline)
# ---------------------------------------------------------------------------
class DStarLite:
    """
    Incremental replanning algorithm. Supports partial real-time adaptation.
    F_D*(s) = G(s) + h(s, s_goal)   Eq.(24) in paper.
    Achieves 44.8% obstacle avoidance — best geometric baseline (Table 8).
    """
    def __init__(self, grid_size, start, goal):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal  = goal
        self.path:  List[Tuple[int,int]] = []
        self.nodes_expanded = 0
        self.computation_time_ms = 0.0

    def plan(self):
        """Initial plan using backward A* from goal."""
        t0 = time.perf_counter()
        open_list  = [(euclidean(self.goal, self.start), self.goal)]
        came_from  = {self.goal: None}
        g_score    = {self.goal: 0.0}
        closed_set = set()
        self.nodes_expanded = 0

        while open_list:
            _, cur = heapq.heappop(open_list)
            if cur in closed_set: continue
            closed_set.add(cur)
            self.nodes_expanded += 1

            if cur == self.start:
                self.path = reconstruct(came_from, self.start)
                self.path.reverse()
                self.computation_time_ms = (time.perf_counter()-t0)*1000
                return self.path

            for nb in neighbours_8(cur, self.rows, self.cols):
                g_new = g_score[cur] + euclidean(cur, nb)
                if g_new >= g_score.get(nb, float('inf')): continue
                g_score[nb] = g_new
                came_from[nb] = cur
                f = g_new + euclidean(nb, self.start)
                heapq.heappush(open_list, (f, nb))

        self.computation_time_ms = (time.perf_counter()-t0)*1000
        return None


# ---------------------------------------------------------------------------
# 5. WC-A*  — Weighted-Cost A* (added in revision, Reviewer R1-3)
# ---------------------------------------------------------------------------
class WCAStar:
    """
    Weighted-Cost A*: semantically-aware but NOT personalised.
    Introduced in revision to address Reviewer 1 Comment R1-3:
    'The experimental design is biased toward the proposed method.'

    F_WC(s) = G(s) + H(s) + λ * sum_k O_k(s)   Eq.(24a) in paper.
    λ = 0.5 (constant, no user preferences, no adaptive scaling, no gradient update).

    Achieves 61.4% obstacle avoidance vs 94.3% for SPPP (Table 8).
    The 32.9 pp gap isolates the contribution of SPC personalisation.
    """
    LAMBDA = 0.5  # constant semantic cost multiplier

    def __init__(self, grid_size, start, goal, obstacle_map: np.ndarray):
        self.rows, self.cols = grid_size
        self.start = start
        self.goal  = goal
        self.obstacle_map = obstacle_map
        self.path:  List[Tuple[int,int]] = []
        self.nodes_expanded = 0
        self.computation_time_ms = 0.0

    def _semantic_cost(self, node):
        """Sum of all obstacle categories at node."""
        row, col = node
        return float(self.obstacle_map[row, col, :].sum())

    def plan(self):
        t0 = time.perf_counter()
        open_list  = [(0.0, self.start)]
        came_from  = {self.start: None}
        g_score    = {self.start: 0.0}
        closed_set = set()
        self.nodes_expanded = 0

        while open_list:
            _, cur = heapq.heappop(open_list)
            if cur in closed_set: continue
            closed_set.add(cur)
            self.nodes_expanded += 1

            if cur == self.goal:
                self.path = reconstruct(came_from, self.goal)
                self.computation_time_ms = (time.perf_counter()-t0)*1000
                return self.path

            for nb in neighbours_8(cur, self.rows, self.cols):
                g_new = euclidean(nb, self.start)
                if g_new >= g_score.get(nb, float('inf')): continue
                g_score[nb] = g_new
                h = euclidean(nb, self.goal)
                sem = self.LAMBDA * self._semantic_cost(nb)
                f = g_new + h + sem
                came_from[nb] = cur
                heapq.heappush(open_list, (f, nb))

        self.computation_time_ms = (time.perf_counter()-t0)*1000
        return None
