"""
Environment and Obstacle Map Utilities
=======================================
Grid environment generation, obstacle map creation, and
obstacle avoidance rate computation for SPPP experiments.

Author      : Saranya C, Janaki G
Institution : SRM Institute of Science and Technology, Kattankulathur
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional

from sppp.spc_algorithm import OBSTACLE_KEYS, INTERACTION_MATRIX, PREFERENCE_KEYS


# ---------------------------------------------------------------------------
# Obstacle Map Generation
# ---------------------------------------------------------------------------

def generate_obstacle_map(
        grid_size: Tuple[int, int] = (8, 8),
        seed: Optional[int] = None,
        environment: str = 'urban') -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Generate randomised obstacle severity map for simulation experiments.

    Severity scores are randomised within realistic bounds per environment
    type, consistent with Phase 1 experimental setup (Section 4.1).

    Parameters
    ----------
    grid_size : tuple of int
        Grid dimensions (rows, cols). Default: (8, 8).
    seed : int, optional
        Random seed for reproducibility.
    environment : str
        Environment type: 'urban', 'highway', or 'rural'.
        Controls obstacle severity distribution bounds.

    Returns
    -------
    dict
        {(row, col): {O1: val, ..., O8: val}} for all grid cells.

    Notes
    -----
    Environment severity bounds (Table 4.1 in paper):
    - urban  : high obstacle density, frequent events and construction
    - highway: moderate traffic severity, low weather variation
    - rural  : low obstacle density, high weather variability
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Severity bounds per environment per obstacle category
    bounds = {
        'urban': {
            'O1': (1.0, 5.0),  # High traffic
            'O2': (1.0, 4.0),  # Variable road surface
            'O3': (0.0, 3.0),  # Moderate weather
            'O4': (0.0, 3.0),  # Moderate pollution
            'O5': (0.0, 3.0),  # Accessibility issues
            'O6': (0.0, 2.0),  # Some dead zones
            'O7': (1.0, 4.0),  # Frequent events
            'O8': (1.0, 4.0),  # Frequent construction
        },
        'highway': {
            'O1': (1.0, 4.0),  # Moderate traffic
            'O2': (0.0, 2.0),  # Good road surface
            'O3': (0.0, 2.0),  # Low weather variation
            'O4': (0.0, 2.0),  # Low pollution
            'O5': (0.0, 1.0),  # Low accessibility issues
            'O6': (0.0, 1.0),  # Few dead zones
            'O7': (0.0, 2.0),  # Rare events
            'O8': (0.0, 2.0),  # Rare construction
        },
        'rural': {
            'O1': (0.0, 2.0),  # Low traffic
            'O2': (1.0, 4.0),  # Variable surface
            'O3': (1.0, 5.0),  # High weather variability
            'O4': (0.0, 1.0),  # Low pollution
            'O5': (1.0, 3.0),  # Accessibility issues
            'O6': (1.0, 4.0),  # Dead zones common
            'O7': (0.0, 1.0),  # Rare events
            'O8': (0.0, 2.0),  # Some construction
        }
    }

    env_bounds = bounds.get(environment, bounds['urban'])
    rows, cols = grid_size
    obstacle_map = {}

    for r in range(rows):
        for c in range(cols):
            scores = {}
            for obs_key in OBSTACLE_KEYS:
                lo, hi = env_bounds[obs_key]
                scores[obs_key] = round(random.uniform(lo, hi), 2)
            obstacle_map[(r, c)] = scores

    return obstacle_map


def get_aggregate_severity(
        obstacle_map: Dict,
        node: Tuple[int, int]) -> float:
    """
    Compute aggregate obstacle severity at a node (sum of O1-O8).

    Parameters
    ----------
    obstacle_map : dict
        Per-node obstacle scores.
    node : tuple of int
        Node coordinates.

    Returns
    -------
    float
        Sum of all obstacle severity scores (scale 0-40).
    """
    scores = obstacle_map.get(node, {k: 0.0 for k in OBSTACLE_KEYS})
    return sum(scores.get(k, 0.0) for k in OBSTACLE_KEYS)


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

def compute_obstacle_avoidance_rate(
        path: List[Tuple[int, int]],
        obstacle_map: Dict,
        active_prefs: List[str],
        weights: Dict[str, float],
        threshold: float = 3.5) -> float:
    """
    Compute obstacle avoidance rate for a given path.

    A node is considered "avoided" if its weighted semantic obstacle
    burden falls below the avoidance threshold.

    Avoidance rate = (nodes with P(s) < threshold) / total path nodes × 100

    Parameters
    ----------
    path : list of tuple
        Planned path nodes.
    obstacle_map : dict
        Per-node obstacle scores.
    active_prefs : list of str
        Active preference keys.
    weights : dict
        Preference weights.
    threshold : float
        P(s) threshold below which a node is considered avoided.
        Default: 3.5 (consistent with Section 4.5 in paper).

    Returns
    -------
    float
        Obstacle avoidance rate as percentage (0-100).
    """
    if not path:
        return 0.0

    from sppp.spc_algorithm import (
        compute_preference_obstacle_values, validate_obstacle_scores
    )

    avoided = 0
    for node in path:
        obs = validate_obstacle_scores(
            obstacle_map.get(node, {k: 0.0 for k in OBSTACLE_KEYS})
        )
        pref_vals = compute_preference_obstacle_values(obs, active_prefs)
        P_raw = sum(
            weights.get(p, 0.0) * pref_vals.get(p, 0.0)
            for p in active_prefs
        )
        if P_raw < threshold:
            avoided += 1

    return round(avoided / len(path) * 100, 1)


def compute_mean_P(
        path: List[Tuple[int, int]],
        obstacle_map: Dict,
        active_prefs: List[str],
        weights: Dict[str, float]) -> float:
    """
    Compute mean personalised cost P(s) per node along a path.

    Parameters
    ----------
    path : list of tuple
        Planned path nodes.
    obstacle_map : dict
        Per-node obstacle scores.
    active_prefs : list of str
        Active preference keys.
    weights : dict
        Preference weights.

    Returns
    -------
    float
        Mean P(s) per node.
    """
    if not path:
        return 0.0

    from sppp.spc_algorithm import (
        compute_P, validate_obstacle_scores, compute_H
    )

    total_P = 0.0
    goal = path[-1]
    for node in path:
        obs = validate_obstacle_scores(
            obstacle_map.get(node, {k: 0.0 for k in OBSTACLE_KEYS})
        )
        H = compute_H(node, goal)
        total_P += compute_P(obs, weights, active_prefs, H)

    return round(total_P / len(path), 2)


# ---------------------------------------------------------------------------
# Standard Paper Scenarios
# ---------------------------------------------------------------------------

PREFERENCE_SCENARIOS = {
    'EDT-A*': {
        'active_prefs': [],
        'description': 'No preference (baseline)'
    },
    'S1': {
        'active_prefs': ['I1', 'I4'],
        'description': 'Speed + Battery priority'
    },
    'S2': {
        'active_prefs': ['I2', 'I3'],
        'description': 'Safety + Time priority'
    },
    'S3': {
        'active_prefs': ['I2', 'I3', 'I5'],
        'description': 'Full Emergency (Time + Safety + Emergency)'
    },
    'S4': {
        'active_prefs': ['I6', 'I7'],
        'description': 'Comfort + Cost priority'
    },
}
