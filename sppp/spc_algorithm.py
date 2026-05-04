"""
Semantic Personalised Cost (SPC) Algorithm
===========================================
Core implementation of the SPC algorithm as described in:

  Saranya C and Janaki G (2026). A Heuristic Intelligent Search with
  Adaptive Personalised Cost Optimisation for Real-Time Obstacle-Aware
  Path Planning in Autonomous Ground Vehicles.
  Applied Sciences (MDPI). Manuscript ID: applsci-4280957.

Equations referenced throughout correspond to the manuscript numbering.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Preference–Obstacle Interaction Matrix  (Table 2 in paper)
# Rows: O1..O8  (obstacle categories)
# Cols: I1..I8  (preference dimensions)
# ---------------------------------------------------------------------------
#               I1     I2     I3     I4     I5     I6     I7     I8
#            Speed   Time  Safety Batt.  Emerg. Comft.  Cost  Scenic
INTERACTION_MATRIX = [
    [True,  True,  False, True,  True,  False, False, False],  # O1 Traffic
    [True,  False, True,  False, False, True,  True,  False],  # O2 Road type
    [False, True,  True,  False, False, True,  False, True ],  # O3 Weather
    [False, False, True,  False, False, False, False, True ],  # O4 Pollution
    [False, True,  True,  True,  True,  False, False, False],  # O5 Accessibility
    [False, False, True,  False, True,  True,  False, False],  # O6 Dead zones
    [False, False, False, False, True,  True,  False, False],  # O7 Events
    [False, False, False, False, True,  False, False, False],  # O8 Construction
]

# Preference dimension names (Table 1)
PREFERENCE_NAMES = [
    "Speed Level",       # I1
    "Time",              # I2
    "Safety",            # I3
    "Battery Efficiency",# I4
    "Emergency Response",# I5
    "Comfort",           # I6
    "Cost",              # I7
    "Scenic Beauty",     # I8
]

# Obstacle category names
OBSTACLE_NAMES = [
    "Traffic Congestion",       # O1
    "Road Type / Surface",      # O2
    "Weather Severity",         # O3
    "Pollution / Restricted",   # O4
    "Accessibility",            # O5
    "Dead Zones",               # O6
    "Live Events",              # O7
    "Construction Zones",       # O8
]

# Hyperparameters (Table 6 / Section 3.3.4)
BETA    = 2.0   # scaling magnitude
GAMMA   = 0.5   # scaling sensitivity
D_TH    = 3.0   # activation threshold distance (grid units)
ETA     = 0.01  # learning rate (proposed value from ablation, Table 3)


# ---------------------------------------------------------------------------
# Core SPC functions
# ---------------------------------------------------------------------------

def compute_g(node: Tuple[int,int], start: Tuple[int,int]) -> float:
    """
    Actual traversal cost G(s): Euclidean distance from start to node.
    Eq. (1): G(s) = sqrt((x - x_s)^2 + (y - y_s)^2)
    """
    return math.sqrt((node[0] - start[0])**2 + (node[1] - start[1])**2)


def compute_h(node: Tuple[int,int], goal: Tuple[int,int]) -> float:
    """
    Heuristic cost H(s): Euclidean distance from node to goal.
    Eq. (3): H(s) = sqrt((x_g - x)^2 + (y_g - y)^2)
    """
    return math.sqrt((goal[0] - node[0])**2 + (goal[1] - node[1])**2)


def compute_p(
    node: Tuple[int,int],
    obstacle_map: np.ndarray,
    weights: np.ndarray,
    active_prefs: List[bool],
) -> float:
    """
    Personalised cost P(s): weighted sum of obstacle severities.
    Eq. (5): P(s) = sum_i  w_i * O_i(s)

    Parameters
    ----------
    node          : (row, col) grid coordinates
    obstacle_map  : shape (8, 8, 8) — [row, col, obstacle_category]
    weights       : shape (8,)      — preference weights w_i
    active_prefs  : shape (8,)      — binary YES/NO activation per preference

    Returns
    -------
    float : personalised cost at this node
    """
    row, col = node
    p_cost = 0.0
    for i, active in enumerate(active_prefs):
        if not active:
            continue
        # Aggregate obstacle impact for preference i
        o_i = 0.0
        for j in range(8):  # obstacle categories O1-O8
            if INTERACTION_MATRIX[j][i]:
                o_i += obstacle_map[row, col, j]
        p_cost += weights[i] * o_i
    return p_cost


def compute_alpha(node: Tuple[int,int], goal: Tuple[int,int],
                  beta: float = BETA, gamma: float = GAMMA,
                  d_th: float = D_TH) -> float:
    """
    Adaptive scaling factor alpha(s).
    Eq. (6): alpha(s) = 1 + beta / (1 + exp(+gamma * (D - D_th)))

    Amplifies obstacle penalties when vehicle is within D_th of goal.
    When D > D_th: alpha ≈ 1 (no amplification).
    When D ≤ D_th: alpha rises toward 1 + beta = 3.0.
    """
    D = compute_h(node, goal)  # remaining distance to goal
    alpha = 1.0 + beta / (1.0 + math.exp(gamma * (D - d_th)))
    return alpha


def scale_personalised_cost(p: float, alpha: float) -> float:
    """
    Apply adaptive scaling to personalised cost.
    Eq. (7): P(s) <- alpha(s) * P(s)
    """
    return alpha * p


def update_weights(
    weights: np.ndarray,
    path: List[Tuple[int,int]],
    obstacle_map: np.ndarray,
    active_prefs: List[bool],
    goal: Tuple[int,int],
    eta: float = ETA,
) -> np.ndarray:
    """
    Gradient-based preference weight update.
    Eq. (10): w_i^{t+1} = w_i^t + eta * sum_{s in path} alpha(s) * O_i(s)
    Eq. (11): normalise weights to sum to 1 (numerical stability)

    Parameters
    ----------
    weights      : current weight vector (8,)
    path         : list of (row,col) nodes in current path
    obstacle_map : (8,8,8) obstacle severity array
    active_prefs : binary activation vector (8,)
    goal         : goal node coordinates
    eta          : learning rate

    Returns
    -------
    np.ndarray : updated and normalised weight vector (8,)
    """
    new_weights = weights.copy()
    for i, active in enumerate(active_prefs):
        if not active:
            continue
        gradient = 0.0
        for node in path:
            row, col = node
            alpha = compute_alpha(node, goal)
            # Gradient: dL/dw_i = sum_{s in path} alpha(s) * O_i(s)  Eq.(9)
            o_i = sum(
                obstacle_map[row, col, j]
                for j in range(8) if INTERACTION_MATRIX[j][i]
            )
            gradient += alpha * o_i
        new_weights[i] += eta * gradient

    # Normalisation Eq. (11)
    active_sum = sum(new_weights[i] for i in range(8) if active_prefs[i])
    if active_sum > 0:
        for i in range(8):
            if active_prefs[i]:
                new_weights[i] /= active_sum
    return new_weights


def initialise_weights(active_prefs: List[bool]) -> np.ndarray:
    """
    Uniform weight initialisation for active preferences.
    Eq. (12): w_i^0 = 1 / N_active
    """
    n_active = sum(active_prefs)
    weights = np.zeros(8)
    if n_active > 0:
        for i in range(8):
            if active_prefs[i]:
                weights[i] = 1.0 / n_active
    return weights


def total_cost(g: float, h: float, p: float) -> float:
    """
    Integrated total cost function.
    Eq. (15): F(s) = G(s) + H(s) + P(s)
    """
    return g + h + p
