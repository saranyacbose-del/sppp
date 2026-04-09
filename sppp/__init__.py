"""
SPPP: Semantic-Weighted A* Framework with Dynamic User Preference Optimisation
==============================================================================
for Real-Time Personalised Path Planning in Autonomous Ground Vehicles

Authors     : Saranya C, Janaki G
Institution : SRM Institute of Science and Technology, Kattankulathur, India
Contact     : saranyaresearch22@gmail.com

Modules
-------
spc_algorithm  : Core SPC cost function components (G, H, P, F, weight update)
sppp_search    : SPPP A* search engine with real-time validation module
baselines      : EDT-A*, Dijkstra, Theta*, D* Lite baseline implementations
environment    : Obstacle map generation and performance metric utilities
"""

from sppp.spc_algorithm import (
    compute_G, compute_H, compute_P, compute_F,
    update_weights, initialise_weights,
    INTERACTION_MATRIX, OBSTACLE_KEYS, PREFERENCE_KEYS,
    BETA, GAMMA, D_TH, ETA
)
from sppp.sppp_search import SPPPSearch, SPPPResult
from sppp.baselines import EDTAStar, Dijkstra, ThetaStar, DStarLite
from sppp.environment import (
    generate_obstacle_map, compute_obstacle_avoidance_rate,
    compute_mean_P, PREFERENCE_SCENARIOS
)

__version__ = "1.0.0"
__all__ = [
    "SPPPSearch", "SPPPResult",
    "EDTAStar", "Dijkstra", "ThetaStar", "DStarLite",
    "compute_G", "compute_H", "compute_P", "compute_F",
    "update_weights", "initialise_weights",
    "generate_obstacle_map", "compute_obstacle_avoidance_rate",
    "compute_mean_P", "PREFERENCE_SCENARIOS",
    "INTERACTION_MATRIX", "OBSTACLE_KEYS", "PREFERENCE_KEYS",
]
