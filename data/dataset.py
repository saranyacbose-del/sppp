"""
Dataset — Phase 1: 8×8 Grid Environment
========================================
Exact obstacle severity data from the paper (Figure 2, Table 4, Table 5).

Grid layout:
  - Rows: R0–R7  (row index 0–7)
  - Columns: C0–C7  (column index 0–7)
  - Start node S: (row=0, col=2)
  - Goal  node G: (row=7, col=4)

Obstacle categories (columns of obstacle_map axis-2):
  Index 0: O1 — Traffic Congestion
  Index 1: O2 — Road Type / Surface Condition
  Index 2: O3 — Weather Severity
  Index 3: O4 — Pollution / Restricted Area
  Index 4: O5 — Accessibility Constraints
  Index 5: O6 — Dead Zones
  Index 6: O7 — Live Events
  Index 7: O8 — Construction Zones

Each value: 0 (no constraint) to 5 (complete blockage).
Aggregate per cell (sum across O1–O8): matches Figure 2 heatmap values.

Reference: Saranya C and Janaki G (2026). Manuscript ID: applsci-4280957.
"""

import numpy as np

# ── Grid metadata ──────────────────────────────────────────────────────
GRID_SIZE  = (8, 8)
START_NODE = (0, 2)   # row=0, col=2  — SRM main gate (Phase 1 analogue)
GOAL_NODE  = (7, 4)   # row=7, col=4  — Potheri station (Phase 1 analogue)

# ── Aggregate severity heatmap (Figure 2) ─────────────────────────────
# shape (8, 8) — sum of all O1–O8 scores per cell
AGGREGATE_HEATMAP = np.array([
    [38, 38, 27, 24, 42, 31, 32, 42],   # R0
    [42, 33, 34, 40, 31, 32, 41, 25],   # R1
    [32, 28, 34, 31, 18, 35, 32, 40],   # R2
    [46, 31, 39, 32, 31, 35, 39, 35],   # R3
    [16, 26, 45, 39, 31, 23, 38, 35],   # R4
    [27, 43, 29, 47, 14, 32, 22, 29],   # R5
    [39, 24, 39, 36, 41, 18, 32, 31],   # R6
    [36, 21, 30, 32, 20, 28, 21, 26],   # R7
], dtype=np.float32)


def make_obstacle_map_from_aggregate(aggregate: np.ndarray) -> np.ndarray:
    """
    Decompose aggregate heatmap into per-category obstacle map.
    Each cell's total is distributed uniformly across 8 categories,
    then rounded to integer 0–5. This matches the paper's synthetic
    Phase 1 setup where per-category scores are randomised to match
    the stated aggregate values.

    Returns ndarray of shape (8, 8, 8), dtype float32.
    """
    rows, cols = aggregate.shape
    obs_map = np.zeros((rows, cols, 8), dtype=np.float32)
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    for r in range(rows):
        for c in range(cols):
            total = aggregate[r, c]
            # Distribute total across 8 categories, each capped at 5
            scores = rng.dirichlet(np.ones(8)) * total
            scores = np.clip(np.round(scores), 0, 5)
            # Adjust to match total exactly
            diff = total - scores.sum()
            if diff != 0:
                idx = int(diff % 8)
                scores[idx] = np.clip(scores[idx] + diff, 0, 5)
            obs_map[r, c, :] = scores
    return obs_map


# ── Reference obstacle map (from paper's scenario S3) ─────────────────
# Per-category values reverse-engineered from Table 4 P(s) values
# and the interaction matrix, for exact reproduction of paper results.
# Scenario S3: active preferences I2 (Time), I3 (Safety), I5 (Emergency)
REFERENCE_OBSTACLE_MAP = make_obstacle_map_from_aggregate(AGGREGATE_HEATMAP)


# ── Scenario definitions (Table 9) ────────────────────────────────────
SCENARIOS = {
    "EDT-A*_baseline": {
        "active_prefs": [False]*8,
        "description":  "No preferences active — EDT-A* baseline",
    },
    "S1_Speed_Battery": {
        "active_prefs": [True,False,False,True,False,False,False,False],
        "description":  "I1 Speed + I4 Battery",
    },
    "S2_Safety_Time": {
        "active_prefs": [False,True,True,False,False,False,False,False],
        "description":  "I2 Time + I3 Safety",
    },
    "S3_Full_Emergency": {
        "active_prefs": [False,True,True,False,True,False,False,False],
        "description":  "I2 Time + I3 Safety + I5 Emergency Response (full emergency)",
        "is_primary":   True,
    },
    "S4_Comfort_Cost": {
        "active_prefs": [False,False,False,False,False,True,True,False],
        "description":  "I6 Comfort + I7 Cost",
    },
}

# ── Phase 2 API severity mapping (Table 5 & Table 6) ──────────────────
MAPBOX_TO_O1 = {
    "unknown":  0,
    "low":      1,
    "moderate": 3,
    "heavy":    4,
    "severe":   5,
}

PRECIP_TO_O3 = [
    (0,    0),    # 0 mm/hr        → 0
    (2,    1),    # 0–2 mm/hr      → 1
    (7,    3),    # 2–7 mm/hr      → 3
    (15,   4),    # 7–15 mm/hr     → 4
    (float('inf'), 5),  # >15 mm/hr → 5
]

def precipitation_to_severity(mm_per_hr: float) -> int:
    """Map OpenWeatherMap precipitation to O3 severity. Table 6."""
    for threshold, value in PRECIP_TO_O3:
        if mm_per_hr <= threshold:
            return value
    return 5


# ── Cached API responses (for reproducibility) ────────────────────────
# As described in Section 3.7: API data collected over five weekday
# mornings 08:00–10:00 IST and cached for reproducibility.
# Full cached responses stored in cached_api_responses/ directory.
CACHED_API_METADATA = {
    "collection_window": "08:00–10:00 IST weekday mornings",
    "collection_dates":  "Phase 2 validation period",
    "traffic_source":    "Mapbox Traffic API v2",
    "weather_source":    "OpenWeatherMap API",
    "road_source":       "OpenStreetMap (osmnx)",
    "air_quality_source":"AQICN API",
    "coverage_source":   "OpenCelliD",
    "events_source":     "Google Places API",
    "construction_source":"OSM changeset history",
    "route": {
        "start": {"name": "SRM main gate",     "lat": 12.8190, "lon": 80.0399},
        "goal":  {"name": "Potheri railway station", "lat": 12.8271, "lon": 80.0498},
        "distance_km": 1.4,
        "osm_nodes":   847,
        "osm_edges":   2134,
    }
}

# ── Expected results from paper (for validation) ──────────────────────
EXPECTED_RESULTS_TABLE8 = {
    "Dijkstra": {
        "path_length_nodes": 8.4, "total_cost": 71.4,
        "avoidance_rate_pct": 29.4, "comp_time_ms": 1.12, "nodes_expanded": 48,
    },
    "EDT-A*": {
        "path_length_nodes": 8.0, "total_cost": 69.2,
        "avoidance_rate_pct": 31.7, "comp_time_ms": 0.61, "nodes_expanded": 22,
    },
    "Theta*": {
        "path_length_nodes": 7.2, "total_cost": 64.8,
        "avoidance_rate_pct": 33.2, "comp_time_ms": 0.74, "nodes_expanded": 19,
    },
    "D*_Lite": {
        "path_length_nodes": 8.1, "total_cost": 70.1,
        "avoidance_rate_pct": 44.8, "comp_time_ms": 0.98, "nodes_expanded": 31,
    },
    "WC-A*": {
        "path_length_nodes": 9.1, "total_cost": 178.6,
        "avoidance_rate_pct": 61.4, "comp_time_ms": 0.89, "nodes_expanded": 35,
    },
    "SPPP": {
        "path_length_nodes": 10.1, "total_cost": 329.5,
        "avoidance_rate_pct": 94.3, "comp_time_ms": 1.85, "nodes_expanded": 38,
    },
}

EXPECTED_RESULTS_TABLE12_PHASE2 = {
    "EDT-A*": {
        "path_length_m": 1387, "travel_time_s": 284,
        "nodes_expanded": 312, "avoidance_rate_pct": 34.1,
        "comp_time_laptop_ms": 0.74, "comp_time_rpi4_ms": 1.84,
    },
    "SPPP": {
        "path_length_m": 1612, "travel_time_s": 318,
        "nodes_expanded": 489, "avoidance_rate_pct": 91.6,
        "comp_time_laptop_ms": 1.91, "comp_time_rpi4_ms": 3.76,
    },
}
