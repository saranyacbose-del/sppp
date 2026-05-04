# SPPP — Semantic Personalised Path Planning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official implementation of the **Semantic Personalised Path Planning (SPPP)** system from:

> **Saranya C and Janaki G (2026)**  
> *A Heuristic Intelligent Search with Adaptive Personalised Cost Optimisation for Real-Time Obstacle-Aware Path Planning in Autonomous Ground Vehicles*  
> Applied Sciences (MDPI). Manuscript ID: applsci-4280957.

---

## Overview

The SPPP system augments the A\* search framework with a dynamically computed **Semantic Personalised Cost (SPC)** term, integrating:

- **8 real-time semantic obstacle categories** (O₁–O₈): traffic, road surface, weather, pollution, accessibility, dead zones, events, construction
- **8 user-defined preference dimensions** (I₁–I₈): speed, time, safety, battery, emergency response, comfort, cost, scenic beauty
- **Adaptive scaling** α(s) that amplifies obstacle penalties near the goal
- **Gradient-based weight evolution** that refines preferences over successive route segments

### Key Results (Scenario S3 — Full Emergency)

| Metric | EDT-A\* (baseline) | **SPPP (proposed)** |
|---|---|---|
| Obstacle avoidance rate | 31.7% | **94.3%** |
| Computation time (laptop) | 0.61 ms | **1.91 ms** |
| Computation time (RPi 4) | 1.84 ms | **3.76 ms** |
| Trajectory feasibility | 91.2% | **94.6%** |
| Mean obstacle severity | 2.81 | **1.24 (−55.9%)** |
| Cohen's d (vs EDT-A\*) | — | **≈ 18.9** (p < 0.001) |

---

## Repository Structure

```
sppp/
├── sppp/
│   ├── spc_algorithm.py      # Core SPC equations (Eq. 1–15)
│   └── planner.py            # SPPP A* planner + validation module
├── baselines/
│   └── baselines.py          # EDT-A*, Dijkstra, Theta*, D* Lite, WC-A*
├── data/
│   └── dataset.py            # Grid data, obstacle maps, scenario definitions
├── experiments/
│   ├── run_phase1.py         # 500-run Phase 1 simulation (Tables 7–11)
│   └── run_phase2_osm.py     # Real-world OSM validation (Table 12)
├── tests/
│   └── test_sppp.py          # Unit tests verifying all paper equations
├── notebooks/
│   └── sppp_demo.ipynb       # Interactive demonstration notebook
├── cached_api_responses/     # Cached Mapbox/OWM API data for reproducibility
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/saranyacbose-del/sppp.git
cd sppp
pip install -r requirements.txt
```

For Phase 2 (real-world OSM validation):
```bash
pip install osmnx networkx
```

---

## Quick Start

### Single path planning run

```python
import numpy as np
from sppp.planner import SPPPPlanner
from data.dataset import GRID_SIZE, START_NODE, GOAL_NODE, REFERENCE_OBSTACLE_MAP, SCENARIOS

# Scenario S3: I2 (Time) + I3 (Safety) + I5 (Emergency Response)
active_prefs = SCENARIOS["S3_Full_Emergency"]["active_prefs"]

planner = SPPPPlanner(
    grid_size    = GRID_SIZE,        # (8, 8)
    start        = START_NODE,       # (0, 2)
    goal         = GOAL_NODE,        # (7, 4)
    active_prefs = active_prefs,
    obstacle_map = REFERENCE_OBSTACLE_MAP.copy(),
    eta   = 0.01,   # learning rate (Table 3)
    beta  = 2.0,    # adaptive scaling magnitude
    gamma = 0.5,    # scaling sensitivity
    d_th  = 3.0,    # activation threshold distance
)

path = planner.plan()

print(f"Path:              {path}")
print(f"Path length:       {len(path)} nodes")
print(f"Computation time:  {planner.computation_time_ms:.3f} ms")
print(f"Validation passed: {planner.validation_passed}")
print(f"MAHC:              {planner.compute_mahc():.1f}°")
print(f"Feasibility rate:  {planner.compute_feasibility()*100:.1f}%")
print(f"Mean obstacle sev: {planner.compute_mean_obstacle_severity():.2f}")
```

### Run full Phase 1 experiment (500 runs)

```bash
# Scenario S3 — Full Emergency (primary scenario in paper)
python experiments/run_phase1.py --scenario S3_Full_Emergency --runs 500

# All scenarios
python experiments/run_phase1.py --scenario all --runs 500

# Learning rate ablation (Table 3)
python experiments/run_phase1.py --ablation --runs 500

# Real-time replanning test (Table 11)
python experiments/run_phase1.py --replan --runs 500
```

### Run Phase 2 OSM validation

```bash
python experiments/run_phase2_osm.py
```

### Run unit tests

```bash
python tests/test_sppp.py
# or
python -m pytest tests/ -v
```

---

## Algorithm Details

### Cost Function (Eq. 15)

```
F(s) = G(s) + H(s) + P(s)
```

| Component | Formula | Description |
|---|---|---|
| G(s) | √((x−x_s)² + (y−y_s)²) | Euclidean distance from start (Eq. 1) |
| H(s) | √((x_g−x)² + (y_g−y)²) | Euclidean heuristic to goal (Eq. 3) |
| P(s) | Σᵢ wᵢ · Oᵢ(s) | Personalised semantic cost (Eq. 5) |

### Adaptive Scaling (Eq. 6–7)

```
α(s) = 1 + β / (1 + exp(+γ(D − D_th)))
P(s) ← α(s) · P(s)
```

Parameters: β=2.0, γ=0.5, D_th=3.0 grid units.

### Weight Evolution (Eq. 10–11)

```
wᵢᵗ⁺¹ = wᵢᵗ + η · Σ_{s∈path} α(s) · Oᵢ(s)
wᵢᵗ⁺¹ = wᵢᵗ⁺¹ / Σⱼ wⱼᵗ⁺¹   (normalisation)
```

Learning rate η = 0.01 selected via ablation (Table 3).

### Preference–Obstacle Interaction Matrix (Table 2)

```
         I1    I2    I3    I4    I5    I6    I7    I8
         Speed Time Safety Batt Emerg Comft  Cost Scenic
O1 Traffic  ✓     ✓     -     ✓     ✓     -     -     -
O2 Road     ✓     -     ✓     -     -     ✓     ✓     -
O3 Weather  -     ✓     ✓     -     -     ✓     -     ✓
O4 Pollut.  -     -     ✓     -     -     -     -     ✓
O5 Access.  -     ✓     ✓     ✓     ✓     -     -     -
O6 Dead zn  -     -     ✓     -     ✓     ✓     -     -
O7 Events   -     -     -     -     ✓     ✓     -     -
O8 Constr.  -     -     -     -     ✓     -     -     -
```

---

## Scenarios

| Scenario | Active Preferences | Avoidance Rate |
|---|---|---|
| EDT-A\* baseline | None | 31.7% |
| S1 — Speed + Battery | I₁, I₄ | 48.2% |
| S2 — Safety + Time | I₂, I₃ | 79.6% |
| **S3 — Full Emergency** | **I₂, I₃, I₅** | **94.3%** |
| S4 — Comfort + Cost | I₆, I₇ | 67.4% |

---

## Baseline Comparison (Table 8, Scenario S3, 500 runs)

| Algorithm | Avoidance | Path len | Comp. time | Semantic |
|---|---|---|---|---|
| Dijkstra | 29.4% | 8.4 | 1.12 ms | No |
| EDT-A\* | 31.7% | 8.0 | 0.61 ms | No |
| Theta\* | 33.2% | 7.2 | 0.74 ms | No |
| D\* Lite | 44.8% | 8.1 | 0.98 ms | No |
| WC-A\* | 61.4% | 9.1 | 0.89 ms | Partial |
| **SPPP** | **94.3%** | 10.1 | 1.85 ms | **Full** |

WC-A\* (Weighted-Cost A\*) was added in the revision (Reviewer R1-3) to isolate
the contribution of SPC personalisation. The 32.9 pp gap (61.4%→94.3%) is
attributable exclusively to user preference weighting, adaptive scaling, and
gradient-based weight evolution.

---

## Data and Reproducibility

All API responses used in Phase 2 are cached in `cached_api_responses/` to ensure reproducibility:
- **Mapbox Traffic API v2** congestion annotations → O₁ (Table 5)
- **OpenWeatherMap API** precipitation data → O₃ (Table 6)
- **OpenStreetMap** road attributes → O₂, O₅, O₈

Data collected over five weekday mornings, 08:00–10:00 IST.

---

## Statistical Validation

Results are statistically significant at p < 0.001 (Wilcoxon signed-rank test, Table 10):

| Metric | EDT-A\* | SPPP | W statistic | p-value |
|---|---|---|---|---|
| Obstacle avoidance | 31.7 ± 4.2% | 94.3 ± 2.1% | 124750 | < 0.001 |
| Path length | 8.0 ± 0.6 | 10.1 ± 0.8 | 118340 | < 0.001 |
| Computation time | 0.61 ± 0.08 ms | 1.85 ± 0.21 ms | 124980 | < 0.001 |

Cohen's d for obstacle avoidance rate: **≈ 18.9** (very large effect).

---

## Citation

```bibtex
@article{saranya2026sppp,
  title   = {A Heuristic Intelligent Search with Adaptive Personalised Cost
             Optimisation for Real-Time Obstacle-Aware Path Planning in
             Autonomous Ground Vehicles},
  author  = {Saranya C and Janaki G},
  journal = {Applied Sciences},
  publisher = {MDPI},
  year    = {2026},
  note    = {Manuscript ID: applsci-4280957}
}
```

---

## Authors

**Saranya C** · [saranyaresearch22@gmail.com](mailto:saranyaresearch22@gmail.com)  
ORCID: [0009-0004-4335-9133](https://orcid.org/0009-0004-4335-9133)

**Janaki G** · [janakig@srmist.edu.in](mailto:janakig@srmist.edu.in)  
ORCID: [0000-0002-0748-4904](https://orcid.org/0000-0002-0748-4904)

Department of Electrical and Electronics Engineering  
SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India

---

## License

© 2026 by the authors. Submitted for possible open access publication under the
terms and conditions of the [Creative Commons Attribution (CC BY) 4.0](https://creativecommons.org/licenses/by/4.0/) license.
