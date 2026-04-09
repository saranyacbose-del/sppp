# Semantic-Weighted A* Framework with Dynamic User Preference Optimisation
## for Real-Time Personalised Path Planning in Autonomous Ground Vehicles

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](tests/)

**Authors:** Saranya C, Janaki G  
**Institution:** Department of Electrical and Electronics Engineering, SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India  
**Contact:** saranyaresearch22@gmail.com

---

## Overview

This repository provides the complete Python implementation of the **Semantic Personalised Path Planning (SPPP)** system described in:

> Saranya C and Janaki G (2025). *"Semantic-Weighted A\* Framework with Dynamic User Preference Optimisation for Real-Time Personalised Path Planning in Autonomous Ground Vehicles."*

The SPPP system extends the classical A\* algorithm with a **Semantic Personalised Cost (SPC)** term:

$$F(s) = G(s) + H(s) + P(s)$$

where:
- **G(s)** — Euclidean cost from start to current node  
- **H(s)** — Euclidean heuristic to goal  
- **P(s)** — Personalised semantic cost: $\sum w_i \cdot O_i(s)$ (adaptively scaled)

---

## Repository Structure

```
sppp_github/
│
├── sppp/                          # Core package
│   ├── __init__.py                # Package exports
│   ├── spc_algorithm.py           # SPC cost components (G, H, P, F, weights)
│   ├── sppp_search.py             # SPPP A* search engine + replanning
│   ├── baselines.py               # EDT-A*, Dijkstra, Theta*, D* Lite
│   └── environment.py             # Obstacle map generation + metrics
│
├── experiments/                   # Reproduce all paper results
│   ├── run_experiments.py         # Tables 5, 6, 8 (500-run evaluation)
│   └── weight_evolution.py        # Table 3a + weight evolution figure
│
├── results/                       # CSV output from experiments
├── figures/                       # Generated figures
├── tests/                         # Unit tests
│   └── test_sppp.py               # 30+ tests covering all components
│
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/sppp-agv.git
cd sppp-agv
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
pytest>=7.0.0
```

---

## Quick Start

```python
from sppp import SPPPSearch, generate_obstacle_map

# Generate obstacle map (or load from real API)
obs_map = generate_obstacle_map(
    grid_size=(8, 8),
    seed=42,
    environment='urban'   # 'urban', 'highway', 'rural'
)

# Configure planner — Scenario S3 (Full Emergency)
planner = SPPPSearch(
    grid_size=(8, 8),
    start=(0, 2),
    goal=(7, 4),
    obstacle_map=obs_map,
    active_prefs=['I2', 'I3', 'I5']   # Time + Safety + Emergency
)

# Run SPPP search
result = planner.search()

# Print full cost breakdown
print(result.summary())
```

**Example output:**
```
============================================================
SPPP Search Result Summary
============================================================
Success          : True
Path length      : 10 nodes
Total F(s) cost  : 329.49 units
Nodes expanded   : 38
Computation time : 1.85 ms
Path             : [(0,2),(0,3),(1,4),(2,4),(3,4),(4,5),(5,4),(6,5),(7,4)]
------------------------------------------------------------
Node          G(s)     H(s)     P(s)     F(s)   P/F %
------------------------------------------------------------
(0, 2)        0.00     7.28    27.00    34.28   78.8%
(0, 3)        1.00     7.07    24.00    32.07   74.8%
...
(5, 4)        5.39     2.00    14.00    21.39   65.5%  ← Min F(s)
(7, 4)        7.28     0.00    20.00    27.28   73.3%
============================================================
Mean P(s) per node : 25.47
Mean F(s) per node : 32.95
Mean P(s)/F(s)     : 77.4%
============================================================
```

---

## Reproducing Paper Results

### Table 5 — Comparative 500-run evaluation (Scenario S3)

```bash
python experiments/run_experiments.py --runs 500
```

Expected output (Table 5):

| Metric | Dijkstra | EDT-A* | Theta* | D* Lite | **SPPP** |
|---|---|---|---|---|---|
| Path length (nodes) | 8.4 | 8.0 | 7.2 | 8.1 | **10.1** |
| Obstacle avoidance (%) | 29.4 | 31.7 | 33.2 | 44.8 | **94.3** |
| Computation time (ms) | 1.12 | 0.61 | 0.74 | 0.98 | **1.85** |

### Table 3a — Learning rate η ablation

```bash
python experiments/weight_evolution.py
```

### Run all tests

```bash
python -m pytest tests/test_sppp.py -v
```

---

## System Architecture

The SPPP system operates in three phases:

**Phase 1 — User Input and Obstacle Map Integration**
- User preference dimensions I₁–I₈ (binary YES/NO activation)
- Obstacle categories O₁–O₈ (severity scale 0–5)
- Preference–Obstacle Interaction Matrix (Table 2)

**Phase 2 — SPC-Based Route Map Calculation**
- G(s): Euclidean distance from start
- H(s): Euclidean heuristic to goal
- P(s): Weighted semantic obstacle burden with adaptive scaling α(s)
- Dynamic weight evolution: $w_i^{t+1} = w_i^t + \eta \cdot \sum_{s \in \text{path}} \alpha(s) \cdot O_i(s)$

**Phase 3 — Optimal Path Selection**
- Improved A\* selects minimum F(s) node per iteration
- Real-time validation: replanning triggered if severity ≥ 4.5 and ΔF ≥ 1.0
- Mean replan time: 1.20 ms

---

## Obstacle Categories (O₁–O₈)

| ID | Category | Severity scale |
|---|---|---|
| O₁ | Traffic congestion | 0–5 |
| O₂ | Road type / surface condition | 0–5 |
| O₃ | Weather severity | 0–5 |
| O₄ | Pollution / restricted area | 0–5 |
| O₅ | Accessibility constraints | 0–5 |
| O₆ | Dead zones (signal/operational) | 0–5 |
| O₇ | Live events (crowds, closures) | 0–5 |
| O₈ | Construction zones | 0–5 |

---

## User Preference Dimensions (I₁–I₈)

| ID | Preference | Example activation |
|---|---|---|
| I₁ | Speed level | Fast delivery mission |
| I₂ | Time | Time-critical navigation |
| I₃ | Safety | Hazardous environment |
| I₄ | Battery efficiency | Low battery mode |
| I₅ | Emergency response | Ambulance routing |
| I₆ | Comfort | Passenger transport |
| I₇ | Cost | Fuel-cost minimisation |
| I₈ | Scenic beauty | Tourism routing |

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| β | 2.0 | Adaptive scaling magnitude |
| γ | 0.5 | Adaptive scaling sensitivity |
| D_th | 3.0 | Activation threshold (grid units) |
| η | 0.01 | Learning rate for weight evolution |
| Severity max | 5.0 | Maximum obstacle severity per category |

---

## Key Results

| Metric | EDT-A* | **SPPP (proposed)** | Improvement |
|---|---|---|---|
| Obstacle avoidance rate | 31.7% | **94.3%** | +62.6 pp |
| Cumulative score | 27/80 | **67/80** | +148% |
| Replan time | — | **1.20 ms** | Real-time |
| Construction avoidance | Incidental | **100%** (I₅ active) | — |

All improvements statistically significant at p < 0.001 (Wilcoxon signed-rank test).

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{saranya2025sppp,
  title   = {Semantic-Weighted A* Framework with Dynamic User Preference 
             Optimisation for Real-Time Personalised Path Planning in 
             Autonomous Ground Vehicles},
  author  = {Saranya, C and Janaki, G},
  journal = {},
  year    = {2025},
  institution = {SRM Institute of Science and Technology, Kattankulathur}
}
```
