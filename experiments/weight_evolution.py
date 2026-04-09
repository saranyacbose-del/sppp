"""
Weight Evolution Experiment
============================
Demonstrates dynamic preference weight evolution over 20 sequential
route segments, reproducing Table 3a and Figure showing convergence.

Usage
-----
    python experiments/weight_evolution.py

Output
------
    results/table3a_eta_ablation.csv
    results/weight_evolution_20episodes.csv
    figures/weight_evolution.png

Author      : Saranya C, Janaki G
Institution : SRM Institute of Science and Technology, Kattankulathur
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sppp import (
    SPPPSearch, generate_obstacle_map, initialise_weights,
    update_weights, compute_obstacle_avoidance_rate, PREFERENCE_SCENARIOS
)
from sppp.spc_algorithm import ETA, OBSTACLE_KEYS


GRID_SIZE = (8, 8)
START = (0, 2)
GOAL = (7, 4)
ACTIVE_PREFS = ['I2', 'I3', 'I5']
N_EPISODES = 20
RANDOM_SEED = 42


def run_eta_ablation(eta_values=None, num_runs=500):
    """
    Table 3a: Learning rate η ablation study.
    Tests η ∈ {0.001, 0.01, 0.1} across 500 simulation configurations.
    """
    if eta_values is None:
        eta_values = [0.001, 0.01, 0.1]

    print("\n[Table 3a] Running η ablation study...")
    results = {}

    for eta in eta_values:
        variances, convergences, avoidances = [], [], []

        for i in range(num_runs):
            seed = RANDOM_SEED + i
            obs_map = generate_obstacle_map(GRID_SIZE, seed=seed,
                                            environment='urban')
            weights = initialise_weights(ACTIVE_PREFS)
            path_nodes = [START, (0,3),(1,4),(2,4),(3,4),
                          (4,5),(5,4),(6,5), GOAL]

            prev_weights = None
            converged_at = 200

            for episode in range(1, 201):
                prev_weights = dict(weights)

                # Corrected gradient with adaptive scaling
                gradient = {p: 0.0 for p in ACTIVE_PREFS}
                from sppp.spc_algorithm import (
                    compute_P, validate_obstacle_scores,
                    compute_H, INTERACTION_MATRIX, PREFERENCE_KEYS
                )
                for node in path_nodes:
                    obs = validate_obstacle_scores(
                        obs_map.get(node, {k: 0.0 for k in OBSTACLE_KEYS})
                    )
                    H = compute_H(node, GOAL)
                    from sppp.spc_algorithm import BETA, GAMMA, D_TH
                    import math
                    alpha = 1 + BETA / (1 + math.exp(-GAMMA*(H - D_TH)))
                    for pref in ACTIVE_PREFS:
                        pref_idx = PREFERENCE_KEYS.index(pref)
                        obs_sum = sum(
                            obs[ok] for ok in OBSTACLE_KEYS
                            if INTERACTION_MATRIX[ok][pref_idx] == 1
                        )
                        gradient[pref] += alpha * obs_sum

                for pref in ACTIVE_PREFS:
                    weights[pref] = weights[pref] + eta * gradient[pref]

                total = sum(weights[p] for p in ACTIVE_PREFS)
                if total > 0:
                    weights = {p: weights[p]/total for p in ACTIVE_PREFS}

                # Check convergence
                if prev_weights:
                    max_diff = max(
                        abs(weights[p] - prev_weights[p])
                        for p in ACTIVE_PREFS
                    )
                    if max_diff < 1e-4:
                        converged_at = episode
                        break

            n = len(ACTIVE_PREFS)
            uniform = 1.0 / n
            variance = sum(
                (weights[p] - uniform)**2 for p in ACTIVE_PREFS
            ) / n
            variances.append(variance)
            convergences.append(converged_at)

            avoidance = compute_obstacle_avoidance_rate(
                path_nodes, obs_map, ACTIVE_PREFS, weights
            )
            avoidances.append(avoidance)

        results[eta] = {
            'variance': round(np.mean(variances), 5),
            'convergence': round(np.mean(convergences), 1),
            'avoidance': round(np.mean(avoidances), 1),
        }

    print("\nTable 3a. Effect of learning rate η on weight evolution")
    print("-" * 65)
    print(f"{'η value':<12} {'Weight Variance':>18} "
          f"{'Convergence (ep)':>18} {'Avoidance (%)':>15}")
    print("-" * 65)
    for eta, vals in results.items():
        marker = " ← proposed" if eta == 0.01 else ""
        print(f"{eta:<12} {vals['variance']:>18} "
              f"{vals['convergence']:>18} "
              f"{vals['avoidance']:>15}{marker}")
    print("-" * 65)

    return results


def run_weight_evolution(n_episodes=N_EPISODES):
    """
    Run 20-episode sequential weight evolution experiment.
    Weights carried forward between episodes (not reset).
    """
    print(f"\n[Weight Evolution] Running {n_episodes} sequential episodes...")
    weights = initialise_weights(ACTIVE_PREFS)
    history = []

    for episode in range(1, n_episodes + 1):
        seed = RANDOM_SEED + episode
        obs_map = generate_obstacle_map(GRID_SIZE, seed=seed,
                                        environment='urban')
        sppp = SPPPSearch(GRID_SIZE, START, GOAL, obs_map, ACTIVE_PREFS)
        sppp.weights = dict(weights)  # carry forward
        result = sppp.search()

        if result.success:
            weights = update_weights(
                weights, result.path, obs_map, ACTIVE_PREFS
            )
            avoidance = compute_obstacle_avoidance_rate(
                result.path, obs_map, ACTIVE_PREFS, weights
            )
        else:
            avoidance = 0.0

        history.append({
            'episode': episode,
            'w_I2': round(weights.get('I2', 0.0), 4),
            'w_I3': round(weights.get('I3', 0.0), 4),
            'w_I5': round(weights.get('I5', 0.0), 4),
            'avoidance': avoidance,
        })

    print("\nWeight Evolution over 20 Sequential Route Segments (S3)")
    print("-" * 65)
    print(f"{'Episode':>8} {'w(I2-Time)':>12} "
          f"{'w(I3-Safety)':>14} {'w(I5-Emerg)':>13} {'Avoid %':>8}")
    print("-" * 65)
    for row in [1, 5, 10, 15, 20]:
        h = next(r for r in history if r['episode'] == row)
        print(f"{h['episode']:>8} {h['w_I2']:>12} "
              f"{h['w_I3']:>14} {h['w_I5']:>13} {h['avoidance']:>8}")
    print("-" * 65)

    return history


def save_results(eta_results, evolution_history, output_dir='results'):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Table 3a
    with open(f'{output_dir}/table3a_eta_ablation.csv', 'w',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eta', 'weight_variance',
                         'convergence_episodes', 'avoidance_rate'])
        for eta, vals in eta_results.items():
            writer.writerow([
                eta, vals['variance'],
                vals['convergence'], vals['avoidance']
            ])
    print(f"\nSaved: {output_dir}/table3a_eta_ablation.csv")

    # Weight evolution
    with open(f'{output_dir}/weight_evolution_20episodes.csv', 'w',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'episode', 'w_I2', 'w_I3', 'w_I5', 'avoidance'
        ])
        writer.writeheader()
        writer.writerows(evolution_history)
    print(f"Saved: {output_dir}/weight_evolution_20episodes.csv")


def plot_weight_evolution(history, output_dir='figures'):
    """Generate weight evolution figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        episodes = [h['episode'] for h in history]
        w_I2 = [h['w_I2'] for h in history]
        w_I3 = [h['w_I3'] for h in history]
        w_I5 = [h['w_I5'] for h in history]
        avoid = [h['avoidance'] for h in history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                        facecolor='white')

        ax1.plot(episodes, w_I2, 'b-o', markersize=5, label='w(I₂) — Time')
        ax1.plot(episodes, w_I3, 'g-s', markersize=5, label='w(I₃) — Safety')
        ax1.plot(episodes, w_I5, 'r-^', markersize=5,
                 label='w(I₅) — Emergency')
        ax1.axhline(y=1/3, color='gray', linestyle='--',
                    alpha=0.5, label='Uniform prior (0.333)')
        ax1.set_xlabel('Route segment (episode)', fontsize=11)
        ax1.set_ylabel('Preference weight', fontsize=11)
        ax1.set_title(
            'Dynamic preference weight evolution over 20 sequential '
            'route segments\n(Scenario S3: I₂ Time + I₃ Safety + '
            'I₅ Emergency, η = 0.01)',
            fontsize=10
        )
        ax1.legend(fontsize=9)
        ax1.set_xlim(1, 20)
        ax1.grid(True, linewidth=0.3, color='#EEEEEE')
        for spine in ax1.spines.values():
            spine.set_linewidth(0.5)

        ax2.plot(episodes, avoid, 'k-D', markersize=5,
                 label='Obstacle avoidance rate (%)')
        ax2.set_xlabel('Route segment (episode)', fontsize=11)
        ax2.set_ylabel('Avoidance rate (%)', fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.set_xlim(1, 20)
        ax2.legend(fontsize=9)
        ax2.grid(True, linewidth=0.3, color='#EEEEEE')
        for spine in ax2.spines.values():
            spine.set_linewidth(0.5)

        plt.tight_layout()
        path = f'{output_dir}/weight_evolution.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path}")
        plt.close()
    except ImportError:
        print("matplotlib not available — skipping figure generation")


if __name__ == '__main__':
    eta_results = run_eta_ablation(num_runs=500)
    history = run_weight_evolution(n_episodes=20)
    save_results(eta_results, history)
    plot_weight_evolution(history)
