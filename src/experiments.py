# ============================================================
# src/experiments.py
# Expérimentations et comparaisons des différents modes Tabou.
# Toutes les fonctions produisent des figures sauvegardables.
# ============================================================

import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.graph import count_conflicts, objective
from src.tabu import tabu_search

SEED      = 42
MODES     = ['random', 'recolor', 'swap', 'kempe', 'ml']
N_COLORS  = 4


def run_comparison(G, modes: list = MODES, n_runs: int = 10,
                   max_iter: int = 500, tabu_tenure: int = 10,
                   n_colors: int = N_COLORS,
                   rf_model=None, scaler=None,
                   verbose: bool = False) -> dict:
    """
    Lance n_runs de Recherche Tabou pour chaque mode sur un graphe donné.

    Retourne un dict {mode: [{'conflicts', 'n_colors', 'obj', 'time', 'n_iter'}, ...]}
    """
    results = {m: [] for m in modes}

    for mode in modes:
        for run in range(n_runs):
            random.seed(run)
            np.random.seed(run)
            t0 = time.time()
            res = tabu_search(
                G,
                n_colors      = n_colors,
                max_iter      = max_iter,
                tabu_tenure   = tabu_tenure,
                operator_mode = mode,
                ml_model      = rf_model if mode == 'ml' else None,
                scaler        = scaler   if mode == 'ml' else None,
            )
            elapsed = time.time() - t0
            results[mode].append({
                'conflicts': res['best_conflicts'],
                'n_colors':  res['best_n_colors'],
                'obj':       res['best_obj'],
                'time':      elapsed,
                'n_iter':    res['n_iter'],
            })

        avg_c = np.mean([r['conflicts'] for r in results[mode]])
        avg_o = np.mean([r['obj']       for r in results[mode]])
        avg_t = np.mean([r['time']      for r in results[mode]])
        if verbose:
            print(f"  {mode:8s} → conflits_moy={avg_c:.2f} "
                  f"| obj_moy={avg_o:.2f} | temps_moy={avg_t:.3f}s")

    return results


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convertit les résultats en DataFrame récapitulatif."""
    rows = []
    for mode, data in results.items():
        rows.append({
            'Mode':              mode,
            'Conflits moy.':     np.mean([r['conflicts'] for r in data]),
            'Conflits std.':     np.std([r['conflicts']  for r in data]),
            'Conflits min.':     np.min([r['conflicts']  for r in data]),
            'Objectif moy.':     np.mean([r['obj']       for r in data]),
            'Temps moy. (s)':    np.mean([r['time']      for r in data]),
            'Nb couleurs moy.':  np.mean([r['n_colors']  for r in data]),
        })
    return pd.DataFrame(rows).set_index('Mode')


def plot_convergence(G, modes: list = ('random', 'ml', 'kempe'),
                     max_iter: int = 500, n_colors: int = N_COLORS,
                     rf_model=None, scaler=None,
                     save_path: str = None) -> None:
    """Trace les courbes de convergence (objectif + conflits) pour plusieurs modes."""
    random.seed(SEED); np.random.seed(SEED)
    colors_mode = {'random': 'gray', 'ml': 'crimson', 'kempe': 'steelblue',
                   'recolor': 'darkorange', 'swap': 'green'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mode in modes:
        res = tabu_search(G, n_colors=n_colors, max_iter=max_iter,
                          operator_mode=mode,
                          ml_model=rf_model if mode == 'ml' else None,
                          scaler=scaler     if mode == 'ml' else None)
        hist = res['history']
        c = colors_mode.get(mode, None)
        axes[0].plot(hist['obj'],       label=mode, color=c, alpha=0.8)
        axes[1].plot(hist['conflicts'], label=mode, color=c, alpha=0.8)

    axes[0].set_title("Convergence — Objectif",  fontsize=12)
    axes[0].set_xlabel("Itération"); axes[0].set_ylabel("Valeur objectif")
    axes[0].legend()

    axes[1].set_title("Convergence — Conflits", fontsize=12)
    axes[1].set_xlabel("Itération"); axes[1].set_ylabel("Nombre de conflits")
    axes[1].axhline(0, color='green', linestyle='--', alpha=0.5, label='Optimal')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_boxplots(results: dict, save_path: str = None) -> None:
    """Boxplots des conflits et temps de calcul pour chaque mode."""
    modes   = list(results.keys())
    palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    data_c = [[r['conflicts'] for r in results[m]] for m in modes]
    bp1 = axes[0].boxplot(data_c, labels=modes, patch_artist=True)
    for patch, color in zip(bp1['boxes'], palette):
        patch.set_facecolor(color)
    axes[0].set_title("Distribution des conflits", fontsize=12)
    axes[0].set_ylabel("Nombre de conflits")
    axes[0].axhline(0, color='green', linestyle='--', alpha=0.5)

    data_t = [[r['time'] for r in results[m]] for m in modes]
    bp2 = axes[1].boxplot(data_t, labels=modes, patch_artist=True)
    for patch, color in zip(bp2['boxes'], palette):
        patch.set_facecolor(color)
    axes[1].set_title("Temps de calcul", fontsize=12)
    axes[1].set_ylabel("Temps (s)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compare_instances(instances: list, modes: list = ('random', 'ml'),
                       n_runs: int = 5, max_iter: int = 500,
                       n_colors: int = N_COLORS,
                       rf_model=None, scaler=None,
                       save_path: str = None) -> pd.DataFrame:
    """
    Compare les modes sur plusieurs instances.

    instances : liste de (nom, graphe)
    Retourne un DataFrame de résultats.
    """
    rows = []
    for name, G in instances:
        for mode in modes:
            conflicts_list, iters_list = [], []
            for run in range(n_runs):
                random.seed(run); np.random.seed(run)
                res = tabu_search(
                    G, n_colors=n_colors, max_iter=max_iter,
                    operator_mode=mode,
                    ml_model=rf_model if mode == 'ml' else None,
                    scaler=scaler     if mode == 'ml' else None,
                )
                conflicts_list.append(res['best_conflicts'])
                iters_list.append(res['n_iter'])
            rows.append({
                'Instance':      name,
                'Mode':          mode,
                'Conflits moy.': np.mean(conflicts_list),
                'Conflits std.': np.std(conflicts_list),
                'Iters moy.':    np.mean(iters_list),
            })

    df = pd.DataFrame(rows)

    # Visualisation
    fig, ax = plt.subplots(figsize=(11, 5))
    inst_names = df['Instance'].unique()
    x     = np.arange(len(inst_names))
    width = 0.35

    for i, mode in enumerate(modes):
        sub    = df[df['Mode'] == mode]
        values = [sub[sub['Instance'] == inst]['Conflits moy.'].values[0]
                  for inst in inst_names]
        errs   = [sub[sub['Instance'] == inst]['Conflits std.'].values[0]
                  for inst in inst_names]
        ax.bar(x + i * width, values, width, yerr=errs,
               label=f'Tabou {mode}',
               color='gray' if mode == 'random' else 'crimson',
               alpha=0.8, capsize=4)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(inst_names)
    ax.set_ylabel("Conflits moyens")
    ax.set_title("Comparaison Tabou Random vs Tabou+ML — Plusieurs instances", fontsize=12)
    ax.legend()
    ax.axhline(0, color='green', linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return df


def print_summary(results: dict, score_test: float = None) -> None:
    """Affiche un résumé final des performances."""
    print("=" * 65)
    print("RÉSUMÉ FINAL DU PROJET")
    print("=" * 65)
    print()
    print("Problème   : Coloration de graphe (cartes géographiques)")
    print("Métaheur.  : Recherche Tabou")
    print("ML         : Random Forest pour sélection d'opérateur")
    print("Opérateurs : Recolor | Swap | Kempe Chain")
    print()

    for mode, data in results.items():
        avg_c = np.mean([r['conflicts'] for r in data])
        avg_t = np.mean([r['time']      for r in data])
        print(f"  Mode {mode:8s} → conflits_moy={avg_c:.2f} | temps_moy={avg_t:.3f}s")

    if 'random' in results and 'ml' in results:
        random_avg = np.mean([r['conflicts'] for r in results['random']])
        ml_avg     = np.mean([r['conflicts'] for r in results['ml']])
        if random_avg > 0:
            gain = (random_avg - ml_avg) / random_avg * 100
            print(f"\n→ Gain ML vs Random : {gain:.1f}% de conflits en moins")

    if score_test is not None:
        print(f"\nPrécision du classifieur ML : {score_test * 100:.1f}%")