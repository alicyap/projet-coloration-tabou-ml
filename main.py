#!/usr/bin/env python3
# ============================================================
# main.py — Script principal d'expérimentation
# Lance toutes les expériences et affiche les résultats
# ============================================================

import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.graph import (
    generate_map_graph, generate_dsjc_like,
    count_conflicts, solution_info, visualize_coloring
)
from src.tabu import tabu_search
from src.ml_selector import collect_training_data, train_model, evaluate_model

SEED     = 42
N_COLORS = 4
MAX_ITER = 500
N_RUNS   = 10

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. Instances de test
# ============================================================

print("=" * 60)
print("Génération des instances")
print("=" * 60)

G_small  = generate_map_graph(n_regions=15, seed=42)
G_medium = generate_map_graph(n_regions=30, seed=43)
G_large  = generate_map_graph(n_regions=50, seed=44)
G_dsjc   = generate_dsjc_like(n=50, p=0.3, seed=42)

for name, G in [("G_small", G_small), ("G_medium", G_medium),
                ("G_large", G_large),  ("G_dsjc",   G_dsjc)]:
    print(f"  {name:10s} | nœuds={G.number_of_nodes():3d} "
          f"| arêtes={G.number_of_edges():4d}")


# ============================================================
# 2. Collecte des données ML
# ============================================================

print("\n" + "=" * 60)
print("Collecte des données d'entraînement ML")
print("=" * 60)

train_graphs = [
    generate_map_graph(n_regions=n, seed=s)
    for n, s in [(15,1),(20,2),(25,3),(30,4),(35,5),(40,6)]
]

t0 = time.time()
X_raw, y_raw = collect_training_data(
    train_graphs, n_colors=N_COLORS, n_runs_per_graph=3, max_iter=200)
print(f"  Terminé en {time.time()-t0:.1f}s")
print(f"  Échantillons : {len(X_raw)}")
print(f"  recolor={sum(y_raw==0)} | swap={sum(y_raw==1)} | kempe={sum(y_raw==2)}")


# ============================================================
# 3. Entraînement du modèle ML
# ============================================================

print("\n" + "=" * 60)
print("Entraînement du Random Forest")
print("=" * 60)

rf_model, scaler, df_features = train_model(
    X_raw, y_raw, train_graphs=train_graphs, n_colors=N_COLORS)

evaluate_model(rf_model, scaler, df_features)


# ============================================================
# 4. Comparaison des modes sur G_medium
# ============================================================

print("\n" + "=" * 60)
print("Comparaison des modes sur G_medium (10 runs chacun)")
print("=" * 60)

modes       = ['random', 'recolor', 'swap', 'kempe', 'ml']
results_all = {m: [] for m in modes}

for mode in modes:
    for run in range(N_RUNS):
        random.seed(run)
        np.random.seed(run)
        res = tabu_search(
            G_medium,
            n_colors      = N_COLORS,
            max_iter      = MAX_ITER,
            tabu_tenure   = 10,
            operator_mode = mode,
            ml_model      = rf_model if mode == 'ml' else None,
            scaler        = scaler   if mode == 'ml' else None,
        )
        results_all[mode].append({
            'conflicts': res['best_conflicts'],
            'n_colors' : res['best_n_colors'],
            'obj'      : res['best_obj'],
            'n_iter'   : res['n_iter'],
        })

    avg_c = np.mean([r['conflicts'] for r in results_all[mode]])
    avg_o = np.mean([r['obj']       for r in results_all[mode]])
    print(f"  {mode:8s} → conflits_moy={avg_c:.2f} | obj_moy={avg_o:.2f}")


# ============================================================
# 5. Tableau récapitulatif
# ============================================================

rows = []
for mode in modes:
    data = results_all[mode]
    rows.append({
        'Mode'            : mode,
        'Conflits moy.'   : np.mean([r['conflicts'] for r in data]),
        'Conflits std.'   : np.std([r['conflicts']  for r in data]),
        'Conflits min.'   : np.min([r['conflicts']  for r in data]),
        'Objectif moy.'   : np.mean([r['obj']       for r in data]),
        'Nb couleurs moy.': np.mean([r['n_colors']  for r in data]),
    })

df_results = pd.DataFrame(rows).set_index('Mode')
print("\nRésultats moyens sur G_medium :")
print(df_results.round(3).to_string())

# Sauvegarde CSV
df_results.to_csv("results/comparaison_modes.csv")
print("\n✅ Tableau sauvegardé dans results/comparaison_modes.csv")


# ============================================================
# 6. Courbes de convergence
# ============================================================

random.seed(SEED); np.random.seed(SEED)

colors_mode  = {'random': 'gray', 'ml': 'crimson', 'kempe': 'steelblue'}
conv_results = {}
for mode in ['random', 'ml', 'kempe']:
    conv_results[mode] = tabu_search(
        G_medium, n_colors=N_COLORS, max_iter=MAX_ITER, tabu_tenure=10,
        operator_mode=mode,
        ml_model=rf_model if mode=='ml' else None,
        scaler=scaler   if mode=='ml' else None,
    )

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for mode in ['random', 'ml', 'kempe']:
    hist = conv_results[mode]['history']
    axes[0].plot(hist['obj'],       label=mode, color=colors_mode[mode], alpha=0.8)
    axes[1].plot(hist['conflicts'], label=mode, color=colors_mode[mode], alpha=0.8)

axes[0].set_title("Convergence — Objectif",  fontsize=12)
axes[0].set_xlabel("Itération"); axes[0].set_ylabel("Valeur objectif")
axes[0].legend()
axes[1].set_title("Convergence — Conflits", fontsize=12)
axes[1].set_xlabel("Itération"); axes[1].set_ylabel("Nombre de conflits")
axes[1].legend()

plt.suptitle("Tabou : Random vs Kempe vs ML — G_medium", fontsize=13)
plt.tight_layout()
plt.savefig("results/convergence_curves.png", dpi=150)
plt.show()
print("✅ Courbes sauvegardées dans results/convergence_curves.png")


# ============================================================
# 7. Meilleure solution ML
# ============================================================

random.seed(SEED); np.random.seed(SEED)
best_ml = tabu_search(
    G_medium, n_colors=N_COLORS, max_iter=MAX_ITER,
    tabu_tenure=10, operator_mode='ml',
    ml_model=rf_model, scaler=scaler
)

print("\nMeilleure solution Tabou+ML (G_medium) :")
solution_info(G_medium, best_ml['best_coloring'])
visualize_coloring(G_medium, best_ml['best_coloring'],
                   "Meilleure solution — Tabou+ML (G_medium)")
