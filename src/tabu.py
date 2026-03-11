# ============================================================
# tabu.py — Algorithme de Recherche Tabou
# Responsable : [Prénom NOM]
# ============================================================

import random
import numpy as np
import networkx as nx
from collections import deque, defaultdict

from src.graph import initial_solution, objective, count_conflicts
from src.operators import op_recolor, op_swap, op_kempe_chain

SEED = 42


def extract_features(G, coloring, no_improve_count, max_iter, n_colors):
    """
    Calcule les 6 features de l'état courant pour le modèle ML.

    Features :
        0 - ratio_conflits            : conflits / nb_arêtes
        1 - plateau_norm              : itérations sans amélioration / max_iter
        2 - densite_graphe            : densité du graphe
        3 - entropie_couleurs         : entropie de la distribution des couleurs
        4 - taux_utilisation_couleurs : nb_couleurs_utilisées / n_colors
        5 - conflits_par_noeud        : conflits / nb_nœuds
    """
    n_conflicts    = count_conflicts(G, coloring)
    conflict_ratio = n_conflicts / max(G.number_of_edges(), 1)

    color_counts = defaultdict(int)
    for c in coloring.values():
        color_counts[c] += 1

    color_entropy = -sum(
        (v / G.number_of_nodes()) * np.log(v / G.number_of_nodes() + 1e-9)
        for v in color_counts.values()
    )

    return [
        conflict_ratio,
        no_improve_count / max(max_iter, 1),
        nx.density(G),
        color_entropy,
        len(set(coloring.values())) / n_colors,
        n_conflicts / max(G.number_of_nodes(), 1),
    ]


def tabu_search(
    G,
    n_colors,
    max_iter      = 500,
    tabu_tenure   = 10,
    operator_mode = 'random',   # 'random' | 'recolor' | 'swap' | 'kempe' | 'ml'
    ml_model      = None,
    scaler        = None,
    verbose       = False
):
    """
    Recherche Tabou pour la coloration de graphe.

    Paramètres :
        G             : graphe NetworkX
        n_colors      : nombre de couleurs autorisées
        max_iter      : nombre max d'itérations
        tabu_tenure   : durée de vie d'un mouvement dans la liste tabou
        operator_mode : stratégie de sélection d'opérateur
        ml_model      : modèle ML entraîné (requis si operator_mode='ml')
        scaler        : StandardScaler associé au modèle ML
        verbose       : affichage des logs

    Retourne :
        dict avec best_coloring, best_obj, best_conflicts, best_n_colors,
        history (obj, conflicts, n_colors, operators, features), n_iter
    """
    # --- Initialisation ---
    current   = initial_solution(G, n_colors)
    best      = dict(current)
    best_obj  = objective(G, best)
    tabu_list = deque(maxlen=tabu_tenure * G.number_of_nodes())

    history = {
        'obj'      : [],
        'conflicts': [],
        'n_colors' : [],
        'operators': [],
        'features' : [],
    }

    no_improve_count = 0
    operators        = ['recolor', 'swap', 'kempe']

    for it in range(max_iter):
        # --- Features de l'état courant ---
        features = extract_features(G, current, no_improve_count, max_iter, n_colors)
        history['features'].append(features)

        # --- Sélection de l'opérateur ---
        if operator_mode == 'random':
            op = random.choice(operators)
        elif operator_mode in operators:
            op = operator_mode
        elif operator_mode == 'ml' and ml_model is not None:
            feat_scaled = scaler.transform([features])
            op_idx      = ml_model.predict(feat_scaled)[0]
            op          = operators[op_idx]
        else:
            # Heuristique de repli : Kempe si plateau prolongé, sinon Recolor
            op = 'kempe' if no_improve_count > 20 else 'recolor'

        # --- Application de l'opérateur ---
        chain_len = 0
        if op == 'recolor':
            neighbor, move, delta = op_recolor(G, current, n_colors, tabu_list)
        elif op == 'swap':
            neighbor, move, delta = op_swap(G, current, n_colors, tabu_list)
        else:
            neighbor, move, delta, chain_len = op_kempe_chain(
                G, current, n_colors, tabu_list)

        history['operators'].append(op)

        # --- Fallback si aucun mouvement non-tabou ---
        if neighbor is None:
            neighbor, move, delta = op_recolor(G, current, n_colors, set())
            if neighbor is None:
                break

        # --- Critère d'aspiration + mise à jour ---
        new_obj = objective(G, neighbor)
        if new_obj < best_obj or move not in tabu_list:
            current = neighbor
            if move:
                tabu_list.append(move)

        if new_obj < best_obj:
            best             = dict(current)
            best_obj         = new_obj
            no_improve_count = 0
        else:
            no_improve_count += 1

        # --- Enregistrement historique ---
        history['obj'].append(new_obj)
        history['conflicts'].append(count_conflicts(G, current))
        history['n_colors'].append(len(set(current.values())))

        if verbose and it % 100 == 0:
            print(f"  iter={it:4d} | obj={new_obj:.1f} "
                  f"| conflicts={count_conflicts(G, current)} "
                  f"| op={op:7s} | plateau={no_improve_count}")

        # --- Arrêt anticipé si solution parfaite ---
        if count_conflicts(G, best) == 0:
            if verbose:
                print(f"  ✅ Solution sans conflit trouvée à l'itération {it}")
            break

    return {
        'best_coloring' : best,
        'best_obj'      : best_obj,
        'best_conflicts': count_conflicts(G, best),
        'best_n_colors' : len(set(best.values())),
        'history'       : history,
        'n_iter'        : it + 1,
    }
