# ============================================================
# graph.py — Modélisation du problème de coloration
# Responsable : [Prénom NOM]
# ============================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEED = 42


# ============================================================
# Génération des instances de graphe
# ============================================================

def generate_map_graph(n_regions=20, seed=SEED):
    """
    Génère un graphe planaire simulant une carte géographique.
    Utilise un graphe de Delaunay approximé via positions aléatoires.
    """
    np.random.seed(seed)
    G = nx.random_geometric_graph(n_regions, radius=3.5, seed=seed)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v)
    return G


def generate_dsjc_like(n, p, seed=SEED):
    """
    Génère un graphe aléatoire de type DSJC (benchmark classique).
    n = nombre de nœuds, p = probabilité d'arête.
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v)
    return G


# ============================================================
# Fonctions utilitaires : solution, conflits, objectif
# ============================================================

def count_conflicts(G, coloring):
    """Nombre d'arêtes dont les deux extrémités ont la même couleur."""
    return sum(1 for u, v in G.edges() if coloring[u] == coloring[v])


def objective(G, coloring, alpha=10, beta=1):
    """
    Fonction objectif à minimiser.
    alpha * conflits + beta * nb_couleurs
    """
    conflicts = count_conflicts(G, coloring)
    n_colors  = len(set(coloring.values()))
    return alpha * conflicts + beta * n_colors


def initial_solution(G, n_colors=None):
    """
    Solution initiale par algorithme greedy (largest_first).
    Si n_colors est fourni, on force l'usage de k couleurs (peut créer des conflits).
    """
    greedy = nx.coloring.greedy_color(G, strategy='largest_first')
    if n_colors is None:
        return dict(greedy)
    return {node: color % n_colors for node, color in greedy.items()}


def solution_info(G, coloring):
    """Affiche un résumé de la solution."""
    c = count_conflicts(G, coloring)
    k = len(set(coloring.values()))
    print(f"  Conflits    : {c}")
    print(f"  Nb couleurs : {k}")
    print(f"  Objectif    : {objective(G, coloring)}")


# ============================================================
# Visualisation
# ============================================================

def visualize_coloring(G, coloring, title="Coloration du graphe", seed=SEED):
    """
    Affiche le graphe avec sa coloration.
    coloring : dict {nœud: couleur (int)}
    """
    palette     = plt.cm.get_cmap('tab10', max(coloring.values()) + 1)
    node_colors = [palette(coloring[n]) for n in G.nodes()]

    conflict_edges = [(u, v) for u, v in G.edges() if coloring[u] == coloring[v]]
    normal_edges   = [(u, v) for u, v in G.edges() if coloring[u] != coloring[v]]

    pos = nx.spring_layout(G, seed=seed)
    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=conflict_edges,
                           edge_color='red', width=2.5, ax=ax)

    n_conflicts = len(conflict_edges)
    n_colors    = len(set(coloring.values()))
    ax.set_title(f"{title}\nCouleurs={n_colors} | Conflits={n_conflicts}", fontsize=12)
    if conflict_edges:
        ax.legend(handles=[
            mpatches.Patch(color='red', label=f'{n_conflicts} conflit(s)')
        ])
    plt.tight_layout()
    plt.show()
