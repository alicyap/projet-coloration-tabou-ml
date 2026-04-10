# ============================================================
# graph.py — Modélisation du problème de coloration
# Responsable : Samira (Modélisation & opérateurs de voisinage)
# ============================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEED = 42


# ============================================================
# Génération des instances de graphe
# ============================================================

# Coordonnées approximatives (lon, lat) du centre de chaque région métropolitaine française
FRANCE_REGIONS = {
    "Île-de-France":            (2.35,  48.85),
    "Hauts-de-France":          (2.60,  50.25),
    "Grand Est":                (6.80,  48.60),
    "Normandie":                (0.10,  49.15),
    "Bretagne":                 (-2.80, 48.10),
    "Pays de la Loire":         (-1.00, 47.50),
    "Centre-Val de Loire":      (1.80,  47.50),
    "Bourgogne-Franche-Comté":  (5.00,  47.00),
    "Auvergne-Rhône-Alpes":     (4.80,  45.50),
    "Occitanie":                (2.50,  43.60),
    "Nouvelle-Aquitaine":       (-0.50, 44.80),
    "Provence-Alpes-Côte d'Azur": (6.00, 43.90),
    "Corse":                    (9.10,  42.00),
}

# Frontières réelles entre régions métropolitaines
FRANCE_ADJACENCY = [
    ("Île-de-France",       "Hauts-de-France"),
    ("Île-de-France",       "Grand Est"),
    ("Île-de-France",       "Normandie"),
    ("Île-de-France",       "Centre-Val de Loire"),
    ("Île-de-France",       "Bourgogne-Franche-Comté"),
    ("Hauts-de-France",     "Grand Est"),
    ("Hauts-de-France",     "Normandie"),
    ("Grand Est",           "Bourgogne-Franche-Comté"),
    ("Normandie",           "Bretagne"),
    ("Normandie",           "Pays de la Loire"),
    ("Normandie",           "Centre-Val de Loire"),
    ("Bretagne",            "Pays de la Loire"),
    ("Pays de la Loire",    "Centre-Val de Loire"),
    ("Pays de la Loire",    "Nouvelle-Aquitaine"),
    ("Centre-Val de Loire", "Bourgogne-Franche-Comté"),
    ("Centre-Val de Loire", "Nouvelle-Aquitaine"),
    ("Centre-Val de Loire", "Auvergne-Rhône-Alpes"),
    ("Centre-Val de Loire", "Occitanie"),
    ("Bourgogne-Franche-Comté", "Auvergne-Rhône-Alpes"),
    ("Auvergne-Rhône-Alpes", "Provence-Alpes-Côte d'Azur"),
    ("Auvergne-Rhône-Alpes", "Occitanie"),
    ("Occitanie",           "Nouvelle-Aquitaine"),
    ("Occitanie",           "Provence-Alpes-Côte d'Azur"),
    # La Corse n'a pas de frontière terrestre, on la connecte à PACA par mer
    ("Corse",               "Provence-Alpes-Côte d'Azur"),
]


def generate_map_graph(n_regions=None, seed=SEED):
    """
    Génère un graphe représentant la carte des 13 régions métropolitaines françaises.
    Les nœuds sont les régions, les arêtes représentent les frontières réelles.

    Le paramètre n_regions est conservé pour compatibilité avec le reste du code,
    mais ignoré : on utilise toujours les 13 régions officielles.

    Attributs des nœuds :
        - 'pos' : (longitude, latitude) pour la visualisation
        - 'label' : nom de la région
    """
    G = nx.Graph()

    # Ajout des nœuds avec attributs de position
    region_names = list(FRANCE_REGIONS.keys())
    for idx, name in enumerate(region_names):
        lon, lat = FRANCE_REGIONS[name]
        G.add_node(idx, pos=(lon, lat), label=name)

    # Index name → id
    name_to_id = {name: i for i, name in enumerate(region_names)}

    # Ajout des arêtes selon les frontières réelles
    for r1, r2 in FRANCE_ADJACENCY:
        G.add_edge(name_to_id[r1], name_to_id[r2])

    # Garantit la connexité (normalement déjà vérifié via les adjacences)
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
    f(s) = alpha * conflits(s) + beta * nb_couleurs(s)
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
# Visualisation géographique (carte de France)
# ============================================================

def visualize_coloring(G, coloring, title="Coloration du graphe", seed=SEED):
    """
    Affiche le graphe avec sa coloration.

    Si les nœuds ont un attribut 'pos' (cas carte de France), utilise les
    coordonnées géographiques réelles pour le placement. Sinon, utilise
    spring_layout (cas graphes DSJC ou génériques).

    coloring : dict {nœud: couleur (int)}
    """
    palette     = plt.cm.get_cmap('tab10', max(coloring.values()) + 1)
    node_colors = [palette(coloring[n]) for n in G.nodes()]

    conflict_edges = [(u, v) for u, v in G.edges() if coloring[u] == coloring[v]]
    normal_edges   = [(u, v) for u, v in G.edges() if coloring[u] != coloring[v]]

    # Utilise les coordonnées géographiques si disponibles (carte de France)
    node_data = dict(G.nodes(data=True))
    if all('pos' in node_data[n] for n in G.nodes()):
        pos = {n: node_data[n]['pos'] for n in G.nodes()}
        labels = {n: node_data[n].get('label', str(n)) for n in G.nodes()}
    else:
        pos = nx.spring_layout(G, seed=seed)
        labels = {n: str(n) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(11, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6.5, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=conflict_edges,
                           edge_color='red', width=2.5, ax=ax)

    n_conflicts = len(conflict_edges)
    n_colors_   = len(set(coloring.values()))
    ax.set_title(f"{title}\nCouleurs={n_colors_} | Conflits={n_conflicts}", fontsize=12)
    if conflict_edges:
        ax.legend(handles=[
            mpatches.Patch(color='red', label=f'{n_conflicts} conflit(s)')
        ])

    # Ajout d'un fond de carte simplifié (optionnel, silencieux si cartopy absent)
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax.set_extent([-5.5, 10, 41, 52])
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4)
        ax.add_feature(cfeature.COASTLINE, alpha=0.4)
    except ImportError:
        pass  # cartopy non disponible, affichage sans fond de carte

    plt.tight_layout()
    plt.show()
