import random
from collections import deque
from itertools import combinations

from src.graph import objective


# ============================================================
# Opérateur 1 : RECOLOR
# Génère le meilleur voisin par recoloration d'un nœud en conflit.
# ============================================================

def op_recolor(G, coloring, n_colors, tabu_list):
    best_neighbor = None
    best_delta    = float('inf')
    best_move     = None

    conflict_nodes = [
        u for u in G.nodes()
        if any(coloring[u] == coloring[v] for v in G.neighbors(u))
    ]
    if not conflict_nodes:
        conflict_nodes = list(G.nodes())

    current_obj = objective(G, coloring)

    for node in conflict_nodes:
        old_color = coloring[node]
        for new_color in range(n_colors):
            if new_color == old_color:
                continue
            move = (node, old_color, new_color)
            if move in tabu_list:
                continue
            new_col = dict(coloring)
            new_col[node] = new_color
            delta = objective(G, new_col) - current_obj
            if delta < best_delta:
                best_delta    = delta
                best_neighbor = new_col
                best_move     = move

    return best_neighbor, best_move, best_delta


# ============================================================
# Opérateur 2 : SWAP
# Génère le meilleur voisin par échange de couleurs entre deux nœuds.
# ============================================================

def op_swap(G, coloring, n_colors, tabu_list, n_candidates=20):
    best_neighbor = None
    best_delta    = float('inf')
    best_move     = None

    current_obj = objective(G, coloring)
    nodes       = list(G.nodes())
    pairs       = [random.sample(nodes, 2) for _ in range(n_candidates)]

    for u, v in pairs:
        if coloring[u] == coloring[v]:
            continue
        move = ('swap', u, v, coloring[u], coloring[v])
        if move in tabu_list:
            continue
        new_col = dict(coloring)
        new_col[u], new_col[v] = coloring[v], coloring[u]
        delta = objective(G, new_col) - current_obj
        if delta < best_delta:
            best_delta    = delta
            best_neighbor = new_col
            best_move     = move

    return best_neighbor, best_move, best_delta


# ============================================================
# Opérateur 3 : KEMPE CHAIN
# ============================================================

def get_kempe_chain(G, coloring, start_node, color_i, color_j):
# Trouver la chaîne de Kempe contenant start_node pour les couleurs i et j.
    if coloring[start_node] not in (color_i, color_j):
        return set()

    chain   = set()
    queue   = deque([start_node])
    visited = {start_node}

    while queue:
        node = queue.popleft()
        chain.add(node)
        # BFS dans le sous-graphe
        for neighbor in G.neighbors(node):
            if neighbor not in visited and coloring[neighbor] in (color_i, color_j):
                visited.add(neighbor)
                queue.append(neighbor)

    return chain

# Appliquer l'inversion de Kempe pour tous les nœuds de la chaîne.
def apply_kempe_chain(coloring, chain, color_i, color_j):
    new_col = dict(coloring)
    for node in chain:
        if new_col[node] == color_i:
            new_col[node] = color_j
        elif new_col[node] == color_j:
            new_col[node] = color_i
    return new_col

# Générer le meilleur voisin par inversion de chaîne de Kempe.
def op_kempe_chain(G, coloring, n_colors, tabu_list, n_candidates=15):
    best_neighbor  = None
    best_delta     = float('inf')
    best_move      = None
    best_chain_len = 0

    current_obj = objective(G, coloring)
    nodes       = list(G.nodes())
    color_pairs = list(combinations(range(n_colors), 2))

    conflict_nodes = [
        u for u in G.nodes()
        if any(coloring[u] == coloring[v] for v in G.neighbors(u))
    ]
    candidate_nodes = conflict_nodes if conflict_nodes else nodes

    attempts = 0
    random.shuffle(candidate_nodes)
    random.shuffle(color_pairs)

    #Tester n_candidates combinaisons (nœud_départ, couleur_i, couleur_j).
    for start in candidate_nodes:
        for (ci, cj) in color_pairs:
            if attempts >= n_candidates:
                break
            if coloring[start] not in (ci, cj):
                continue

            chain = get_kempe_chain(G, coloring, start, ci, cj)
            if len(chain) < 2:
                continue

            move = ('kempe', start, ci, cj, len(chain))
            # Vérification tabou sur les 4 premiers éléments du mouvement
            if move[:4] in [
                (m[:4] if isinstance(m, tuple) and m[0] == 'kempe' else m)
                for m in tabu_list
            ]:
                continue

            new_col = apply_kempe_chain(coloring, chain, ci, cj)
            delta   = objective(G, new_col) - current_obj

            if delta < best_delta:
                best_delta     = delta
                best_neighbor  = new_col
                best_move      = move
                best_chain_len = len(chain)

            attempts += 1
        if attempts >= n_candidates:
            break

    return best_neighbor, best_move, best_delta, best_chain_len
