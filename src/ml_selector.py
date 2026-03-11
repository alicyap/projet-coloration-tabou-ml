# ============================================================
# ml_selector.py — Sélection d'opérateur par Machine Learning
# Responsable : [Prénom NOM]
# ============================================================
#
# Pipeline :
#   1. collect_training_data() — génère le dataset supervisé
#   2. train_model()           — entraîne le Random Forest
#   3. evaluate_model()        — évalue et affiche les métriques
# ============================================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import deque, defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

import networkx as nx

from src.graph import initial_solution, objective, count_conflicts
from src.operators import op_recolor, op_swap, op_kempe_chain

SEED = 42

FEATURE_NAMES = [
    'ratio_conflits',
    'plateau_norm',
    'densite_graphe',
    'entropie_couleurs',
    'taux_utilisation_couleurs',
    'conflits_par_noeud',
]

OPERATOR_MAP     = {'recolor': 0, 'swap': 1, 'kempe': 2}
OPERATOR_NAMES   = ['recolor', 'swap', 'kempe']


# ============================================================
# 1. Collecte des données d'entraînement
# ============================================================

def collect_training_data(graphs, n_colors=4, n_runs_per_graph=5, max_iter=300):
    """
    Génère le dataset supervisé pour entraîner le classifieur ML.

    Pour chaque itération de chaque run :
      - calcule les features de l'état courant
      - teste les 3 opérateurs et étiquette avec le meilleur delta

    Label : 0=recolor, 1=swap, 2=kempe

    Retourne :
        X : np.array (n_samples, n_features)
        y : np.array (n_samples,)
    """
    X, y = [], []

    for G in graphs:
        for run in range(n_runs_per_graph):
            seed_run = SEED + run
            random.seed(seed_run)
            np.random.seed(seed_run)

            current   = initial_solution(G, n_colors)
            tabu_list = deque(maxlen=10 * G.number_of_nodes())
            no_improve = 0

            for it in range(max_iter):
                current_obj = objective(G, current)
                n_conflicts = count_conflicts(G, current)

                # Features
                conflict_ratio = n_conflicts / max(G.number_of_edges(), 1)
                color_counts   = defaultdict(int)
                for c in current.values():
                    color_counts[c] += 1
                color_entropy = -sum(
                    (v / G.number_of_nodes()) *
                    np.log(v / G.number_of_nodes() + 1e-9)
                    for v in color_counts.values()
                )
                features = [
                    conflict_ratio,
                    no_improve / max(max_iter, 1),
                    nx.density(G),
                    color_entropy,
                    len(set(current.values())) / n_colors,
                    n_conflicts / max(G.number_of_nodes(), 1),
                ]

                # Tester les 3 opérateurs — étiqueter avec le meilleur delta
                results = {}
                r, _, d    = op_recolor(G, current, n_colors, tabu_list)
                results['recolor'] = d if r is not None else float('inf')

                r, _, d    = op_swap(G, current, n_colors, tabu_list)
                results['swap']    = d if r is not None else float('inf')

                r, _, d, _ = op_kempe_chain(G, current, n_colors, tabu_list)
                results['kempe']   = d if r is not None else float('inf')

                best_op = min(results, key=results.get)
                if results[best_op] == float('inf'):
                    break

                X.append(features)
                y.append(OPERATOR_MAP[best_op])

                # Appliquer le meilleur opérateur pour continuer la trajectoire
                if best_op == 'recolor':
                    neighbor, move, _ = op_recolor(G, current, n_colors, tabu_list)
                elif best_op == 'swap':
                    neighbor, move, _ = op_swap(G, current, n_colors, tabu_list)
                else:
                    neighbor, move, _, _ = op_kempe_chain(
                        G, current, n_colors, tabu_list)

                if neighbor is None:
                    break

                new_obj = objective(G, neighbor)
                no_improve = 0 if new_obj < current_obj else no_improve + 1

                current = neighbor
                if move:
                    tabu_list.append(move)

                if count_conflicts(G, current) == 0:
                    break

    return np.array(X), np.array(y)


# ============================================================
# 2. Entraînement du modèle
# ============================================================

def train_model(X_raw, y_raw, train_graphs=None, n_colors=4):
    """
    Prépare les données, équilibre les classes et entraîne un Random Forest.

    Paramètres :
        X_raw        : features brutes (np.array)
        y_raw        : labels bruts (np.array)
        train_graphs : graphes supplémentaires si enrichissement nécessaire
        n_colors     : nombre de couleurs (pour l'éventuel enrichissement)

    Retourne :
        rf_model, scaler, df_features
    """
    df = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
    df['operateur'] = [OPERATOR_NAMES[l] for l in y_raw]

    # Diagnostic — enrichissement si classe absente ou trop rare
    classes_presentes  = set(df['operateur'].unique())
    classes_manquantes = [op for op in OPERATOR_NAMES if op not in classes_presentes]

    if (classes_manquantes or df['operateur'].value_counts().min() < 5) \
            and train_graphs is not None:
        print(f"⚠️  Classes absentes/rares : {classes_manquantes}")
        print("   → Enrichissement du dataset...")
        X_extra, y_extra = collect_training_data(
            train_graphs, n_colors=n_colors,
            n_runs_per_graph=8, max_iter=300)
        df_extra = pd.DataFrame(X_extra, columns=FEATURE_NAMES)
        df_extra['operateur'] = [OPERATOR_NAMES[l] for l in y_extra]
        df = pd.concat([df, df_extra], ignore_index=True)
        print("Distribution après enrichissement :")
        print(df['operateur'].value_counts())

    # Équilibrage des classes
    min_class   = max(df['operateur'].value_counts().min(), 10)
    df_balanced = pd.concat([
        resample(df[df['operateur'] == op],
                 replace=True, n_samples=min_class, random_state=SEED)
        for op in OPERATOR_NAMES
        if op in df['operateur'].values
    ])

    X_bal = df_balanced[FEATURE_NAMES].values
    y_bal = df_balanced['operateur'].map(OPERATOR_MAP).values

    print(f"\nÉchantillons après équilibrage : {len(X_bal)}")
    print("Classes dans y_bal :", sorted(set(y_bal)))

    # Split train / test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.25, random_state=SEED, stratify=y_bal)

    # Normalisation
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1
    )
    rf_model.fit(X_tr_s, y_tr)

    score_test = rf_model.score(X_te_s, y_te)
    cv_scores  = cross_val_score(rf_model, X_tr_s, y_tr, cv=5)

    print(f"\nAccuracy test : {score_test:.3f}")
    print(f"CV scores     : {cv_scores.round(3)} → mean={cv_scores.mean():.3f}")

    # Rapport de classification
    labels_presents          = sorted(set(y_te))
    target_names_filtered    = [OPERATOR_NAMES[i] for i in labels_presents]
    y_pred                   = rf_model.predict(X_te_s)

    print("\nRapport de classification :")
    print(classification_report(y_te, y_pred,
          labels=labels_presents,
          target_names=target_names_filtered))

    return rf_model, scaler, df


# ============================================================
# 3. Évaluation et visualisations
# ============================================================

def evaluate_model(rf_model, scaler, df_features):
    """
    Affiche :
      - la distribution des features par opérateur
      - la matrice de confusion
      - l'importance des features
    """
    X_bal   = df_features[FEATURE_NAMES].values
    y_bal   = df_features['operateur'].map(OPERATOR_MAP).values
    _, X_te, _, y_te = train_test_split(
        X_bal, y_bal, test_size=0.25, random_state=SEED, stratify=y_bal)
    X_te_s = scaler.transform(X_te)
    y_pred = rf_model.predict(X_te_s)

    # Distribution des features
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes       = axes.flatten()
    colors_op  = {'recolor': 'steelblue', 'swap': 'darkorange', 'kempe': 'green'}

    for i, feat in enumerate(FEATURE_NAMES):
        for op, col in colors_op.items():
            if op not in df_features['operateur'].values:
                continue
            subset = df_features[df_features['operateur'] == op][feat]
            axes[i].hist(subset, bins=30, alpha=0.5, color=col, label=op, density=True)
        axes[i].set_title(feat, fontsize=10)
        axes[i].legend(fontsize=8)

    plt.suptitle("Distribution des features par opérateur optimal", fontsize=13)
    plt.tight_layout()
    plt.show()

    # Matrice de confusion + importance des features
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    labels_presents = sorted(set(y_te))
    names_filtered  = [OPERATOR_NAMES[i] for i in labels_presents]
    cm              = confusion_matrix(y_te, y_pred, labels=labels_presents)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=names_filtered, yticklabels=names_filtered)
    ax1.set_title("Matrice de confusion", fontsize=12)
    ax1.set_xlabel("Prédit")
    ax1.set_ylabel("Réel")

    importances = rf_model.feature_importances_
    ax2.barh(FEATURE_NAMES, importances, color='steelblue')
    ax2.set_title("Importance des features (Random Forest)", fontsize=12)
    ax2.set_xlabel("Importance")

    plt.tight_layout()
    plt.show()
