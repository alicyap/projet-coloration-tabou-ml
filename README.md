# Optimisation de la Coloration de Cartes Géographiques
#### Recherche Tabou hybridée avec Machine Learning pour le choix des opérateurs de voisinage

> Projet Optimisation Combinatoire

---

## Description du projet

Ce projet explore l'apport du **Machine Learning** pour améliorer une **Recherche Tabou** appliquée au problème de **coloration de graphe** (cartes géographiques).

### Problème traité
Le problème de coloration de graphe consiste à assigner une couleur à chaque région d'une carte telle que deux régions voisines n'aient jamais la même couleur, en **minimisant le nombre de couleurs utilisées**.

**Fonction objectif :**
```
f(s) = α × conflits(s) + β × nb_couleurs(s)
```

### Approche hybride
À chaque itération de la Recherche Tabou, un classifieur **Random Forest** prédit quel opérateur de voisinage appliquer selon l'état courant de la recherche.

### Les 3 opérateurs de voisinage
| Opérateur | Description | Rôle |
|-----------|-------------|------|
| **Recolor** | Change la couleur d'un nœud en conflit | Intensification fine |
| **Swap** | Échange les couleurs de deux nœuds | Intensification locale |
| **Kempe Chain** | Inverse deux couleurs dans une composante connexe | Diversification / sortie de plateau |

---

## Structure du dépôt

```
projet-coloration-tabou-ml/
│
├── notebook/
│   └── projet_coloration_tabou_ml.ipynb   # Notebook principal
│
├── results/                                # Résultats expérimentaux
│   ├── figures/                            # Graphiques générés
│   └── tables/                             # Tableaux CSV
│
├── README.md                               # Ce fichier
└── requirements.txt                        # Dépendances Python
```

---

## Instructions d'exécution

### 1. Cloner le dépôt
```bash
git clone https://github.com/alicyap/projet-coloration-tabou-ml.git
cd projet-coloration-tabou-ml
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```
Sur linux
```bash
sudo apt update
sudo apt install python3-pip
sudo apt install pipx
```
Recommendé de créer un environment
```bash
python3 -m venv myenv
```
Activate the virtual environment:
On Linux/macOS:
```bash
source myenv/bin/activate
```
On Windows:
```bash
myenv\Scripts\activate
```
Et enfin installer les dependances
```bash
pip install -r requirements.txt
```
Désactivé env quand fini
(Recommendé pour revenir au path par defaut)
```bash
deactivate
```


### 3. Lancer le notebook
```bash
jupyter notebook notebook/projet_coloration_tabou_ml.ipynb
```

### 4. Exécuter toutes les cellules
Dans Jupyter : `Kernel > Restart & Run All`

> ⚠️ La collecte des données d'entraînement (section 5) peut prendre 2-5 minutes selon la machine.

---

## Résultats expérimentaux

Les expérimentations comparent 5 modes de sélection d'opérateur sur 4 instances de graphe :

| Mode | Description |
|------|-------------|
| `random` | Opérateur choisi aléatoirement (baseline) |
| `recolor` | Uniquement Recolor |
| `swap` | Uniquement Swap |
| `kempe` | Uniquement Kempe Chain |
| `ml` | Opérateur prédit par Random Forest (**notre approche**) |

Les métriques évaluées sont : nombre de conflits, nombre de couleurs utilisées, stabilité inter-runs et temps de calcul.

---

## Dépendances

- Python 3.8+
- networkx
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

---

## Auteurs

- **[Alim Samira ASTAGABDO AMADOU](https://github.com/Alim-Samira)** — Modélisation & opérateurs de voisinage
- **[Alicya-Pearl MARRAS](https://github.com/alicyap)** — Recherche Tabou & expérimentations  
- **[Lynda HAFIDI](https://github.com/Lynda-HAFIDI)**  — Machine Learning & analyse des résultats

M1 MIAGE MIXTE — Université Paris Nanterre — 2025/2026
