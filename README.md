
# ✈️ Projet LEPL1507 — Groupe 2

Bienvenue dans notre projet de modélisation d’un **réseau aérien**, développé dans le cadre du cours **LEPL1507**.  
Notre solution s’appuie sur :



---

## 📁 Structure du projet

```
├── README.md
├── requirements.txt
├── SIR_mine.py
├── SIR_probability.py
├── files/
│   ├── airports.csv
│   ├── capacities_airports.csv
│   ├── capacities_connexions.csv
│   ├── pre_existing_routes.csv
│   ├── prices.csv
│   └── waiting_times.csv
└── src/
    ├── OBJ_A.py
    ├── benchmark.py
    ├── data_processing.py
    ├── distance.py
    ├── genetique.py
    ├── interface.py
    ├── new_network.py
    ├── objectif_B.py
    ├── optimisation.py
    ├── plot_network.py
    ├── pygad_library.py
    ├── robustesse_analyse.py
    ├── robustesse_guer.py
    ├── robustesse_prev.py
    ├── robustesse.py
    ├── test_opti.py
```

---

## Modules clés

### `genetique.py`
Contient toutes les fonctions nécessaires à l’**algorithme génétique** utilisé pour l'optimisation du réseau.

### `distance.py`
Fonction utilitaire pour calculer la distance entre deux points géographiques.

### `optimisation.py`
Permet de résoudre l’**objectif A** à l’aide de solveurs d’optimisation via **Pyomo**.

> ⚠️ Ce module n’est pas requis pour la version finale du projet utilisant `new_network.py`.

### `plot_network.py`
Affiche le réseau aérien sur une carte du monde avec `cartopy`.

**Arguments :**
- Graphe NetworkX
- Fichier `airports.csv`

### `interface.py`
Crée une **interface visuelle avec Streamlit** pour l’interaction avec le réseau.

**Nécessite :** `streamlit`, `cartopy`

### `data_processing.py`
Transforme les fichiers du projet en un **graphe NetworkX** utilisable.

**Utilise :**
- `airports.csv`
- `pre_existing_routes.csv`

### `pygad_library.py`
Version alternative de l’algorithme génétique avec la librairie `pygad`.

> ❌ **Non utilisée** dans la version finale.

### `OBJ_A.py`
Fichier contenant les codes pour l'analyse de l'objectif A. 

**Nécessite :** `pyomo`, solver `glpk``

### `objectif_b.py`

Contient les fichiers pour l'analyse de l'objectif B.

**Nécessite :** `ndlib``

### `Objectif C`

Robustesse_analyse, robustesse_guer et robustesse_prev contient les fichiers pour l'analyse de l'objectif C. 


---

## ✈️ Création d’un nouveau réseau aérien

Le fichier principal de la fontion est :
```python
src/new_network.py
```

### 🧰 Librairies nécessaires

- `numpy`
- `pandas`
- `networkx`

📦 Installez-les via :

```bash
pip install numpy pandas networkx
# ou
conda install numpy pandas networkx
```

### ▶️ Exécution d'un test

```bash
python main.py 
```

Il suffit de changer les arguments dans la fonction du fichier pour mettre vos paramètres.

---

## 🖥️ Interface de recommandation de vols

### ⚙️ Librairies nécessaires

- `pandas`
- `networkx`
- `matplotlib`
- `cartopy`
- `numpy`
- `streamlit`

💡 Installation :

```bash
pip install <library>
# ou
conda install <library>
```

### 🚀 Lancer l’interface Streamlit

```bash
streamlit run src/interface.py
```

Cela ouvrira automatiquement l’interface dans votre navigateur 🌐

> ⚠️ Vous devrez peut-être autoriser Streamlit à accéder à votre navigateur.

---

### 🌍 Démo en ligne (données de base)

**Vous pouvez directement cliquer sur ce lien pour tester l'interface avec les données fournies du projet sans devoir installer streamlit:**  
👉 [https://lepl1507g02.streamlit.app](https://lepl1507g02.streamlit.app)

---



