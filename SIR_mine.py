import networkx as nx  # Importation de NetworkX pour la gestion des graphes
import numpy as np  # Importation de NumPy pour les calculs numériques
import matplotlib.pyplot as plt  # Importation de Matplotlib pour l'affichage des graphes
import random  # Importation du module random pour générer des nombres aléatoires
from src.data_processing import data_processing

# Définition des paramètres du modèle

G, edges = data_processing("files/airports.csv", "files/pre_existing_routes.csv" , "files/capacities_airports.csv")
# Probabilité d'infection lorsqu'un individu susceptible est en contact avec un infecté généré aléatoiremnt entre 0.25 - 0.35
beta = np.random.uniform(0.25, 0.35)
# Probabilité de guérison par unité de temps
gamma = 0.1 
initial_infected = 1 # Nombre initial d'individus infectés

#somme des capacités
sum_capacities = 0
for node in G.nodes:
    sum_capacities += G.nodes[node]["capacity"]

# Initialisation des états des nœuds (tous sont susceptibles au début)
states = {node: "S" for node in G.nodes()}

# Sélection aléatoire des nœuds initialement infectés
infected_nodes = random.sample(list(G.nodes()), initial_infected)
for node in infected_nodes:
    states[node] = "I"

# varaible pour stocker le nombre de noeuds infectés
init = 1

# Fonction pour mettre à jour les états des nœuds à chaque étape
def update_states(states, nodes_infected):
    new_states = states.copy()
    max_prob = 0.0
    Ni = nodes_infected
    infected_count = 0  # Compteur pour le nombre d'infectés

    for node in list(G.nodes()):
        if states[node] == "Blocked":
            continue
        beta_node = beta * G.nodes[node]["capacity"] / sum_capacities  # Probabilité d'infection pondérée par la capacité de l'aéroport
        if states[node] == "I":  # Si le nœud est infecté
            infected_count += 1  # Incrémente le compteur d'infectés
            if random.random() < gamma:  # Guérison avec probabilité gamma
                new_states[node] = "R"
            else:
                for neighbor in G.successors(node):  # Propagation vers les voisins sortants
                    if states[neighbor] == "S" and random.random() < beta_node * G.edges[(node, neighbor)]["weight"]:
                        new_states[neighbor] = "I"  # Infection du voisin
                        infected_count += 1  # Incrémente le compteur d'infectés

        # suppression du noeud avec la plus grande probabilité d'infection
        if beta_node > max_prob:
            max_prob = beta_node
            max_node = node
    new_states[max_node] = "Blocked"

    print(f"Nombre d'infectés à cette étape: {infected_count%len(G.nodes())}")  # Affiche le nombre d'infectés
    return new_states, infected_count%len(G.nodes())

# Simulation
steps = 150  # Nombre d'itérations
history = [states.copy()]  # Historique des états

print("nombre total de noeuds",len(G.nodes()))


while steps > 0 and init > 0:
    new_states,init = update_states(history[-1],init)

    history.append(new_states)
    steps -= 1

print(f"Nombre d'étapes: {len(history)}")  # Affiche le nombre d'étapes

"""
# Affichage de l'évolution de l'épidémie
fig, axes = plt.subplots(6, 5, figsize=(15, 15))  # 30 étapes affichées en grille
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= len(history):
        ax.axis("off")
        continue

    colors = {"S": "blue", "I": "red", "R": "green", "Blocked": "yellow"}
    node_colors = [colors[history[i][node]] for node in G.nodes()]
    
    ax.set_title(f"Step {i}")
    nx.draw(G, pos=nx.spring_layout(G, seed=42), node_color=node_colors, ax=ax, with_labels=False, node_size=100, arrows=True)

plt.tight_layout()
plt.show()

"""

