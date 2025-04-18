import pandas as pd
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import data_processing as data_processing
import optimisation as optimisation
import pygad
import genetique 
import pygad_library as pygad_library
import time
import robustesse_guer
 
 
def generate_unique_pairs(nodes, number_of_pairs, directed=True):
    seen = set()
    pairs = []

    while len(pairs) < number_of_pairs:
        a, b = random.sample(nodes, 2)
        pair = (a, b) if directed else tuple(sorted((a, b)))
        if pair not in seen:
            seen.add(pair)
            pairs.append((a, b))  # garder (a,b) même si non orienté pour compatibilité
    return pairs
 
 
def test_correcteness():
    file = "files/airports.csv"
    route = "files/pre_existing_routes.csv"
    G, edges = data_processing.data_processing(file, route)
    P = []
    for start, end in edges.keys():
        P.append((start, end))
    pairs = generate_unique_pairs(list(G.nodes()), 10, directed=True)
    C = 10
    _,best_bef,best_scores_bef = genetique.genetic_algorithm(P, pairs, C, edges)
    print("Meilleur coût avec l'algorithme génétique :", best_scores_bef[-1])
 
     # Enlever des arêtes faisant partie de la solution
    nmbr_edges_to_remove = 10
    idx_to_remove = np.random.choice(len(best_bef), size=nmbr_edges_to_remove, replace=False)
    edges_to_remove = [best_bef[i] for i in idx_to_remove]
    kept_temp = [x for x in best_bef if x not in edges_to_remove]
    for element in edges_to_remove:
        if element in P:
            P.remove(element)
    print("Arêtes enlevées :", edges_to_remove)
    
     # Enlever des noeuds faisant partie de la solution mais n'appartenant pas à J
    nmbr_nodes_to_remove = 3
    nodes_J = set()
    for u, v in pairs:
        nodes_J.add(u)
        nodes_J.add(v)
    nodes_used = set()
    for u, v in best_bef:
        if u not in nodes_J: nodes_used.add(u)
        if v not in nodes_J: nodes_used.add(v)
    nodes_to_remove = random.choices(list(nodes_used), k=nmbr_nodes_to_remove)
    kept = []
    for i in range(len(kept_temp)):
        u, v = kept_temp[i]
        if u not in nodes_to_remove and v not in nodes_to_remove:
            kept.append((u,v))
    for elem in P:
        u, v = elem
        if u in nodes_to_remove or v in nodes_to_remove:
            P.remove(elem)
    print("Noeuds enlevés :", nodes_to_remove)
 
    _,best_aft,best_scores_aft = robustesse_guer.genetic_algorithm(P, pairs, C, edges, kept)
    G = nx.DiGraph([(start, end, {"weight": edges[(start, end)]}) for start, end in best_aft])
 
    for (u,v) in pairs:
        if not nx.has_path(G, u, v): 
            print("Graphe déconnecté")
            continue
 
    print("Meilleur coût avec l'algorithme génétique guérisseur :", best_scores_aft[-1])
 
    plt.plot(best_scores_aft, label='Algorithme génétique')
    plt.title("Évolution du coût au fil des générations")
    plt.xlabel("Génération")
    plt.ylabel("Coût")
    plt.grid()
    plt.legend()
    plt.show()
 
test_correcteness()