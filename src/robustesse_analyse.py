import pandas as pd
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import data_processing as data_processing
import optimisation as optimisation
import pygad
import genetique 
import test_pygad
import time
import robustesse_prev
import test_rob_prev


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


# Analyse du nombre de paires non déservies après perturbation pour différents alpha.
# La perturbation consiste en la perte de 5% des arêtes du réseau. C = 10.
##### Comparer aussi avec les différences de coûts.
def prev_alpha():
    file = "files/airports.csv"
    route = "files/pre_existing_routes.csv"
    nmbr_pairs = 200
    G, edges = data_processing.data_processing(file, route)
    P = []
    for start, end in edges.keys():
        P.append((start, end))
    pairs = generate_unique_pairs(list(G.nodes()), nmbr_pairs, directed=True)
    # test_rob(G, edges, P, pairs, 10, 15, 4, 10)

    # alphas = [0, 1, 5, 10, 25, 50]
    alphas = [10]
    results = []
    for alpha in alphas:
        print("hey")
        G_rob = test_rob_prev.test_rob(G, edges, P, pairs, 10, alpha, 0, 0)
        print("hey")
        _,best,_ = genetique.genetic_algorithm(P, pairs, 10, edges)
        print("hey")
        G_opt = nx.DiGraph([(start, end, {"weight": edges[(start, end)]}) for start, end in best])

        # Couts
        # Enlever des arêtes et voir cmb de paires sont encore reliées en pourcentage
prev_alpha()
# Faire la même pour les noeuds ?

# Analyse pour les chemins prioritaires
# Faire la même pour les noeuds ?

# Coût différent entre prev et guer (en considérant que certaines paires ne sont plus atteintes ducp)
# Faire la même pour les noeuds ?