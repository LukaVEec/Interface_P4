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


def test_rob(G, edges, P, pairs, C, alpha, nmbr_prior, path_prior):
    J_prior = pairs[:nmbr_prior]

    _,best,best_scores = robustesse_prev.genetic_algorithm(P, pairs, C, edges, alpha, J_prior, path_prior)
    print("Meilleur coût avec l'algorithme génétique :", best_scores[-1])

    sample_size = 10
    pairs = np.array(pairs)
    random_indices = np.random.choice(len(pairs), size=sample_size, replace=True)
    samples = pairs[random_indices]
    G = nx.DiGraph([(start, end, {"weight": edges[(start, end)]}) for start, end in best])
    avg_node_conn = 0
    avg_edge_conn = 0
    for u, v in samples:
        try :
            avg_node_conn += nx.node_connectivity(G, u, v)
            avg_edge_conn += nx.edge_connectivity(G, u, v)
        except:
            pass
    avg_node_conn /= sample_size
    avg_edge_conn /= sample_size
    print("Estimation de la noeud-connexité moyenne :", avg_node_conn)
    print("Estimation de l'arête-connexité moyenne :", avg_edge_conn)

    for u, v in J_prior:
        try:
            num_paths = len(list(nx.node_disjoint_paths(G, u, v)))
            if num_paths < path_prior:
                print("La connexion (", u, ",", v, ")", " n'a pas pu être assurée plus qu'à ", num_paths, " chemins disjoints.", sep="")
        except:
            print("La connexion (", u, ",", v, ")", " n'a pas pu être assurée plus qu'à 0 chemins disjoints.", sep="")

    return G

if __name__ == "__main__":
    file = "files/airports.csv"
    route = "files/pre_existing_routes.csv"
    nmbr_pairs = 200
    G, edges = data_processing.data_processing(file, route)
    P = []
    for start, end in edges.keys():
        P.append((start, end))
    pairs = generate_unique_pairs(list(G.nodes()), nmbr_pairs, directed=True)
    test_rob(G, edges, P, pairs, 10, 15, 4, 10)