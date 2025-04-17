import pandas as pd
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import data_processing as data_processing
import optimisation as optimisation
import pygad
import genetique 



def build_graph(solution, P):
    G = nx.DiGraph()
    for i, bit in enumerate(solution):
        if bit == 1:
            u, v, w = P[i]
            G.add_edge(u, v, weight=w)
    return G


def new_network(airports, pre_existing_routes, J, C):
    G, P_edges = data_processing.data_processing(airports, pre_existing_routes)
    P = []
    for start, end in P_edges.keys():
        P.append((start, end, P_edges[(start, end)]))
    
    penalty = 10**6

    def fitness_func(ga_instance, solution, solution_idx):
        G = build_graph(solution, P)

            # Construction des chemins les plus courts par source, avec gestion d'échec
        shortest_paths = {}
        for src in {s for s, _ in J}:
            try:
                shortest_paths[src] = nx.single_source_dijkstra_path_length(G, src, weight='weight')
            except:
                shortest_paths[src] = {}  # Aucun chemin atteignable depuis cette source

        # Évaluation cumulée
        total_distance = 0
        for src, dest in J:
            if dest in shortest_paths.get(src, {}):
                total_distance += shortest_paths[src][dest]
            else:
                total_distance += penalty  # pénalisation du trajet manquant

        return -total_distance / len(J) - C * len(solution)

        
    num_genes = len(P)
    ga_instance = pygad.GA(
        num_generations=200,
        num_parents_mating=40,
        fitness_func=fitness_func,
        sol_per_pop=100,
        num_genes=num_genes,
        gene_type=int,
        gene_space=[0, 1],
        mutation_type="random",
        mutation_probability=0.1,
        crossover_type="uniform",
        stop_criteria=None,
        keep_elitism=5
    )
    ga_instance.run()
    #ga_instance.plot_fitness()

    solution, solution_fitness, _ = ga_instance.best_solution()
    selected_edges = [P[i] for i in range(len(solution)) if solution[i] == 1]
    return selected_edges, -solution_fitness, ga_instance.best_solutions_fitness








