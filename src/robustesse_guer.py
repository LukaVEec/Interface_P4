import pandas as pd
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import data_processing as data_processing
import optimisation as optimisation


def evaluate(solution, J, C, P, kept, penalty=1e6):
    """
    Évaluation d'une solution avec Dijkstra par source
    avec une solution de départ à la quelle on a retiré 
    un certain nombre d'arête, pénalisation locale pour 
    les trajets impossibles.
    """
    G = nx.DiGraph([(start, end, {"weight": P[(start, end)]}) for start, end in solution + kept])
    G_kept = nx.DiGraph([(start, end, {"weight": P[(start, end)]}) for start, end in kept])
    penalty_cost = 0

    # Construction des chemins les plus courts par source, avec gestion d'échec
    shortest_paths = {}
    shortest_paths_kept = {}
    for src in {s for s, _ in J}:
        try:
            shortest_paths[src] = nx.single_source_dijkstra_path_length(G, src, weight='weight')
        except:
            shortest_paths[src] = {}  # Aucun chemin atteignable depuis cette source
        try:
            shortest_paths_kept[src] = nx.single_source_dijkstra_path_length(G_kept, src, weight='weight')
        except:
            shortest_paths_kept[src] = {}  # Aucun chemin atteignable depuis cette source

    # Évaluation cumulée
    total_distance = 0
    total_distance_kept = 0
    for src, dest in J:
        if dest in shortest_paths.get(src, {}):
            total_distance += shortest_paths[src][dest]
        else:
            penalty_cost += penalty  # pénalisation du trajet manquant
        if dest in shortest_paths_kept.get(src, {}):
            total_distance_kept += shortest_paths_kept[src][dest]


    return (total_distance - total_distance_kept) / len(J) + C * (len(solution)) + penalty_cost


def generate_initial_population(pop_size, P):
    return [random.sample(P, random.randint(0, len(P)-len(P)//4)) for i in range(pop_size)]


def crossover(parent1, parent2):
    """
    Croisement à deux points : prend un segment du parent2 et le réinsère dans parent1.
    """
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1.copy()

    idx1 = random.randint(0, min(len(parent1), len(parent2)) - 2)
    idx2 = random.randint(idx1 + 1, min(len(parent1), len(parent2)) - 1)

    segment = parent2[idx1:idx2]
    child = [e for e in parent1 if e not in segment]
    child[idx1:idx1] = segment
    return list(dict.fromkeys(child))  # Évite les doublons


def mutate(individual, P, mutation_rate=0.1):
    """
    Mutation : remplace plusieurs arêtes aléatoirement.
    """
    if random.random() < mutation_rate:
        n = max(1, len(individual) // 10)  # 10% des gènes
        for _ in range(n):
            if individual:
                individual.remove(random.choice(individual))
            remaining = list(set(P) - set(individual))
            if remaining:
                individual.append(random.choice(remaining))
    return individual


def tournament_selection(population, scores, k=5):
    selected = random.sample(list(zip(population, scores)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]


def genetic_algorithm(P, J, C, edges, kept, generations=100, pop_size=200):
    """
    Algorithme génétique pour optimiser le réseau de routes (individus = liste d'arêtes).
    """
    population = generate_initial_population(pop_size, P)
    best_scores = []
    J = np.array(J)

    for gen in range(generations):

        fitnesses_with_individuals = [
            (evaluate(ind, J, C, edges, kept), ind)
            for ind in population
        ]
        fitnesses_with_individuals.sort(key=lambda x: x[0])
        population = [ind for _, ind in fitnesses_with_individuals]
        scores = [fit for fit, _ in fitnesses_with_individuals]

        # Sélection des meilleurs
        elite_size = int(pop_size * 0.25)
        top = population[:elite_size]
        children = []

        # Génération des enfants
        while len(children) < pop_size - len(top):
            p1 = tournament_selection(population, scores)
            p2 = tournament_selection(population, scores)
            child = crossover(p1, p2)
            child = mutate(child, P, mutation_rate=0.1)
            children.append(child)

        population = top + children
        best_scores.append(scores[0])

    # Sélection finale du meilleur
    best = population[0]
    best_cost = scores[0]

    return best_cost, best, best_scores