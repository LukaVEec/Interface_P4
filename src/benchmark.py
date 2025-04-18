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



def benchmark_pairs():
    C = 1000
    file = "files/airports.csv"
    route = "files/pre_existing_routes.csv"
    G, edges = data_processing.data_processing(file, route)
    P = []
    for start, end in edges.keys():
        P.append((start, end, edges[(start, end)]))

    for i in range(10, 100, 10):
        pairs = generate_unique_pairs(list(G.nodes()), i, directed=True)
        start = time.time()
        _, cost = pygad_library.new_network(file, route, pairs, C)
        time_taken = time.time() - start
        print(f"Nombre de paires : {i} - Coût total optimisé avec PyGAD : {cost} - Temps pris : {time_taken:.2f} secondes")

        start = time.time()
        best_cost, selected_edges = genetique.genetic_algorithm(P, pairs, C)
        time_taken = time.time() - start
        print(f"Nombre de paires : {i} - Meilleur coût avec l'algorithme génétique : {best_cost} - Temps pris : {time_taken:.2f} secondes")
        sum = 0
        for start, end in pairs:
            sum += nx.shortest_path_length(G, start, end, weight='weight')
        sum /= len(pairs)
        print("Coût total avant optimisation :", sum+len(selected_edges)*C)


    for i in range(100, 1001, 100):
        pairs = generate_unique_pairs(list(G.nodes()), i, directed=True)
        start = time.time()
        _, cost,_ = pygad_library.new_network(file, route, pairs, C)
        time_taken = time.time() - start
        print(f"Nombre de paires : {i} - Coût total optimisé avec PyGAD : {cost} - Temps pris : {time_taken:.2f} secondes")

        start = time.time()
        best_cost, selected_edges,_ = genetique.genetic_algorithm(P, pairs, C)
        time_taken = time.time() - start
        print(f"Nombre de paires : {i} - Meilleur coût avec l'algorithme génétique : {best_cost} - Temps pris : {time_taken:.2f} secondes")
        sum = 0
        for start, end in pairs:
            sum += nx.shortest_path_length(G, start, end, weight='weight')
        sum /= len(pairs)
        print("Coût total avant optimisation :", sum+len(selected_edges)*C)    


def benchmark_airports():
    C = 1000
    for i in range(500, 2001, 100):
        if i==2000:
            file = f"files/airports.csv"
            route = f"files/route/routes_{i}.csv"
            G, edges = data_processing.data_processing(file, route)
            P = []
            for start, end in edges.keys():
                P.append((start, end))
            pairs = generate_unique_pairs(list(G.nodes()), 200, directed=True)

            start = time.time()
            _, cost,_ = pygad_library.new_network(file, route, pairs, C)
            time_taken = time.time() - start
            print(f"Nombre d'aéroports : {i} - Coût total optimisé avec PyGAD : {cost} - Temps pris : {time_taken:.2f} secondes")
            start = time.time()
            best_cost, selected_edges,_ = genetique.genetic_algorithm(P, pairs, C, edges)
            time_taken = time.time() - start
            print(f"Nombre d'aéroports : {i} - Meilleur coût avec l'algorithme génétique : {best_cost} - Temps pris : {time_taken:.2f} secondes")
            sum = 0
            for start, end in pairs:
                sum += nx.shortest_path_length(G, start, end, weight='weight')
            sum /= len(pairs)
            print("Coût total avant optimisation :", sum+len(selected_edges)*C)
    
def plot_fitness():
    file = "files/airports.csv"
    route = "files/pre_existing_routes.csv"
    G, edges = data_processing.data_processing(file, route)
    P = []
    for start, end in edges.keys():
        P.append((start, end))
    pairs = generate_unique_pairs(list(G.nodes()), 100, directed=True)
    C = 1

    _,_,best_scores = genetique.genetic_algorithm(P, pairs, C, edges)
    _,_,best_scores2 = pygad_library.new_network(file, route, pairs, C)
    sum = 0
    for start, end in pairs:
        sum += nx.shortest_path_length(G, start, end, weight='weight')
    sum /= len(pairs)
    min_borne = [sum+len(pairs)*C for i in range(len(best_scores))]
    plt.plot(best_scores, label='Algorithme génétique')
    plt.plot(-np.array(best_scores2), label='PyGAD')
    #plt.plot(min_borne, label='Borne inférieure')
    plt.title("Évolution du coût au fil des générations")
    plt.xlabel("Génération")
    plt.ylabel("Coût")
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig("fitness.png")


def test_correcteness():
    file = "files/airports.csv"
    route = "files/pre_existing_routes.csv"
    G, edges = data_processing.data_processing(file, route)
    P = []
    for start, end in edges.keys():
        P.append((start, end))
    for o in range(10,101,10):
        number_of_pairs = o
        pairs = generate_unique_pairs(list(G.nodes()), number_of_pairs, directed=True)
        C = 3000
        start = time.time()
        solution,best,best_scores = genetique.genetic_algorithm(P, pairs, C, edges)
        time_taken = time.time() - start
        G_genetique = nx.DiGraph([(start, end, {"weight": edges[(start, end)]}) for start, end in best])
        #plot_network.plot_network(G_genetique, airports_csv=file)
        # Vérifie si il y a des arêtes inutiles dans la résolution du génétique
        useful_edges = set()
        for start, end in pairs:
            if nx.has_path(G_genetique, start, end):   
                path = nx.shortest_path(G_genetique, start, end, weight='weight')
                for i in range(len(path) - 1):
                    if(path[i], path[i+1] not in useful_edges):
                        useful_edges.add((path[i], path[i + 1]))
        print(f"Temps pris : {time_taken:.2f} secondes")
        print(f"Nombre d'arêtes utiles pour len(J) = {number_of_pairs} :", len(useful_edges))
        print(f"Nombre d'arêtes totales pour len(J) = {number_of_pairs} : {len(best)}")

        number_of_pairs = o*10
        pairs = generate_unique_pairs(list(G.nodes()), number_of_pairs, directed=True)
        C = 3000
        start = time.time()
        solution,best,best_scores = genetique.genetic_algorithm(P, pairs, C, edges)
        time_taken = time.time() - start
        G_genetique = nx.DiGraph([(start, end, {"weight": edges[(start, end)]}) for start, end in best])
        #plot_network.plot_network(G_genetique, airports_csv=file)
        # Vérifie si il y a des arêtes inutiles dans la résolution du génétique
        useful_edges = set()
        for start, end in pairs:
            if nx.has_path(G_genetique, start, end):   
                path = nx.shortest_path(G_genetique, start, end, weight='weight')
                for i in range(len(path) - 1):
                    if(path[i], path[i+1] not in useful_edges):
                        useful_edges.add((path[i], path[i + 1]))
        print(f"Temps pris : {time_taken:.2f} secondes")
        print(f"Nombre d'arêtes utiles pour len(J) = {number_of_pairs} :", len(useful_edges))
        print(f"Nombre d'arêtes totales pour len(J) = {number_of_pairs} : {len(best)}")
        """
        plt.plot(best_scores, label='Algorithme génétique')
        plt.plot(plot_best, label='Optimisation solver')
        plt.plot(-np.array(best_scores2), label='PyGAD')
        #plt.plot(min_borne, label='Borne inférieure')
        plt.title("Évolution du coût au fil des générations")
        plt.xlabel("Génération")
        plt.ylabel("Coût")
        plt.grid()
        plt.legend()
        plt.show()
        """

test_correcteness()