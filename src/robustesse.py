import networkx as nx
import random
import numpy as np
import data_processing as data_processing
import optimisation as optimisation
import genetique 
import robustesse_prev
import robustesse_guer


def generate_unique_pairs(nodes, number_of_pairs, directed=True):
    seen = set()
    pairs = []

    while len(pairs) < number_of_pairs:
        a, b = random.sample(nodes, 2)
        pair = (a, b) if directed else tuple(sorted((a, b)))
        if pair not in seen:
            seen.add(pair)
            pairs.append((a, b))
    return pairs


def prevenir(pairs, C, edges, alpha, J_prior, path_prior):
    """
    Arguments :
    - pairs : liste de paires d'aéroports à relier
    - C : constante reflétant l'importance donnée à un réseau à peu de connexions entre aéroports
    - edges : ensemble des connexions entre aéroports possibles
    - alpha : constante reflétant l'importance donnée à un réseau robuste
    - J_prior, path_prior : ensemble des connexions J_prior entre aéroports à relier avec au minimum path_prior chemins distincts

    Retourne :
    - cout : coût de la fonction objectif
    """
    P = []
    for start, end in edges.keys():
        P.append((start, end))
    _,best,_ = robustesse_prev.genetic_algorithm(P, pairs, C, edges, alpha, J_prior, path_prior)

    # Calcul du coût réel
    G = nx.DiGraph([(start, end, {"weight": edges[(start, end)]}) for start, end in best])
    shortest_paths = {}
    for src in {s for s, _ in pairs}:
        try:
            shortest_paths[src] = nx.single_source_dijkstra_path_length(G, src, weight='weight')
        except:
            shortest_paths[src] = {}
    total_distance = 0
    for src, dest in pairs:
        if dest in shortest_paths.get(src, {}):
            total_distance += shortest_paths[src][dest]
    cout = total_distance / len(pairs) + C * len(best)

    # Vérification des contraintes
    for u, v in pairs:
        if not nx.has_path(G, u, v):
            print("Solution non connexe.")
            return -1

    for u, v in J_prior:
        num_paths = len(list(nx.node_disjoint_paths(G, u, v)))
        if num_paths < path_prior:
            print("La connexion (", u, ",", v, ")", " n'a pas pu être assurée plus qu'à ", num_paths, " chemins disjoints.", sep="")

    return cout

def guerir(pairs, C, edges, removed_edges, removed_nodes):
    """
    Arguments :
    - pairs : liste de paires d'aéroports à relier
    - C : constante reflétant l'importance donnée à un réseau à peu de connexions entre aéroports
    - edges : ensemble des connexions entre aéroports possibles
    - removed_edges : pourcentage des arêtes perturbées
    - removed_nodes : pourcentage des nodes retirés

    Retourne :
    - cout_bef : coût de la fonction objectif sans perturbation
    - cout_aft : coût supplémentaire de la fonction objectif après adaptation aux perturbations
    """
    P = []
    for start, end in edges.keys():
        P.append((start, end))
    cost_bef,best_bef,_ = genetique.genetic_algorithm(P, pairs, C, edges)
    G_bef = nx.DiGraph([(start, end, {"weight": edges[(start, end)]}) for start, end in best_bef])

    # Vérification de la connexité
    for u, v in pairs:
        if not nx.has_path(G_bef, u, v):
            print("Solution non connexe.")
            return cost_bef, -1

    # Suppression de certaines arêtes
    nmbr_edges_to_remove = int(removed_edges * len(best_bef))
    idx_to_remove = np.random.choice(len(best_bef), size=nmbr_edges_to_remove, replace=False)
    edges_to_remove = [best_bef[i] for i in idx_to_remove]
    kept_temp = [x for x in best_bef if x not in edges_to_remove]
    Pcopy = []
    for element in P:
        Pcopy.append(element)
    for element in edges_to_remove:
        if element in Pcopy:
            Pcopy.remove(element)

    # Suppression de certains noeuds
    nodes_J = set()
    for u, v in pairs:
        nodes_J.add(u)
        nodes_J.add(v)
    nodes_used = set()
    for u, v in best_bef:
        if u not in nodes_J: nodes_used.add(u)
        if v not in nodes_J: nodes_used.add(v)
    nodes_to_remove = random.choices(list(nodes_used), k=int(removed_nodes * len(nodes_used)))

    kept = []
    for i in range(len(kept_temp)):
        u, v = kept_temp[i]
        if u not in nodes_to_remove and v not in nodes_to_remove:
            kept.append((u,v))
    for elem in P:
        u, v = elem
        if u in nodes_to_remove or v in nodes_to_remove:
            P.remove(elem)

    cost_aft,best_aft,_ = robustesse_guer.genetic_algorithm(Pcopy, pairs, C, edges, kept)
    G_aft = nx.DiGraph([(start, end, {"weight": edges[(start, end)]}) for start, end in best_aft + kept])

    # Vérification de la connexité
    for u, v in pairs:
        try:
            if not nx.has_path(G_aft, u, v):
                print("Solution non connexe.")
                return cost_bef, -1
        except:
            print("Solution non connexe.")
            return cost_bef, -1

    return cost_bef, cost_aft



if __name__ == "__main__":
    file = "files/airports.csv"
    route = "files/pre_existing_routes.csv"
    nmbr_pairs = 200
    G, edges = data_processing.data_processing(file, route)
    pairs = generate_unique_pairs(list(G.nodes()), nmbr_pairs, directed=True)

    # Exemple prévenir
    print(prevenir(pairs, 500, edges, 50, pairs[:4], 5))

    # Exemple guérir
    nmbr_pairs = 30
    # pairs = generate_unique_pairs(list(G.nodes()), nmbr_pairs, directed=True)
    print(guerir(pairs, 500, edges, 0.1, 0.05))