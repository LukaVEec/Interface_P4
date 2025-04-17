import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyomo.environ import *
from itertools import combinations
import builtins
import src.distance as distance

def robustesse(model, G):
    H = nx.DiGraph()  
    for e in G.edges:
        if model.x[e].value == 1:
            p,q = e
            if p not in H.nodes:
                H.add_node(p)
            if q not in H.nodes:
                H.add_node(q) 
            H.add_edge(p,q)
    k_node_connectivity = nx.node_connectivity(H)
    k_edge_connectivity = nx.edge_connectivity(H)
    print(f"k_node_connectivity of the resulting graph: {k_node_connectivity}")
    print(f"k_edge_connectivity of the resulting graph: {k_edge_connectivity}")



def resolution(G, pairs_to_connect, edges, C):    
    
    # Création du modèle
    model = ConcreteModel()

    # Ici ce sont les variables binaires qui indiquent si on inclut l'arête dans notre réseau ou non
    model.x = Var(G.edges, within=Binary)
    
    # Variable qui représente la distance entre chaque paire d'aéroports qu'on veut lier
    model.d = Var(pairs_to_connect, within=NonNegativeReals)

    # Variable pour s'assurer qu'on inclut bien un chemin entre les paires qu'on veut lier
    # Une variable par noeud par paire à lier
    model.f = Var(pairs_to_connect, G.edges, within=NonNegativeReals)

    # Création de la fonction objectif : somme des distances des chemins entre paires + C * nbre d'arêtes
    model.obj = Objective(expr=builtins.sum(model.d[p] for p in pairs_to_connect)/len(pairs_to_connect) + C*builtins.sum(model.x[e] for e in G.edges),sense=minimize)
    
    # On ajoute une contrainte pour chaque paire qui dit que d >= somme des distances du chemin qu'on a choisi pour 
    # les lier
    for i in pairs_to_connect:
        model.add_component(f"shortest_path_{i}", Constraint(expr=model.d[i] == builtins.sum(edges[e] * model.f[i, e] for e in G.edges)))

    # Pour chaque paire on dit que la somme des chemins entrants = sortants sauf si on est au noeud dans la paire 
    # dans ce cas tu dois sortir plus que tu rentres et vice versa
    # Evidemment il y a pour chaque noeud une variable par paire
    for (p, q) in pairs_to_connect:
        for node in G.nodes:
            ini = builtins.sum(model.f[(p, q), i] for i in G.in_edges(node) if i in G.edges)
            out = builtins.sum(model.f[(p, q), i] for i in G.out_edges(node) if i in G.edges)
            if node == p:  
                model.add_component(f"source_{p}_{q}_{node}", Constraint(expr=out - ini == 1))
            elif node == q:  
                model.add_component(f"sink_{p}_{q}_{node}", Constraint(expr= ini - out == 1))
            else:  
                model.add_component(f"conservation_{p}_{q}_{node}", Constraint(expr=out - ini == 0))

    # Assure que une arête n'est utilisée uniquement si elle est prise en compte dans notre réseau
    for i in pairs_to_connect:
        for e in G.edges:
            model.add_component(f"activation_{i}_{e}", Constraint(expr=model.f[i, e] <= model.x[e]))

    # Contraintes pour assurer une k-node-connectivité

    # Contraintes pour assurer une k-arête-connectivité
    for (u, v) in G.edges:
        for (p, q) in pairs_to_connect:
            # Supprimer d'abord la variable si elle existe déjà
            if hasattr(model, 'g'):
                model.del_component(model.g)

            # Créer une nouvelle variable de flot pour simuler la suppression de l'arête (u, v)
            model.g = Var(G.edges, within=NonNegativeReals)

            # Flot sans passer par (u, v)
            for node in G.nodes:
                ini = builtins.sum(model.g[i] for i in G.in_edges(node) if i in G.edges and i != (u, v))
                out = builtins.sum(model.g[i] for i in G.out_edges(node) if i in G.edges and i != (u, v))

                if node == p:
                    model.add_component(f"backup_source_{p}_{q}_{u}_{v}_{node}", Constraint(expr=out - ini == 1))
                elif node == q:
                    model.add_component(f"backup_sink_{p}_{q}_{u}_{v}_{node}", Constraint(expr=ini - out == 1))
                else:
                    model.add_component(f"backup_conservation_{p}_{q}_{u}_{v}_{node}", Constraint(expr=out - ini == 0))

            # Assurer qu'un flot circule même sans (u, v) si (u, v) est utilisé
            model.add_component(f"edge_redundancy_{p}_{q}_{u}_{v}", 
                                Constraint(expr=model.x[(u, v)] <= builtins.sum(model.g[i] for i in G.edges)))
            

    

    # Contraintes pour assurer une k-arête-connectivité



    solver = SolverFactory('scip')  
    solver.solve(model, tee=False)

    robustesse(model, G)

    return model
