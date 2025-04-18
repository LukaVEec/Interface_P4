import pandas as pd
import networkx as nx
from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
from  distance import distance


############    STEP 1 : Create the DIRECTED GRAPH


def data_processing(file1,file2, route):
    t_airports = pd.read_csv(file1)
    capacities_airports = pd.read_csv(file2)
    airports = pd.merge(t_airports, capacities_airports, left_on='ID', right_on='airportsID', how='left')
    routes = pd.read_csv(route)
    
    
    G = nx.DiGraph()
    for _, row in airports.iterrows():
        G.add_node(row["ID"], name=row["name"], capacity= row["capacity"], latitude=row["latitude"], longitude=row["longitude"])
        

        edges = {}
        for _, row in routes.iterrows():
            start, end = row["ID_start"], row["ID_end"]
            if start in G.nodes and end in G.nodes:
                lat1 = G.nodes[start]["latitude"]
                lon1 = G.nodes[start]["longitude"]
                lat2 = G.nodes[end]["latitude"]
                lon2 = G.nodes[end]["longitude"]
            
                dist = distance(lat1, lat2, lon1,lon2)  
            
                G.add_edge(start, end, distance=dist,capacity=row["connexion capacity"])      
                edges[(start, end)] = dist  
            
    return G


############    STEP 2 : Create the model 

def distance_totale(model):
    return sum(model.x[i, j].value * model.d[i, j] for (i, j) in model.E)

def pyomo_model(G, source, puits, Flux):
    
    
    model = ConcreteModel()

    model.V = Set(initialize=list(G.nodes))
    model.E = Set(initialize=list(G.edges), dimen=2)

    def get_capacity(model, i, j):
        return G.edges[i, j]['capacity']

    def get_distance(model, i, j):
        return G.edges[i, j]['distance']
    
    def obj_function(model):
        return sum(model.x[i, j] * model.d[i, j] for (i, j) in model.E)

    model.c = Param(model.E, initialize=get_capacity)
    model.d = Param(model.E, initialize=get_distance)

    model.x = Var(model.E, domain=NonNegativeReals)
    #model.y = Var(model.E, domain=Binary)


    model.obj = Objective(rule=obj_function, sense=minimize)

    def flow_condition(model, i):
        if i == source:
            return sum(model.x[i, j] for j in model.V if (i, j) in model.E) - sum(model.x[j, i] for j in model.V if (j, i) in model.E) == Flux
        elif i == puits:
            return sum(model.x[j, i] for j in model.V if (j, i) in model.E) - sum(model.x[i, j] for j in model.V if (i, j) in model.E) == Flux
        else:
            return sum(model.x[j, i] for j in model.V if (j, i) in model.E) - sum(model.x[i, j] for j in model.V if (i, j) in model.E) == 0

    model.flow_conservation = Constraint(model.V, rule= flow_condition)

    def capacity_constraint(model, i, j):
        return model.x[i, j] <= model.c[i, j] #* model.y[i, j]
    
    model.cap_constraint = Constraint(model.E, rule=capacity_constraint)

    return model



############    STEP 3 : Create the function to solve OBJ A 


def OBJ_A(G, A_d, A_a, F):
    
    
    # 1. Déterminer le flux maximum à l'aide algo Ford-Fulkerson
    
    flux_max, flow_dict = nx.maximum_flow(G, A_d, A_a, capacity='capacity')
    
    print(f"Flux maximum possible entre {A_d} et {A_a} : {flux_max} passagers")
    
    modeles = []
    # 2. Cas 1: Flux <= flux_max
    
    
    if F <= flux_max:
       
        model = pyomo_model(G, A_d, A_a, F)
        solver = SolverFactory('glpk')
        solver.solve(model)
        modeles.append(model)

        print("\n Aéroports visités et évolution du flux :")
        for (i, j) in model.E:
            if model.x[i, j].value > 0:
                print(f"{i} → {j} : {model.x[i, j].value:.2f} passagers")

        distance_tot = distance_totale(model)
        dist_moy = distance_tot / F
        print(f" La distance moyenne parcourue par passager : {dist_moy:.2f} km")
        return



    # 3. Cas 2 : diviser le flux en m morceaux avec Fi<= flux_max
    else:
        
        m = F // flux_max  
        reste = F % flux_max  
       
        print(f"\n Le flux F = {F} est trop grand pour être acheminé en une fois.")
        print(f"==> Découpage en {m} morceaux de {flux_max} passagers")
        if reste > 0:
            print(f" + un sous-flux supplémentaire de {reste} passagers\n")


    # Si le flux total est divisible par flux_max, on résout une seule fois 
        if m > 0:
            print(f"Résolution pour {m} sous-flux de {flux_max} passagers :")
            model = pyomo_model(G, A_d, A_a, flux_max)
            solver = SolverFactory('glpk')
            solver.solve(model)
            modeles.append(model)

            for (i_, j_) in model.E:
                if model.x[i_, j_].value > 0:
                    print(f"{i_} → {j_} : {model.x[i_, j_].value:.2f}")
        
            avg = distance_totale(model) / flux_max
            print(f"Distance moyenne parcourue par passager : {avg:.2f} km\n")
    
    # S'il y a reste
        if reste > 0:
            print(f"Résolution pour le sous-flux restant de {reste} passagers :")
            model = pyomo_model(G, A_d, A_a, reste)
            solver.solve(model)
            modeles.append(model)
        
            print("Flux restant :")
            for (i_, j_) in model.E:
                if model.x[i_, j_].value > 0:
                    print(f"{i_} → {j_} : {model.x[i_, j_].value:.2f}")
        
            avg = distance_totale(model) / reste
            print(f"Distance moyenne (reste) : {avg:.2f} km")
            
    return modeles
            
            
 ############    STEP 4 : Apply the function to the data + visualisation
 

import plotly.graph_objects as go

def tracer_flux_sur_carte(G, models, airports_csv):
    
    df_airports = pd.read_csv(airports_csv)
    coords = {row['ID']: (row['longitude'], row['latitude']) for _, row in df_airports.iterrows()}


    fig = go.Figure()

    #  Afficher les aéroports
    for code, (lon, lat) in coords.items():
        fig.add_trace(go.Scattergeo(
            lon=[lon],
            lat=[lat],
            text=code,
            mode='markers+text',
            textposition="top center",
            marker=dict(size=4, color="blue"),
            name=code
        ))

    # Afficher les  flux
    for model in models:
        for (i, j) in model.E:
            v = model.x[i, j].value
            if v and v > 0:
                lon_i, lat_i = coords[i]
                lon_j, lat_j = coords[j]
                fig.add_trace(go.Scattergeo(
                    lon=[lon_i, lon_j],
                    lat=[lat_i, lat_j],
                    mode='lines',
                    line=dict(width=0.5 + v / 5000, color='red'),
                    opacity=0.7,
                    name=f"{i} → {j} ({v:.0f})"
                ))

    fig.update_layout(
        title="Évolution du flux",
        geo=dict(
            scope='world',
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showcountries=True,
        )
    )

    fig.show()

       

G = data_processing('files/airports.csv', 'files/capacities_airports.csv', 'files/capacities_connexions.csv')
modeles =OBJ_A(G, A_d="FRA", A_a="ALG", F=56000)
tracer_flux_sur_carte(G, modeles, 'files/airports.csv')
