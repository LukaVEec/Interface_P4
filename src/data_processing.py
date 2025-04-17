import pandas as pd
import networkx as nx
from distance import distance


def data_processing(file, route):
    airports = pd.read_csv(file)
    routes = pd.read_csv(route)

    G = nx.DiGraph()

    for _, row in airports.iterrows():
        G.add_node(row["ID"], name=row["name"], city=row["city"], country=row["country"], latitude=row["latitude"], longitude=row["longitude"])
        

    edges = {}
    for _, row in routes.iterrows():
        start, end = row["ID_start"], row["ID_end"]
        
        if start in G.nodes and end in G.nodes:
            lat1 = G.nodes[start]["latitude"]
            lon1 = G.nodes[start]["longitude"]
            lat2 = G.nodes[end]["latitude"]
            lon2 = G.nodes[end]["longitude"]
            dist = distance(lat1, lat2, lon1,lon2)  
            G.add_edge(start, end, weight=dist)
            edges[(start, end)] = dist  
    return G, edges