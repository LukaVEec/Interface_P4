import src.optimisation as optimisation 
import data_processing as data_processing
import distance as distance
import unittest
import networkx as nx
import csv
import pandas as pd
from src.distance import distance
import time
import pandas as pd
import random

# Charger les données des aéroports
airports_df = pd.read_csv("files/airports.csv")  # Assure-toi que le chemin du fichier est correct

# Extraire les identifiants des aéroports
airport_ids = airports_df["ID"].tolist()

# Vérifier qu'il y a suffisamment d'aéroports pour générer des paires uniques
num_pairs = 200
if len(airport_ids) < 2:
    raise ValueError("Pas assez d'aéroports pour générer des paires uniques.")

# Générer 200 paires uniques d'aéroports sans doublons
pairs = set()
while len(pairs) < num_pairs:
    a1, a2 = random.sample(airport_ids, 2)  # Sélectionner deux aéroports différents
    pairs.add((a1, a2))
pairs_list = list(pairs)



G, edges = data_processing.data_processing("files/airports.csv", "files/pre_existing_routes.csv")

G_small, filtered_routes = data_processing.data_processing("files/test_airports.csv", "files/pre_existing_routes.csv")

def test_network():
    G_test = nx.DiGraph()
    edges = {}
    G_test.add_node("ABC")
    G_test.add_node("DEF")
    G_test.add_node("GHI")
    G_test.add_node("JKL")
    G_test.add_node("MNO")
    G_test.add_node("PQR")
    G_test.add_node("STU")
    G_test.add_node("VWX")
    G_test.add_edge("ABC", "DEF", weight=50000)
    edges[("ABC", "DEF")] = 50000
    G_test.add_edge("VWX","ABC",weight=150000)
    edges[("VWX","ABC")] = 150000
    G_test.add_edge("GHI","ABC", weight=50000)
    edges[("GHI","ABC")] = 50000
    G_test.add_edge("DEF","STU", weight=200000)
    edges[("DEF","STU")] = 200000
    G_test.add_edge("JKL","DEF" ,weight=40000)
    edges[("JKL","DEF")] = 40000
    G_test.add_edge("DEF","MNO", weight=400000)
    edges[("DEF","MNO")] = 400000
    G_test.add_edge("DEF","GHI", weight=20000)
    edges[("DEF","GHI")] = 20000
    G_test.add_edge("VWX","GHI", weight=100000)
    edges[("VWX","GHI")] = 100000
    G_test.add_edge("MNO","JKL", weight=30000)
    edges[("MNO","JKL")] = 30000
    G_test.add_edge("JKL", "PQR",weight=50000)
    edges[("JKL", "PQR")] = 50000
    G_test.add_edge("PQR","MNO", weight=100000)
    edges[("PQR","MNO")] = 100000
    G_test.add_edge("STU","PQR", weight=50000)
    edges[("STU","PQR")] = 50000
    G_test.add_edge("STU","VWX", weight=50000)
    edges[("STU","VWX")] = 50000
    G_test.add_edge("VWX","GHI", weight=100000)
    edges[("VWX","GHI")] = 100000

    return G_test, edges

G_test, edges_test = test_network()

def test_shortest():
    pairs_to_connect = [('LOS', 'BOS')]
    model = optimisation.resolution(G, pairs_to_connect, edges, 0)
    return model
cmax = 10**18
def test_shortest_cbig():
    pairs_to_connect = [('ALG', 'DXB')]
    edges[('ALG', 'DXB')] = 50000000000
    edges[('DXB', 'ALG')] = 50000000000
    model = optimisation.resolution(G, pairs_to_connect, edges,cmax)
    return model.obj()

def test_direct():
    pairs_to_connect = [('BKK','ADD')]

    model = optimisation.resolution(G, pairs_to_connect, edges, 1)
    
    return model

def test_not_direct(C=0):
    current = []
    pairs_to_connect = [('LGW','ADD'), ('LGW','BOS'), ('LGW','LOS'), ('LGW','BKK'), ('BKK','LOS'), ('BKK','BOS'), ('BKK','ADD'), ('BOS','LOS'), ('BOS','ADD'), ('LOS','ADD')]
    model,current = optimisation.resolution(G, pairs_to_connect, edges, C,current, len(pairs_to_connect))
    return model.obj()

def test_complex():
    pairs_to_connect = [('LOS', 'BOS'), ('BKK','ADD'), ('LGW','ADD'), ('BKK','LOS'), ('LGW','BOS'), ('LGW','LOS'), ('BKK','BOS'), ('BKK','LGW'), ('LOS','ADD'), ('BOS','ADD')]
    model = optimisation.resolution(G, pairs_to_connect, edges, 0)    
    return model.obj()

def test_complex_c5():
    pairs_to_connect = [('YYZ','BKK'),('LHR','BKK')]
    model = optimisation.resolution(G_small,pairs_to_connect,filtered_routes,5000000)
    return model.obj()




def test_bcp_bcp(C=0):
    # Vérifier qu'il y a suffisamment d'aéroports pour générer des paires uniques
    num_pairs = 200
    if len(airport_ids) < 2:
        raise ValueError("Pas assez d'aéroports pour générer des paires uniques.")

    # Générer 200 paires uniques d'aéroports sans doublons
    pairs = set()
    while len(pairs) < num_pairs:
        a1, a2 = random.sample(airport_ids, 2)  # Sélectionner deux aéroports différents
        pairs.add((a1, a2))
    pairs_list = list(pairs)
    model = optimisation.resolution(G, pairs_list, edges, C)
    sum = 0
    for i,j in pairs_list:
        sum += nx.shortest_path_length(G, source=i, target=j,weight='weight')
    
    return model.obj(), sum/len(pairs_list)


def test_bcp_bcp_bcp(C=0):
    # Vérifier qu'il y a suffisamment d'aéroports pour générer des paires uniques
    num_pairs = 2000
    if len(airport_ids) < 2:
        raise ValueError("Pas assez d'aéroports pour générer des paires uniques.")

    # Générer 200 paires uniques d'aéroports sans doublons
    pairs = set()
    while len(pairs) < num_pairs:
        a1, a2 = random.sample(airport_ids, 2)  # Sélectionner deux aéroports différents
        pairs.add((a1, a2))
    pairs_list = list(pairs)
    print("Pairs list generated")
    model = optimisation.resolution(G, pairs_list, edges, C)
    sum = 0
    #for i,j in pairs_list:
    #    sum += nx.shortest_path_length(G, source=i, target=j,weight='weight')
    
    return model.obj()

def test_optimality():
    model = optimisation.resolution(G_test, [("DEF", "MNO")], edges_test, 50000)
    return model.obj()

class TestResolution(unittest.TestCase):
    def test_correcteness(self):
        start = time.time()
        self.assertAlmostEqual(test_shortest().obj(), nx.shortest_path_length(G, source="LOS", target="BOS",weight='weight'))
        print(f"\nTest with C = 0 and 1 pair in {round(time.time()-start, 4)} seconds")  

        start = time.time()
        self.assertAlmostEqual(round(test_direct().obj(),4), round(nx.shortest_path_length(G, source="BKK", target="ADD",weight='weight'),4)+1)
        print(f"Test with C = 1 and 1 pair in {round(time.time()-start, 4)} seconds")
        G_solution = optimisation.build_graph_from_solution(test_direct(), edges, G)     


        start = time.time()
        # self.kassertAlmostEqual(test_not_direct(1000), nx.shortest_path_length(G, source="LGW", target="ADD",weight='weight'))
        result = test_not_direct(1000)
        print(f"Test with C = 1000 and 10 pair in {round(time.time()-start, 4)} seconds")

        start = time.time()
        self.assertAlmostEqual(test_shortest_cbig(), 50000000000+10**18)
        print(f"Test with C = cmax and 1 pair in {round(time.time()-start, 4)} seconds")

        start = time.time()
        #self.assertAlmostEqual(test_complex_c5(), 5000000*2+9855600.810234353/1000+14115511.52441033/2000)
        print(f"Test with C = 5000000 and 2 pairs on a small graph in {round(time.time()-start, 4)} seconds")

        start = time.time()       
        self.assertEqual(test_optimality(), 450000)
        print(f"Test with C = 50000 and 1 pair in {round(time.time()-start, 4)} seconds")





        # start = time.time()
        # result2, sum2 = test_bcp_bcp(C=1000)
        #self.assertAlmostEqual(result2, sum2)
        #print(f"Test with C = 0 and 200 pairs in {round(time.time()-start, 4)} seconds")
        
        # start = time.time()
        # result2 = test_bcp_bcp_bcp(C=0)
        #self.assertAlmostEqual(result2, sum2)
        # print(f"Test with C = 0 and 500 pairs in {round(time.time()-start, 4)} seconds")
        







if __name__ == '__main__':
    unittest.main(verbosity=2)

