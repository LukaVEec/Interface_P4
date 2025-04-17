from Recommandations import recommandations_interface
import networkx as nx
import src.data_processing as data_processing
import src.optimisation as optimisation
import pandas as pd
import src.distance as distance

G, edges = data_processing.data_processing("files/airports.csv", "files/pre_existing_routes.csv")


attente = pd.read_csv("files/waiting_times.csv")
prix = pd.read_csv("files/prices.csv")

prices = dict(zip(zip(prix["ID_start"], prix["ID_end"]), prix["price_tag"]))
waiting_times = dict(zip(attente["ID"], attente["idle_time"]))


# Test the recommandations function
recommandations_interface(G, waiting_times, prices,[('BKK','ADD')])