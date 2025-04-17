import genetique
import data_processing
import pandas as pd


def new_network(airports, pre_existing_routes, wanted_journeys, C):
    """
    Fonction principale pour créer un nouveau réseau à partir des aéroports et des routes préexistantes.

    Args:
        airports (csv): Chemin vers le fichier contenant les données des aéroports.
        pre_existing_routes (csv): Chemin vers le fichier contenant les routes préexistantes.
        wanted_journeys (csv): Liste des trajets souhaités sous forme de tuples (départ, arrivée).
        C (float ou int): Coût de création d'une nouvelle route.

    Returns:
        tuple: Coût total optimisé et la liste des arêtes sélectionnées et crée un fichier new_routes.csv avec les arêtes retenues.
    """
    G, edges = data_processing.data_processing(airports, pre_existing_routes)
    P = []
    for start, end in edges.keys():
        P.append((start, end, edges[(start, end)]))

    wanted_routes = pd.read_csv(wanted_journeys)
    pairs = []
    for _, row in wanted_routes.iterrows():
        start, end = row["ID_start"], row["ID_end"]
        pairs.append((start, end))
    
    print("Paires à lier :", pairs)

    best_cost, selected_edges = genetique.genetic_algorithm(P, pairs, C, edges, generations=200, population_size=100)
    
    # Création d'un fichier new_routes.csv avec les arêtes retenues
    with open("files/new_routes.csv", "w") as f:
        f.write("start,end\n")
        for start, end in selected_edges:
            f.write(f"{start},{end}\n")
    
    return best_cost, selected_edges