import src.data_processing as data_processing
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

def remove_random_nodes(G, number):

    nodes_to_remove = random.sample(list(G.nodes), number)
    for airport in nodes_to_remove:
        G.remove_node(airport)

    return G
 
def remove_closeness(G, treshold=15):
    closeness_centrality = nx.closeness_centrality(G)
    sorted_closeness = dict(sorted(closeness_centrality.items(), key=lambda item: item[1]))
    airports_to_remove = list(sorted_closeness.keys())[-treshold:]
    for airport in airports_to_remove:
        G.remove_node(airport)
    return G

def remove_closeness_treshold(G, threshold=0.65):
    closeness_centrality = nx.closeness_centrality(G)
    airports_to_remove = [node for node, centrality in closeness_centrality.items() if centrality > threshold]
    G.remove_nodes_from(airports_to_remove)
    
    return G, len(airports_to_remove)

def remove_betweenness(G,treshold=15):
    betweenness_centrality = nx.betweenness_centrality(G)
    sorted_betweenness = dict(sorted(betweenness_centrality.items(), key=lambda item: item[1]))
    airports_to_remove = list(sorted_betweenness.keys())[-treshold:]
    for airport in airports_to_remove:
        G.remove_node(airport)
    return G

def remove_betweenness_treshold(G, threshold=0.9):
    betweenness_centrality = nx.betweenness_centrality(G)
    airports_to_remove = [node for node, centrality in betweenness_centrality.items() if centrality > threshold]
    G.remove_nodes_from(airports_to_remove)
    
    return G, len(airports_to_remove)
def remove_degree(G, treshold=15):
    degree_centrality = nx.out_degree_centrality(G)
    sorted_degree = dict(sorted(degree_centrality.items(), key=lambda item: item[1]))
    airports_to_remove = list(sorted_degree.keys())[-treshold:]
    for airport in airports_to_remove:
        G.remove_node(airport)
    return G

def remove_degree_treshold(G, threshold=0.9):
    degree_centrality = nx.out_degree_centrality(G)
    airports_to_remove = [node for node, centrality in degree_centrality.items() if centrality > threshold]
    G.remove_nodes_from(airports_to_remove)
    
    return G, len(airports_to_remove)



### SIR model ###
def simulate_SIR(G, beta=0.01, gamma=0.05, fraction_infected=0.05, max_iterations=100):
    """
    Simulates the SIR model on a given graph and plots the results.

    Parameters:
        G (networkx.Graph): The graph representing the network.
        beta (float): Infection rate.
        gamma (float): Recovery rate.
        fraction_infected (float): Initial fraction of infected nodes.
        max_iterations (int): Maximum number of iterations for the simulation.

    Returns:
        extinction_time (int): The time step at which the disease is extinguished.
        num_infected_airports (int): The total number of infected airports.
    """

    model = ep.SIRModel(G)
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)
    cfg.add_model_parameter('gamma', gamma)
    cfg.add_model_parameter("fraction_infected", fraction_infected)
    model.set_initial_status(cfg)

    iterations = model.iteration_bunch(max_iterations)

    infected_nodes = set()
    propagation_time = 0
    max_infected = 0

    susceptible = []
    infected = []
    recovered = []

    for t, iteration in enumerate(iterations):
        susceptible.append(iteration['node_count'][0])  
        infected.append(iteration['node_count'][1])    
        recovered.append(iteration['node_count'][2])  

        
        for node, status in iteration['status'].items():
            if status == 1:  
                infected_nodes.add(node)


    num_infected_airports = len(infected_nodes)

#    Plot the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(susceptible, label="Susceptible", color="blue")
    # plt.plot(infected, label="Infected", color="red")
    # plt.plot(recovered, label="Recovered", color="green")
    # plt.xlabel("Time Steps")
    # plt.ylabel("Number of Nodes")
    # plt.title("SIR Model Simulation")
    # plt.legend()
    # plt.grid()
    # plt.savefig("sir_simulation_random_20.pdf")


    return propagation_time, num_infected_airports, max_infected


G, _ = data_processing.data_processing("files/airports.csv", "files/pre_existing_routes.csv")
# time, number, max =simulate_SIR(remove_random_nodes(G,20), beta=0.01, gamma=0.05, fraction_infected=0.05, max_iterations=100)


G_removed = remove_random_nodes(G, 20)

num_components = max(len(c) for c in nx.strongly_connected_components(G_removed)) 
print(f"Nombre de composantes connexes : {num_components}")

betas = [0.005, 0.01, 0.02]
gammas = [0.01, 0.05, 0.1]
n_simulations = 20

results = []

for beta in betas:
    for gamma in gammas:
        for _ in range(n_simulations):
            G, _ = data_processing.data_processing("files/airports.csv", "files/pre_existing_routes.csv")
            _, num_infected, _ = simulate_SIR(remove_betweenness(G, 20),beta=beta,gamma=gamma)
            results.append({
                'beta': beta,
                'gamma': gamma,
                'num_infected': num_infected
            })


df = pd.DataFrame(results)

# plt.figure(figsize=(12, 8))
# sns.boxplot(
#     data=df,
#     y="num_infected"
#     , x="beta",
#     hue="gamma",
#     palette="Set2",
# )
# plt.title(f"Variation de beta et gamma avec 20 aéroports retirés")
# plt.ylabel("Nombre total d’aéroports infectés")
# plt.xlabel("Taux d’infection")
# plt.legend(title="Taux de guérison")
# plt.grid()
# plt.tight_layout()
# plt.savefig("beta_gamma_between.pdf")
