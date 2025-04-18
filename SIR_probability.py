import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from data_processing import data_processing

# Chargement des données du réseau aérien
G, edges = data_processing("files/airports_extended.csv",
                           "files/pre_existing_routes_extended.csv",
                           "files/capacities_airports_extended.csv")


def update_states(states, current_step, G, beta, gamma, sum_capacities, blocked_nodes_info):
    """Mise à jour des états des nœuds pour une étape de simulation"""
    new_states = states.copy()
    max_prob = 0.0
    infected_count = 0
    already_checked = []
    max_node = None

    for node in G.nodes():
        if states[node] == "Blocked":
            continue

        # Calcul du risque nodal
        beta_node = beta * G.nodes[node]["capacity"] / sum_capacities

        if states[node] == "I":
            if node not in already_checked:
                already_checked.append(node)
                infected_count += 1
                # Guérison
                if random.random() < gamma:
                    new_states[node] = "R"
                else:
                    # Propagation aux voisins
                    for neighbor in G.successors(node):
                        if states[neighbor] == "S":
                            infection_prob = beta_node * G.edges[(node, neighbor)].get("weight", 1.0)
                            if random.random() < infection_prob:
                                new_states[neighbor] = "I"

        # Sélection du nœud à bloquer
        if beta_node > max_prob:
            max_prob = beta_node
            max_node = node

    # Blocage du nœud sélectionné
    if max_node is not None:
        new_states[max_node] = "Blocked"
        blocked_nodes_info.append((max_node, current_step, max_prob))

    return new_states, infected_count


def run_single_simulation(G, beta_range=(0.25, 0.35), gamma=0.1, max_steps=1000, initial_infected=1):
    """Exécute une simulation complète SIR avec blocage"""
    beta = np.random.uniform(*beta_range)
    sum_capacities = sum(G.nodes[node]["capacity"] for node in G.nodes)

    # Initialisation
    states = {node: "S" for node in G.nodes()}
    infected_nodes = random.sample(list(G.nodes()), initial_infected)
    for node in infected_nodes:
        states[node] = "I"

    # Stockage des résultats
    results = {
        'infected': np.zeros(max_steps),
        'recovered': np.zeros(max_steps),
        'susceptible': np.zeros(max_steps),
        'blocked': np.zeros(max_steps),
        'blocked_nodes': []
    }

    current_states = states.copy()
    for step in range(max_steps):
        current_states, infected_count = update_states(
            current_states, step, G, beta, gamma, sum_capacities, results['blocked_nodes'])

        # Enregistrement des statistiques
        results['infected'][step] = list(current_states.values()).count("I")
        results['recovered'][step] = list(current_states.values()).count("R")
        results['susceptible'][step] = list(current_states.values()).count("S")
        results['blocked'][step] = list(current_states.values()).count("Blocked")

        if infected_count == 0:  # Arrêt si plus d'infectés
            break

    # Tronquer les résultats à la durée réelle de la simulation
    actual_steps = step + 1
    for key in ['infected', 'recovered', 'susceptible', 'blocked']:
        results[key] = results[key][:actual_steps]

    return results


def monte_carlo_simulation(G, n_simulations=100,max_steps=1000):
    """Exécute l'analyse Monte Carlo"""
    results = {
        'all_infected': [],
        'all_recovered': [],
        'all_blocked': [],
        'peak_infections': [],
        'time_to_peak': [],
        'duration': [],
        'blocked_nodes': []
    }

    for _ in tqdm(range(n_simulations), desc="Monte Carlo Simulations"):
        sim_result = run_single_simulation(G, max_steps=max_steps)

        # Stockage des séries temporelles
        results['all_infected'].append(sim_result['infected'])
        results['all_recovered'].append(sim_result['recovered'])
        results['all_blocked'].append(sim_result['blocked'])
        results['blocked_nodes'].append(sim_result['blocked_nodes'])

        # Calcul des métriques
        results['peak_infections'].append(np.max(sim_result['infected']))
        results['time_to_peak'].append(np.argmax(sim_result['infected']))
        results['duration'].append(len(sim_result['infected']))

    return results


def analyze_results(mc_results):
    """Analyse statistique des résultats Monte Carlo"""
    analysis = {}

    # Trouver la longueur maximale des simulations
    max_len = max(len(sim) for sim in mc_results['all_infected'])

    # Initialisation des tableaux de résultats
    for metric in ['infected', 'recovered', 'blocked']:
        padded = [np.pad(sim, (0, max_len - len(sim)), 'constant', constant_values=np.nan)
                  for sim in mc_results[f'all_{metric}']]
        analysis[f'{metric}_mean'] = np.nanmean(padded, axis=0)
        analysis[f'{metric}_lower'] = np.nanpercentile(padded, 2.5, axis=0)
        analysis[f'{metric}_upper'] = np.nanpercentile(padded, 97.5, axis=0)

    # Statistiques sur les pics
    analysis['peak_stats'] = {
        'mean': np.mean(mc_results['peak_infections']),
        'std': np.std(mc_results['peak_infections']),
        'ci': (np.percentile(mc_results['peak_infections'], 2.5),
               np.percentile(mc_results['peak_infections'], 97.5))
    }

    # Statistiques sur les nœuds bloqués
    all_blocked = [node for sublist in mc_results['blocked_nodes'] for node, step, prob in sublist]
    analysis['frequently_blocked'] = {
        node: all_blocked.count(node) / len(mc_results['blocked_nodes'])
        for node in set(all_blocked)
    }

    return analysis


def plot_results(analysis):
    """Visualisation des résultats"""
    plt.figure(figsize=(14, 8))

    # Courbes principales avec intervalles de confiance
    metrics = [
        ('infected', 'red', 'Infectés'),
        ('recovered', 'green', 'Guéris'),
        ('blocked', 'orange', 'Bloqués')
    ]

    for metric, color, label in metrics:
        mean = analysis[f'{metric}_mean']
        steps = range(len(mean))

        plt.plot(steps, mean, label=f'{label} (moyenne)', color=color, linewidth=2)
        plt.fill_between(steps, analysis[f'{metric}_lower'], analysis[f'{metric}_upper'],
                         color=color, alpha=0.2)

    plt.xlabel('Étapes de simulation', fontsize=12)
    plt.ylabel('Nombre d\'aéroports', fontsize=12)
    plt.title('Dynamique SIR avec blocage stratégique\nAnalyse Monte Carlo (100 simulations)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)


    plt.tight_layout()
    plt.show()

    # Affichage des statistiques
    print("\n=== Statistiques clés ===")
    print(f"Pic d'infection moyen: {analysis['peak_stats']['mean']:.1f} ± {analysis['peak_stats']['std']:.1f}")
    print(
        f"Intervalle de confiance 95%: [{analysis['peak_stats']['ci'][0]:.1f}, {analysis['peak_stats']['ci'][1]:.1f}]")

    print("\nAéroports fréquemment bloqués (pourcentage des simulations):")
    for node, freq in sorted(analysis['frequently_blocked'].items(), key=lambda x: -x[1]):
        print(f"- {node}: {freq * 100:.1f}%")


if __name__ == "__main__":
    # Exécution de l'analyse Monte Carlo
    print("Démarrage de la simulation Monte Carlo...")
    mc_results = monte_carlo_simulation(G, n_simulations=100)

    # Analyse des résultats
    print("\nAnalyse des résultats...")
    analysis = analyze_results(mc_results)

    # Visualisation
    plot_results(analysis)