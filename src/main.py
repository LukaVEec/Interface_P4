import new_network


best_cost, selected_edges = new_network.new_network("files/airports.csv", "files/pre_existing_routes.csv", "files/wanted_journeys.csv", 1000)

print("Meilleur coût :", best_cost)
print("Arêtes sélectionnées :", selected_edges)