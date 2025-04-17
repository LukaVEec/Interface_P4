# Objectif II
import tkinter as tk
import networkx as nx
import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.data_processing import data_processing


def plot_solution(G,solution):
    """
    Plots the network on a sphere representing the Earth by using cartopy.

    Input:
        - G: a networkx graph representing the network.
    
    """
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0, central_latitude=0))
    ax.set_global()

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    for node in G.nodes():
        lat, lon = G.nodes[node]["latitude"], G.nodes[node]["longitude"]
        ax.scatter(lon, lat, color='red', s=10, transform=ccrs.PlateCarree())

    for i in range (len(solution)-1) :
        start = solution[i]
        end = solution[i+1]
        lat1, lon1 = G.nodes[start]["latitude"], G.nodes[start]["longitude"]
        lat2, lon2 = G.nodes[end]["latitude"], G.nodes[end]["longitude"]

        ax.plot([lon1, lon2], [lat1, lat2], color='blue', linewidth=0.3, transform=ccrs.PlateCarree())

    plt.title("Réseau aérien mondial avec Cartopy")
    plt.show()


def recommandations_interface(G, waiting_times, prices, J):
    root = tk.Tk()
    root.title("Recommandations")
    
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)
    
    canvas = tk.Canvas(frame)
    canvas.pack(side="left", fill="both", expand=True)
    
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Create a frame inside the canvas to hold all the widgets
    canvas_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=canvas_frame, anchor="nw")

    # Bind the scrollbar to the canvas's viewport
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    canvas_frame.bind("<Configure>", on_frame_configure)
    
    preference = tk.StringVar()
    preference.set("Distance")
    
    # Add question and radio buttons for preference
    Question1 = tk.Label(canvas_frame, text="Quel critère est le plus important pour vous ?")
    Question1.pack()
    tk.Radiobutton(canvas_frame, text="Distance [km]", value="Distance", variable=preference).pack()
    tk.Radiobutton(canvas_frame, text="Temps [min]", value="Temps", variable=preference).pack()
    tk.Radiobutton(canvas_frame, text="Prix [euros]", value="Prix", variable=preference).pack()

    start = tk.StringVar()
    start.set("Aéroport de départ")
    Question2 = tk.Label(canvas_frame, text="Aéroport de départ : ")
    Question2.pack()
    for airport in G.nodes:
        tk.Radiobutton(canvas_frame, text=airport, value=airport, variable=start).pack()

    end = tk.StringVar()
    end.set("Aéroport d'arrivée")
    Question3 = tk.Label(canvas_frame, text="Aéroport d'arrivée : ")
    Question3.pack()
    for airport in G.nodes:
        tk.Radiobutton(canvas_frame, text=airport, value=airport, variable=end).pack()

    # Add confirm button
    Confirm3 = tk.Button(canvas_frame, text="Confirmer le trajet", command=lambda: confirm_traject(root, start.get(), end.get(), G, J, preference.get(), prices, waiting_times))
    Confirm3.pack()
    
    # List to store confirmed paths' labels
    confirmed_labels = []
    confirmed_limit = 3  # Limit the number of confirmed paths to 3

    def confirm_traject(root, start, end, G, J, preference, prices, waiting_times):
        if(start not in G.nodes or end not in G.nodes):
            label = tk.Label(root, text="Veuillez entrer des aéroports valides.")
            label.pack()
            
            if len(confirmed_labels) >= confirmed_limit:
                label_to_remove = confirmed_labels.pop(0)  # Remove the first confirmed label
                label_to_remove.destroy()  # Destroy the old label
        
        # Add the new label to the list of confirmed labels
            confirmed_labels.append(label)
            return
        if (start, end) not in J:
            if dfs(start, end, G) == False:
                label = tk.Label(root, text="Ce trajet n'est pas possible.")
                label.pack()
                
                if len(confirmed_labels) >= confirmed_limit:
                    label_to_remove = confirmed_labels.pop(0)  # Remove the first confirmed label
                    label_to_remove.destroy()  # Destroy the old label
        
        # Add the new label to the list of confirmed labels
                confirmed_labels.append(label)
                return
        

        
        if preference == "Distance":
            values = nx.get_edge_attributes(G, "weight")
        elif preference == "Temps":
            values = {k: v / 900 * 60 for k, v in nx.get_edge_attributes(G, "weight").items()} # on divise par la vitesse moyenne d'un avion et on multiplie par 60 pour avoir le temps en minutes
        elif preference == "Prix":
            values = prices
        else:
            label = tk.Label(root, text="Veuillez entrer un critère valide.")
            label.pack()
            return

        routes_values = nx.DiGraph(G)
        if preference != "Distance":
            for u, v in routes_values.edges:
                routes_values[u][v]["weight"] = values.get((u, v))
                if preference == "Temps":
                    if u != start:
                        routes_values[u][v]["weight"] += waiting_times[u]

        solution = nx.shortest_path(G, source=start, target=end, weight="weight")
        total_value = sum(routes_values[u][v]["weight"] for u, v in zip(solution[:-1], solution[1:]))
    
        label = tk.Label(root, text=f"Chemin trouvé: {solution}\nValeur totale: {total_value}")
        label.pack()
        
        plot_solution(G,solution)
        
        if len(confirmed_labels) >= confirmed_limit:
            label_to_remove = confirmed_labels.pop(0)  # Remove the first confirmed label
            label_to_remove.destroy()  # Destroy the old label
        
        # Add the new label to the list of confirmed labels
        confirmed_labels.append(label)

    def dfs(start, end, G):
        visited = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current == end:
                return True
            if current not in visited:
                visited.add(current)
                stack.extend(neighbor for neighbor in G.neighbors(current) if neighbor not in visited)
        return False
    
    tk.mainloop()

