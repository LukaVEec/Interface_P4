import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.data_processing import data_processing


def plot_network(G):
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

    for start, end in G.edges():
        lat1, lon1 = G.nodes[start]["latitude"], G.nodes[start]["longitude"]
        lat2, lon2 = G.nodes[end]["latitude"], G.nodes[end]["longitude"]

        ax.plot([lon1, lon2], [lat1, lat2], color='blue', linewidth=0.3, transform=ccrs.PlateCarree())

    plt.title("Réseau aérien mondial avec Cartopy")
    plt.show()


G, _ = data_processing("files/airports.csv", "files/pre_existing_routes.csv")
plot_network(G)