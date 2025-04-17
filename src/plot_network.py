import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import data_processing

def plot_network(G, airports_csv="airports.csv"):
    """
    Plots the network on a flat map using Cartopy.

    Inputs:
        - G: a networkx graph where nodes are 3-letter IATA airport codes.
        - airports_csv: path to the CSV file containing airport data (with columns: ID, latitude, longitude).
    """
    # Charger les données d’aéroports
    airport_data = pd.read_csv(airports_csv)
    airport_data.set_index("ID", inplace=True)

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    for node in G.nodes():
        if node in airport_data.index:
            lat = airport_data.loc[node, "latitude"]
            lon = airport_data.loc[node, "longitude"]
            ax.scatter(lon, lat, color='red', s=10, transform=ccrs.PlateCarree())
            ax.text(lon, lat, node, fontsize=8, ha='right', transform=ccrs.PlateCarree(), fontweight='bold')

    for start, end in G.edges():
        if start in airport_data.index and end in airport_data.index:
            lat1 = airport_data.loc[start, "latitude"]
            lon1 = airport_data.loc[start, "longitude"]
            lat2 = airport_data.loc[end, "latitude"]
            lon2 = airport_data.loc[end, "longitude"]
            ax.plot([lon1, lon2], [lat1, lat2], color='blue', linewidth=0.3, transform=ccrs.PlateCarree())

    plt.title("Vos trajets recommandés")
    return fig


G, edges = data_processing.data_processing("files/airports.csv", "files/pre_existing_routes.csv")


