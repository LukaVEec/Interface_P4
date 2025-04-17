import streamlit as st
import data_processing
import pandas as pd
import networkx as nx
import plot_network

# Liste fictive d'aéroports
G, edges = data_processing.data_processing("files/airports.csv", "files/pre_existing_routes.csv")

attente = pd.read_csv("files/waiting_times.csv")
prix = pd.read_csv("files/prices.csv")

prices = dict(zip(zip(prix["ID_start"], prix["ID_end"]), prix["price_tag"]))
waiting_times = dict(zip(attente["ID"], attente["idle_time"]))
# Ajout de nouvel attribut au graphe (prix) en plus du poids
for start, end in G.edges():
   
    G[start][end]['prices'] = prices[(start, end)]
    G[start][end]['time'] = G[start][end]['weight'] / 900 * 60
    


airport_data = pd.read_csv("files/airports.csv")
airport_data.set_index("ID", inplace=True)
ids = list(G.nodes)
# Titre
st.title("✈️ Recommandations de Vols Internationaux")

# Sélections utilisateur
departure = st.selectbox("Départ", ids, format_func=lambda x: f"{x} - {airport_data.loc[x, "name"]} - {airport_data.loc[x, "city"]}")
arrival = st.selectbox("Arrivée", ids, format_func=lambda x: f"{x} - {airport_data.loc[x, "name"]} - {airport_data.loc[x, "city"]}")
criterion = st.radio("Critère préféré", ["Temps", "Distance", "Prix"])

# Bouton de lancement
if st.button("Trouver le trajet optimal"):
    
    if departure == arrival:
        st.warning("🚨 Attention le départ et l’arrivée ne peuvent pas être identiques.")
    elif nx.has_path(G, departure, arrival) == False:
        st.warning("🚨 Attention, il n'existe pas de chemin entre ces deux aéroports.")
    else:
        # Ajout des temps d'attente pour les noeuds qui ne sont pas les noeuds de départ ou d'arrivée
        for edge in G.edges():
            if edge[1] != arrival:
                G[edge[0]][edge[1]]['time'] += waiting_times[edge[1]]
                

        if criterion == "Distance":
            new_G = nx.DiGraph()
            st.success(f"✔️ Trajet optimal de {departure} à {arrival}")
            st.write("Critère choisi : Distance")
            st.write("Itinéraire suggéré :")
            shortest_path = nx.shortest_path(G, source=departure, target=arrival, weight='weight')
            distance = nx.shortest_path_length(G, source=departure, target=arrival, weight='weight')
            # Création d'un graphe avec seulement les arêtes du chemin le plus court
            for i in range(len(shortest_path)-1):
                edge = (shortest_path[i], shortest_path[i+1])
                new_G.add_edge(shortest_path[i], shortest_path[i+1], weight=G[edge[0]][edge[1]]['weight'])
            duration = 0
            price = 0
            for i in range(len(shortest_path)-1):
                edge = (shortest_path[i], shortest_path[i+1])
                if edge in prices:
                    price += prices[edge]
                duration += G[edge[0]][edge[1]]['time']
                
            string = " → ".join(shortest_path)
            st.markdown("➡️ " + string)
            st.write(f"🕓 Durée estimée : {int(duration//60)} h {int((duration/60-duration//60)*60)} min")
            st.write(f"💰 Prix estimé : {"%.2f" % price} €")
            st.write(f"📏 Distance estimée : {distance} km")
            st.pyplot(plot_network.plot_network(new_G, "files/airports.csv"))
        elif criterion == "Prix":
            new_G = nx.DiGraph()
            st.success(f"✔️ Trajet optimal de {departure} à {arrival}")
            st.write("Critère choisi : Prix")
            st.write("Itinéraire suggéré :")
            shortest_path = nx.shortest_path(G, source=departure, target=arrival, weight='prices')
            price = nx.shortest_path_length(G, source=departure, target=arrival, weight='prices')
            duration = 0
            distance = 0
            for i in range(len(shortest_path)-1):
                edge = (shortest_path[i], shortest_path[i+1])
                distance += G[edge[0]][edge[1]]['weight']
                duration += G[edge[0]][edge[1]]['time']          
            string = " → ".join(shortest_path)
            # Création d'un graphe avec seulement les arêtes du chemin le plus court
            for i in range(len(shortest_path)-1):
                edge = (shortest_path[i], shortest_path[i+1])
                new_G.add_edge(shortest_path[i], shortest_path[i+1], weight=G[edge[0]][edge[1]]['weight'])
            st.markdown("➡️ " + string)
            st.write(f"🕓 Durée estimée : {int(duration//60)} h {int((duration/60-duration//60)*60)} min")
            st.write(f"💰 Prix estimé : {"%.2f" % price} €")
            st.write(f"📏 Distance estimée : {distance} km")
            st.pyplot(plot_network.plot_network(new_G, "files/airports.csv"))

        else:
            new_G = nx.DiGraph()
            st.success(f"✔️ Trajet optimal de {departure} à {arrival}")
            st.write("Critère choisi : Temps")
            st.write("Itinéraire suggéré :")
            shortest_path = nx.shortest_path(G, source=departure, target=arrival, weight='time')
            time = nx.shortest_path_length(G, source=departure, target=arrival, weight='time')
            price = 0
            distance = 0
            for i in range(len(shortest_path)-1):
                edge = (shortest_path[i], shortest_path[i+1])
                distance += G[edge[0]][edge[1]]['weight']
                price += prices[edge]

            string = " → ".join(shortest_path)
            st.markdown("➡️ " + string)
            st.write(f"🕓 Durée estimée : {int(time//60)} h {int((time/60-time//60)*60)} min")
            st.write(f"💰 Prix estimé : {"%.2f" % price} €")
            st.write(f"📏 Distance estimée : {distance} km")
            # Création d'un graphe avec seulement les arêtes du chemin le plus court
            for i in range(len(shortest_path)-1):
                edge = (shortest_path[i], shortest_path[i+1])
                new_G.add_edge(shortest_path[i], shortest_path[i+1], weight=G[edge[0]][edge[1]]['weight'])
            st.pyplot(plot_network.plot_network(new_G, "files/airports.csv"))
