"""
Project 12: Airline Route Graph Analysis
Simple, clear, and direct Python implementation following all task specifications.
"""

from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Resolve base project directory so paths work from anywhere
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=========================================================")
print("     PROJECT 12: AIRLINE ROUTE GRAPH ANALYSIS           ")
print("=========================================================\n")

# -------------------------------------------------------------------------
# TASK 1: Load airline route data and construct a directed graph
# -------------------------------------------------------------------------
print("--- [TASK 1] Loading Data & Constructing Directed Graph ---")

# Load route data and master airports list
routes_df = pd.read_csv(DATA_DIR / 'Air-Clean.csv')
airports_df = pd.read_csv(DATA_DIR / 'indian_airports_master.csv')

# Strip whitespace from city names
routes_df['origin'] = routes_df['origin'].str.strip()
routes_df['destination'] = routes_df['destination'].str.strip()
airports_df['city'] = airports_df['city'].str.strip()

# Initialize Directed Graph
G = nx.DiGraph()

# Add all airports as vertices (including isolated airports from master list)
for _, row in airports_df.iterrows():
    city = row['city']
    G.add_node(city, state=row.get('state_ut', ''), iata=row.get('iata_code', ''))

# Count flight frequencies for each direct route (origin -> destination)
route_counts = routes_df.groupby(['origin', 'destination']).size().reset_index(name='flight_count')

# Add directed edges with flight count as weight
for _, row in route_counts.iterrows():
    G.add_edge(row['origin'], row['destination'], weight=row['flight_count'])

print(f"Total Airports (Vertices) in Network : {G.number_of_nodes()}")
print(f"Total Directed Routes (Edges)        : {G.number_of_edges()}")
print(f"Total Flight Schedules Processed     : {len(routes_df):,}\n")


# -------------------------------------------------------------------------
# TASK 2: Compute the total number of direct connections per airport
# -------------------------------------------------------------------------
print("--- [TASK 2] Direct Connections Per Airport ---")

# In directed graph:
# Out-degree = Direct destinations reached from this airport
# In-degree  = Direct origins flying into this airport
# Total direct connections = In-degree + Out-degree
degree_data = []
for node in G.nodes():
    in_deg = G.in_degree(node)
    out_deg = G.out_degree(node)
    total_connections = in_deg + out_deg
    degree_data.append({
        'Airport': node,
        'In_Degree': in_deg,
        'Out_Degree': out_deg,
        'Total_Connections': total_connections
    })

conn_df = pd.DataFrame(degree_data).sort_values(by='Total_Connections', ascending=False)

print("Top 10 Most Connected Airports:")
print(conn_df.head(10).to_string(index=False))
print("\n")


# -------------------------------------------------------------------------
# TASK 3: Identify the busiest airport using degree centrality
# -------------------------------------------------------------------------
print("--- [TASK 3] Busiest Airport Using Degree Centrality ---")

# Degree centrality measures how connected a node is relative to all possible connections
centrality = nx.degree_centrality(G)
conn_df['Degree_Centrality'] = conn_df['Airport'].map(centrality)

# Sort by Degree Centrality
conn_df = conn_df.sort_values(by='Degree_Centrality', ascending=False)
busiest_airport = conn_df.iloc[0]

print(f"Busiest Airport        : {busiest_airport['Airport']}")
print(f"Degree Centrality Score: {busiest_airport['Degree_Centrality']:.4f}")
print(f"Total Direct Routes    : {busiest_airport['Total_Connections']} (In: {busiest_airport['In_Degree']}, Out: {busiest_airport['Out_Degree']})")

print("\nTop 5 Busiest Airports by Degree Centrality:")
for idx, r in conn_df.head(5).reset_index().iterrows():
    print(f" {idx+1}. {r['Airport']} - Centrality: {r['Degree_Centrality']:.4f} ({r['Total_Connections']} connections)")
print("\n")


# -------------------------------------------------------------------------
# TASK 4: Find least connected or isolated airports in the network
# -------------------------------------------------------------------------
print("--- [TASK 4] Least Connected & Isolated Airports ---")

# Isolated airports: 0 total connections
isolated_airports = conn_df[conn_df['Total_Connections'] == 0]['Airport'].tolist()
print(f"Number of Isolated Airports (0 connections): {len(isolated_airports)}")
print(f"Sample Isolated Airports: {', '.join(isolated_airports[:10])}")

# Least connected active airports (connections > 0 but minimum)
active_airports = conn_df[conn_df['Total_Connections'] > 0]
min_connections = active_airports['Total_Connections'].min()
least_connected_active = active_airports[active_airports['Total_Connections'] == min_connections]['Airport'].tolist()

print(f"\nLeast Connected Active Airports ({min_connections} connection):")
print(f"{', '.join(least_connected_active[:10])}\n")


# -------------------------------------------------------------------------
# TASK 5: Compute PageRank to rank airports by importance
# -------------------------------------------------------------------------
print("--- [TASK 5] Ranking Airports by PageRank (Network Importance) ---")

# PageRank assigns higher importance to airports linked to other major hubs
pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight')
conn_df['PageRank'] = conn_df['Airport'].map(pagerank_scores)

# Sort by PageRank
conn_df = conn_df.sort_values(by='PageRank', ascending=False)

print("Top 10 Airports by PageRank Importance:")
for rank, (_, row) in enumerate(conn_df.head(10).iterrows(), 1):
    print(f" {rank:2d}. {row['Airport']:<18} | PageRank: {row['PageRank']:.5f} | Connections: {row['Total_Connections']}")

# Save full results to CSV
output_csv = OUTPUT_DIR / 'python_airport_analysis.csv'
conn_df.to_csv(output_csv, index=False)
print(f"\nFull analysis results saved to: {output_csv}\n")


# -------------------------------------------------------------------------
# TASK 6: Visualize the route network using a hub connectivity chart
# -------------------------------------------------------------------------
print("--- [TASK 6] Generating Hub Connectivity Visualizations ---")

plt.figure(figsize=(14, 6))

# Subplot 1: Bar chart of Top 15 Hubs by Total Connections
plt.subplot(1, 2, 1)
top_15 = conn_df.sort_values(by='Total_Connections', ascending=False).head(15)
bars = plt.barh(top_15['Airport'], top_15['Total_Connections'], color='#2563eb')
plt.gca().invert_yaxis()
plt.title('Top 15 Airline Hubs by Direct Connections', fontsize=12, fontweight='bold')
plt.xlabel('Total Connections (In + Out Degree)')
plt.ylabel('Airport')
plt.grid(axis='x', linestyle='--', alpha=0.6)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 1, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', fontsize=9)

# Subplot 2: Hub Connectivity Network Diagram
plt.subplot(1, 2, 2)
# Select top 20 hubs for a clean, readable network layout
top_hub_names = set(conn_df.head(20)['Airport'])
subgraph = G.subgraph(top_hub_names)

pos = nx.spring_layout(subgraph, seed=42, k=0.9)
node_sizes = [pagerank_scores.get(n, 0.01) * 30000 for n in subgraph.nodes()]

nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color='#38bdf8', edgecolors='#0284c7', alpha=0.85)
nx.draw_networkx_edges(subgraph, pos, edge_color='#94a3b8', alpha=0.4, arrows=True, arrowsize=8, connectionstyle='arc3,rad=0.1')
nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold', font_color='#0f172a')

plt.title('Hub Connectivity Network (Top 20 Airports)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
chart_path = OUTPUT_DIR / 'hub_connectivity_chart.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Chart successfully saved to: {chart_path}")
print("\n=========================================================")
print("            ALL 6 TASKS COMPLETED SUCCESSFULLY!          ")
print("=========================================================")
