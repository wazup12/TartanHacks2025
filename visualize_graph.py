import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(json_file):
    # Load the graph JSON from file.
    with open(json_file, "r") as f:
        graph_data = json.load(f)

    # Create a NetworkX graph.
    G = nx.Graph()

    # Add nodes with their positions.
    pos = {}
    for node_id, node_info in graph_data["nodes"].items():
        # Ensure keys are strings.
        node_key = str(node_id)
        pos[node_key] = node_info["pos"]
        G.add_node(node_key, pos=node_info["pos"])

    # Collect weights for all edges and add them to the graph.
    edge_weights = []
    for edge in graph_data.get("edges", []):
        source = str(edge[0])
        target = str(edge[1])
        weight = edge[2]["weight"]
        edge_weights.append(weight)
        G.add_edge(source, target, weight=weight)

    # Convert weights to a numpy array.
    weights = np.array(edge_weights)

    # To map weight to edge thickness, we want lower weight (more likely) to be thicker.
    # We do this by normalizing the weights and inverting the scale.
    eps = 1e-6
    min_weight = np.min(weights)
    max_weight = np.max(weights)

    # Normalize the weights to [0, 1].
    normalized = (weights - min_weight) / (max_weight - min_weight + eps)
    # Invert the normalized value: lower weight (normalized near 0) should be thicker.
    # Here, we set a minimum thickness and a maximum thickness.
    min_thickness = 0.5
    max_thickness = 5.0
    # Compute thickness: when normalized==0 -> thickness = max_thickness, when normalized==1 -> thickness = min_thickness.
    edge_thicknesses = max_thickness - normalized * (max_thickness - min_thickness)

    # Create a mapping from edge to thickness.
    # The order of edges in G.edges() should match the order in our edge_weights list.
    widths = []
    for i, edge in enumerate(G.edges()):
        widths.append(edge_thicknesses[i])

    # Draw the graph.
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=widths)

    plt.title("Graph Visualization with Edge Thickness Based on Weight")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Update the file name to your updated JSON graph file.
    visualize_graph("updated_street_graph.json")
