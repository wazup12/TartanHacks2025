import networkx as nx
import numpy as np
import osmnx as ox
from PIL import Image, ImageDraw

def condense_nodes(nodes, G, grid_size=0.001):
    """
    Condenses nodes within a given grid window into an average node while preserving connectivity.

    Parameters:
        nodes: GeoDataFrame containing node data
        G: NetworkX graph of the street network
        grid_size (float): Size of the window for clustering (in degrees)

    Returns:
        tuple: (new_nodes, new_edges) where new_nodes is a dict of averaged points,
               and new_edges is a list of reconstructed edges.
    """
    from collections import defaultdict

    grid = defaultdict(list)
    node_mapping = {}

    # Assign nodes to grid cells
    for node_id, row in nodes.iterrows():
        grid_x = round(row.geometry.x / grid_size)
        grid_y = round(row.geometry.y / grid_size)
        grid[(grid_x, grid_y)].append((node_id, row.geometry.x, row.geometry.y))

    # Compute average positions for each cluster
    new_nodes = {}
    for (grid_x, grid_y), cluster in grid.items():
        avg_x = np.mean([x for _, x, _ in cluster])
        avg_y = np.mean([y for _, _, y in cluster])
        new_node_id = len(new_nodes)  # Assign new node ID
        new_nodes[new_node_id] = (avg_x, avg_y)
        for node_id, _, _ in cluster:
            node_mapping[node_id] = new_node_id

    # Reconstruct edges from the original graph
    new_edges = set()
    for u, v in G.edges():
        if u in node_mapping and v in node_mapping:
            new_u, new_v = node_mapping[u], node_mapping[v]
            if new_u != new_v:  # Avoid self-loops
                new_edges.add((new_u, new_v))

    return new_nodes, list(new_edges)

def draw_map(new_nodes, new_edges, image_size=(800, 800)):
    """
    Draws the condensed street network and saves the images.

    Parameters:
        new_nodes: Dict of condensed node positions
        new_edges: List of reconstructed edges
        image_size (tuple): Size of the output image (width, height)
    """
    img = Image.new("RGB", image_size, "black")
    draw = ImageDraw.Draw(img)

    # Normalize node coordinates
    min_x = min(x for x, y in new_nodes.values())
    max_x = max(x for x, y in new_nodes.values())
    min_y = min(y for x, y in new_nodes.values())
    max_y = max(y for x, y in new_nodes.values())

    def transform(x, y):
        x_norm = int((x - min_x) / (max_x - min_x) * (image_size[0] - 10) + 5)
        y_norm = int((y - min_y) / (max_y - min_y) * (image_size[1] - 10) + 5)
        return x_norm, y_norm

    # Draw edges
    for u, v in new_edges:
        x1, y1 = transform(*new_nodes[u])
        x2, y2 = transform(*new_nodes[v])
        draw.line((x1, y1, x2, y2), fill="white", width=2)

    # Draw nodes
    for x, y in new_nodes.values():
        x_norm, y_norm = transform(x, y)
        draw.ellipse((x_norm - 3, y_norm - 3, x_norm + 3, y_norm + 3), fill="red")

    img.save("condensed_street_map.png")
    print("Saved condensed_street_map.png")

# Example usage
if __name__ == "__main__":
    lat, lon = 42.3143359, -71.0525965
    G = ox.graph_from_point((lat, lon), dist=5000, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    new_nodes, new_edges = condense_nodes(nodes, G, grid_size=0.001)
    print(f"Reduced {len(nodes)} nodes to {len(new_nodes)} nodes.")
    print(f"Reconstructed {len(G.edges)} edges to {len(new_edges)} edges.")
    draw_map(new_nodes, new_edges)
