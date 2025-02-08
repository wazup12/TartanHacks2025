import osmnx as ox
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
import time
import cv2

# Define road importance mapping
ROAD_IMPORTANCE = {
    "motorway": 5,
    "trunk": 4,
    "primary": 3,
    "secondary": 2,
    "tertiary": 1,
    "residential": 0.5,
    "unclassified": 0.5,
    "service": 0.3
}

def fetch_street_network(lat, lon, dist=100):
    G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
    nodes = ox.graph_to_gdfs(G, edges=False)
    bounds = edges.total_bounds
    return edges, nodes, bounds

def create_coordinate_transformer(bounds, image_size, scale=1.0):
    def geo_to_pixel(lon, lat):
        x = int((lon - bounds[0]) / (bounds[2] - bounds[0]) * (image_size[0] - 1) * scale)
        y = int((bounds[3] - lat) / (bounds[3] - bounds[1]) * (image_size[1] - 1) * scale)
        return x, y
    return geo_to_pixel

def create_graph_from_streets(intersections, edges, geo_to_pixel, output_file="graph.json"):
    G = nx.Graph()
    intersection_array = np.array(intersections)
    
    for idx, (x, y) in enumerate(intersections):
        G.add_node(idx, pos=(int(x), int(y)))  # Convert to native Python int
    
    def find_nearest_intersection(pixel_coord):
        diffs = intersection_array - np.array(pixel_coord)
        dists = np.einsum('ij,ij->i', diffs, diffs)
        return np.argmin(dists)
    
    for _, row in edges.iterrows():
        coords = np.array([geo_to_pixel(lon, lat) for lon, lat in row.geometry.coords])
        node_indices = np.apply_along_axis(find_nearest_intersection, 1, coords)
        unique_edges = set(zip(node_indices[:-1], node_indices[1:]))
        
        # Determine road importance score
        highway_type = row.get("highway", "unclassified")
        if isinstance(highway_type, list):  
            highway_type = highway_type[0]  # If multiple types exist, take the first one
        
        importance_value = ROAD_IMPORTANCE.get(highway_type, 0.5)  # Default to 0.5 if unknown

        for node1, node2 in unique_edges:
            if node1 != node2:
                G.add_edge(int(node1), int(node2), weight=0, importance=importance_value)  
    
    # Convert networkx graph to a JSON-friendly format
    graph_dict = {
        "nodes": {int(n): {"pos": (int(G.nodes[n]["pos"][0]), int(G.nodes[n]["pos"][1]))} for n in G.nodes},
        "edges": [(int(u), int(v), {"weight": int(G.edges[u, v]["weight"]), "importance": float(G.edges[u, v]["importance"])}) for u, v in G.edges]
    }
    
    with open(output_file, "w") as f:
        json.dump(graph_dict, f)
    
    print(f"Graph saved to {output_file}")
    return G

def draw_streets(edges, geo_to_pixel, image_size):
    road_img = np.zeros(image_size, dtype=np.uint8)
    overlay_img = np.zeros((*image_size, 3), dtype=np.uint8)
    
    for _, row in edges.iterrows():
        coords = row.geometry.coords
        pixel_coords = [geo_to_pixel(lon, lat) for lon, lat in coords]
        
        for i in range(len(pixel_coords) - 1):
            x1, y1 = pixel_coords[i]
            x2, y2 = pixel_coords[i + 1]
            
            num_points = max(abs(x2 - x1), abs(y2 - y1)) * 2
            x_coords = np.linspace(x1, x2, num=int(num_points))
            y_coords = np.linspace(y1, y2, num=int(num_points))
            
            x_coords = np.clip(x_coords.astype(int), 0, image_size[0] - 1)
            y_coords = np.clip(y_coords.astype(int), 0, image_size[1] - 1)
            
            road_img[y_coords, x_coords] = 255
            overlay_img[y_coords, x_coords] = [255, 255, 255]
    
    return road_img, overlay_img

def create_street_and_intersection_maps(lat, lon, image_size=(800, 800), dist=1000, scale=1.0, intersection_radius=5):
    try:
        edges, nodes, bounds = fetch_street_network(lat, lon, dist)
        geo_to_pixel = create_coordinate_transformer(bounds, image_size, scale)

        road_img, _ = draw_streets(edges, geo_to_pixel, image_size)
        intersection_img, intersections = mark_intersections(nodes, geo_to_pixel, image_size, intersection_radius)
        overlay_img = create_overlay(road_img, intersection_img)

        save_images(road_img, intersection_img, overlay_img)
        return intersections
        
    except Exception as e:
        print(f"Error generating maps: {str(e)}")
        return None
    
def create_overlay(road_img, intersection_img):
    if len(road_img.shape) == 3:
        road_img = cv2.cvtColor(road_img, cv2.COLOR_BGR2GRAY)  

    overlay_img = cv2.merge([road_img, road_img, road_img])  
    overlay_img[intersection_img > 0] = [0, 0, 255]  

    return overlay_img

def mark_intersections(nodes, geo_to_pixel, image_size, intersection_radius=5):
    """
    Marks intersections on an image.
    """
    intersection_img = np.zeros(image_size, dtype=np.uint8)
    intersections = []

    for _, row in nodes.iterrows():
        x, y = geo_to_pixel(row.geometry.x, row.geometry.y)
        intersections.append((x, y))
        cv2.circle(intersection_img, (x, y), intersection_radius, 255, -1)  # Draw intersection as a filled circle

    return intersection_img, intersections

def save_images(road_img, intersection_img, overlay_img):
    cv2.imwrite("road_map.png", road_img)
    cv2.imwrite("intersection_map.png", intersection_img)
    cv2.imwrite("overlay.png", overlay_img)
    print("Saved road_map.png, intersection_map.png, and overlay.png successfully.")

def plot_graph(G):
    plt.figure(figsize=(10, 10))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get edge importance values
    edge_weights = [G.edges[e]["importance"] for e in G.edges]
    edge_colors = ["red" if w > 3 else "blue" for w in edge_weights]  # Highlight important roads
    
    nx.draw(G, pos, with_labels=False, node_size=50, node_color='black', edge_color=edge_colors)
    plt.title("Street Graph with Road Importance")
    plt.show()

import matplotlib.colors as mcolors

def plot_color_graph(G):
    """
    Plots the graph with edges color-coded based on importance.
    """
    plt.figure(figsize=(10, 10))
    pos = nx.get_node_attributes(G, 'pos')

    # Define color map for different road importance levels
    color_map = {
        5: "red",        # Highways
        4: "darkorange", # Trunk roads
        3: "orange",     # Primary roads
        2: "yellow",     # Secondary roads
        1: "green",      # Tertiary roads
        0.5: "blue",     # Residential, unclassified
        0.3: "cyan"      # Service roads
    }

    # Create edge colors based on importance
    edge_colors = [
        color_map.get(G.edges[e]["importance"], "gray")  # Default to gray if not found
        for e in G.edges
    ]

    nx.draw(G, pos, with_labels=False, node_size=20, node_color="black", edge_color=edge_colors, width=1)

    # Create a legend for the color-coded road importance
    legend_labels = {
        "red": "Highways",
        "darkorange": "Trunk",
        "orange": "Primary",
        "yellow": "Secondary",
        "green": "Tertiary",
        "blue": "Residential",
        "cyan": "Service"
    }

    # Draw legend
    legend_patches = [plt.Line2D([0], [0], color=color, lw=3, label=label) for color, label in legend_labels.items()]
    plt.legend(handles=legend_patches, loc="upper right")

    plt.title("Street Graph with Importance-Based Road Colors")
    plt.show()

if __name__ == "__main__":
    lat, lon = 52.2216033, 21.057503
    start = time.time()
    intersections = create_street_and_intersection_maps(lat, lon, intersection_radius=3, dist=7000)
    
    if intersections:
        edges, nodes, bounds = fetch_street_network(lat, lon, dist=2000)
        geo_to_pixel = create_coordinate_transformer(bounds, (800, 800), scale=1.0)
        street_graph = create_graph_from_streets(intersections, edges, geo_to_pixel, output_file="street_graph.json")
        print(f"Graph created with {len(street_graph.nodes)} nodes and {len(street_graph.edges)} edges.")
    
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
    plot_color_graph(street_graph)
