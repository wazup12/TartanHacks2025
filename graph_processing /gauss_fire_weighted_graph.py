import osmnx as ox
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
import time
import cv2
import matplotlib.colors as mcolors

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
        "edges": [(int(u), int(v), {"weight": float(G.edges[u, v]["weight"]), "importance": float(G.edges[u, v]["importance"])}) for u, v in G.edges]
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

def create_overlay(road_img, intersection_img):
    if len(road_img.shape) == 3:
        road_img = cv2.cvtColor(road_img, cv2.COLOR_BGR2GRAY)  

    overlay_img = cv2.merge([road_img, road_img, road_img])  
    overlay_img[intersection_img > 0] = [0, 0, 255]  

    return overlay_img

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

def create_street_and_intersection_maps(lat, lon, image_size=(800, 800), dist=1000, scale=1.0, intersection_radius=5):
    try:
        edges, nodes, bounds = fetch_street_network(lat, lon, dist)
        geo_to_pixel = create_coordinate_transformer(bounds, image_size, scale)

        road_img, _ = draw_streets(edges, geo_to_pixel, image_size)
        intersection_img, intersections = mark_intersections(nodes, geo_to_pixel, image_size, intersection_radius)
        overlay_img = create_overlay(road_img, intersection_img)

        save_images(road_img, intersection_img, overlay_img)
        return intersections, bounds, image_size  # also return bounds and image_size for later use
        
    except Exception as e:
        print(f"Error generating maps: {str(e)}")
        return None, None, None

################################################################################
# NEW FUNCTION: Update edge weights using the fire mask
################################################################################
def update_edge_weights_with_fire(G, fire_mask, road_image_size=(800,800), fire_mask_size=(500,500), sigma=20, beta=1):
    """
    Updates the 'weight' attribute for each edge in the graph G using a combination
    of the road importance and the distance from the edge (midpoint) to the active fire.
    """
    # Compute scale factors to convert coordinates from road_image to fire_mask
    scale_x = fire_mask_size[0] / road_image_size[0]
    scale_y = fire_mask_size[1] / road_image_size[1]
    
    # Ensure the fire mask is binary (assume white=fire)
    fire_binary = (fire_mask > 127).astype(np.uint8) * 255
    # Invert the mask so that fire pixels become 0 (so that distance transform gives 0 distance inside fire)
    inverted_fire = 255 - fire_binary
    # Compute the distance transform (each pixel will hold the distance to the nearest fire pixel)
    dt = cv2.distanceTransform(inverted_fire, cv2.DIST_L2, 5)
    
    for u, v, data in G.edges(data=True):
        pos_u = G.nodes[u]["pos"]
        pos_v = G.nodes[v]["pos"]
        # Compute the midpoint (in road image coordinates)
        mid_x = (pos_u[0] + pos_v[0]) / 2.0
        mid_y = (pos_u[1] + pos_v[1]) / 2.0
        # Convert midpoint to fire mask coordinates
        fire_x = int(mid_x * scale_x)
        fire_y = int(mid_y * scale_y)
        # Make sure the coordinates are within the fire mask bounds
        fire_x = np.clip(fire_x, 0, fire_mask_size[0] - 1)
        fire_y = np.clip(fire_y, 0, fire_mask_size[1] - 1)
        # Look up the distance value from the distance transform
        distance = dt[fire_y, fire_x]
        # Get the base road importance
        road_importance = data["importance"]
        # Compute a new weight.
        new_weight = road_importance * (1 + beta * np.exp(-distance / sigma))
        # Update the edge attribute
        G.edges[u, v]["weight"] = new_weight
    print("Edge weights updated based on fire mask.")
    return G

################################################################################
# NEW FUNCTION: Generate a heatmap image from the weighted graph
################################################################################
def generate_fire_heatmap(G, road_image_size=(800,800), line_thickness=4, blur_kernel=(15,15)):
    """
    Generates and saves a heatmap image where each edge is drawn in a grayscale value 
    corresponding to its computed weight. The resulting grayscale image is then 
    converted to a color heatmap using cv2.applyColorMap.
    
    A uniform Gaussian blur is applied to the final color heatmap before saving.
    
    The heatmap is saved as 'fire_heatmap.png'.
    """
    # Create a blank grayscale image (height, width)
    heatmap_img = np.zeros((road_image_size[1], road_image_size[0]), dtype=np.float32)
    
    # Collect all weights to normalize later
    weights = [G.edges[e]["weight"] for e in G.edges]
    if not weights:
        print("No edges in graph to create heatmap.")
        return None
    min_w, max_w = min(weights), max(weights)
    
    # Draw each edge on the heatmap image using a normalized intensity
    for u, v, data in G.edges(data=True):
        pos_u = G.nodes[u]["pos"]
        pos_v = G.nodes[v]["pos"]
        weight_val = data["weight"]
        # Normalize the weight to [0, 255]
        if max_w != min_w:
            normalized = 255 * (weight_val - min_w) / (max_w - min_w)
        else:
            normalized = 127
        pt1 = (int(pos_u[0]), int(pos_u[1]))
        pt2 = (int(pos_v[0]), int(pos_v[1]))
        cv2.line(heatmap_img, pt1, pt2, color=float(normalized), thickness=line_thickness)
    
    # Convert the float image to uint8
    heatmap_uint8 = np.clip(heatmap_img, 0, 255).astype(np.uint8)
    # Apply a colormap to get a color heatmap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # -----------------------
    # Apply Gaussian blur
    # -----------------------
    blurred_heatmap = cv2.GaussianBlur(heatmap_color, blur_kernel, 0)
    
    # Save the final blurred heatmap
    cv2.imwrite("fire_heatmap.png", blurred_heatmap)
    print("Fire heatmap saved as fire_heatmap.png")
    return blurred_heatmap

################################################################################
# Main execution
################################################################################
if __name__ == "__main__":
    # Coordinates for the center point
    lat, lon = 40.4450105,-79.9657356
    start = time.time()
    
    # Create the road and intersection maps; note that we now also return bounds and image_size
    intersections, bounds, road_image_size = create_street_and_intersection_maps(lat, lon, intersection_radius=3, dist=5000)
    
    if intersections is not None:
        # Fetch a road network at a (possibly different) scale to build the graph.
        edges, nodes, _ = fetch_street_network(lat, lon, dist=5000)
        geo_to_pixel = create_coordinate_transformer(bounds, road_image_size, scale=1.0)
        street_graph = create_graph_from_streets(intersections, edges, geo_to_pixel, output_file="street_graph.json")
        print(f"Graph created with {len(street_graph.nodes)} nodes and {len(street_graph.edges)} edges.")
        
        # Plot the graph with original (importance-based) colors.
        plot_color_graph(street_graph)
        
        # ----------------------------------------------------------------------
        # NEW: Load a fire mask and update the edge weights accordingly.
        # It is assumed that 'mask_test.jpg' is a 500x500 grayscale image where white (255)
        # indicates areas of active fire.
        # ----------------------------------------------------------------------
        fire_mask = cv2.imread("mask_test.jpg", cv2.IMREAD_GRAYSCALE)
        if fire_mask is None:
            print("Fire mask image not found. Please provide a 500x500 image named 'mask_test.jpg'.")
        else:
            # Update the edge weights using the fire mask.
            street_graph = update_edge_weights_with_fire(
                street_graph, fire_mask,
                road_image_size=road_image_size, fire_mask_size=(500,500),
                sigma=20, beta=1
            )
            
            # Generate and save the fire heatmap image with Gaussian blur applied.
            heatmap_img = generate_fire_heatmap(street_graph, road_image_size=road_image_size, line_thickness=7)
    
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
