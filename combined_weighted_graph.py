#!/usr/bin/env python3
"""
Combined street network and fire-aware mapping tool.

Usage:
  python combined_fire_map.py "Place Name" [fire_mask_file]

The script uses the place name (e.g., "New York, NY") to look up coordinates,
fetch the street network, generate road and intersection maps, build a graph,
and (if a fire mask file is provided) update edge weights based on the fire mask,
producing a fire heatmap image. All output files are saved to disk, and a JSON
summary is printed to stdout.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox

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

#######################################
#  Utility Functions for Map Creation #
#######################################

def fetch_street_network(lat, lon, dist=1000):
    """
    Given a center coordinate and distance, fetch the street network.
    Returns edges, nodes and total bounds.
    """
    G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
    nodes = ox.graph_to_gdfs(G, edges=False)
    bounds = edges.total_bounds  # (min_x, min_y, max_x, max_y)
    return edges, nodes, bounds

def create_coordinate_transformer(bounds, image_size, scale=1.0):
    """
    Returns a function to convert geographic (lon, lat) coordinates to
    pixel coordinates for an image of the given size.
    """
    min_x, min_y, max_x, max_y = bounds
    width, height = image_size

    def geo_to_pixel(lon, lat):
        x = int((lon - min_x) / (max_x - min_x) * (width - 1) * scale)
        y = int((max_y - lat) / (max_y - min_y) * (height - 1) * scale)
        return x, y

    return geo_to_pixel

def draw_streets(edges, geo_to_pixel, image_size):
    """
    Draws roads on a blank image by interpolating points along each street segment.
    Returns a grayscale road image and an overlay image.
    """
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
    Marks intersections on a blank image. Returns the image and a list of intersection pixel coordinates.
    """
    intersection_img = np.zeros(image_size, dtype=np.uint8)
    intersections = []
    
    for _, row in nodes.iterrows():
        x, y = geo_to_pixel(row.geometry.x, row.geometry.y)
        intersections.append((x, y))
        cv2.circle(intersection_img, (x, y), intersection_radius, 255, -1)
    
    return intersection_img, intersections

def create_overlay(road_img, intersection_img):
    """
    Creates an overlay image by merging the road image with colored intersection markers.
    """
    if len(road_img.shape) == 3:
        road_img = cv2.cvtColor(road_img, cv2.COLOR_BGR2GRAY)
    overlay_img = cv2.merge([road_img, road_img, road_img])
    overlay_img[intersection_img > 0] = [0, 0, 255]
    return overlay_img

def save_images(road_img, intersection_img, overlay_img, place, out_dir="static/"):
    """
    Saves the provided images with filenames that incorporate the place string.
    Returns the file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    road_path = os.path.join(out_dir, f"{place}_road_map.png")
    inter_path = os.path.join(out_dir, f"{place}_intersection_map.png")
    overlay_path = os.path.join(out_dir, f"{place}_overlay.png")
    cv2.imwrite(road_path, road_img)
    cv2.imwrite(inter_path, intersection_img)
    cv2.imwrite(overlay_path, overlay_img)
    return road_path, inter_path, overlay_path

def create_street_and_intersection_maps(lat, lon, image_size=(800, 800), dist=1000,
                                          scale=1.0, intersection_radius=5, place="default"):
    """
    Creates the street and intersection maps:
      - Fetches street network data.
      - Converts geographic coordinates to pixel space.
      - Draws roads and marks intersections.
      - Creates an overlay image.
      - Saves the images to disk.
    
    Returns a tuple: (intersections, bounds, image_size, road_img_path, intersection_img_path, overlay_img_path)
    """
    try:
        edges, nodes, bounds = fetch_street_network(lat, lon, dist)
        geo_to_pixel = create_coordinate_transformer(bounds, image_size, scale)
        road_img, _ = draw_streets(edges, geo_to_pixel, image_size)
        intersection_img, intersections = mark_intersections(nodes, geo_to_pixel, image_size, intersection_radius)
        overlay_img = create_overlay(road_img, intersection_img)
        road_path, inter_path, overlay_path = save_images(road_img, intersection_img, overlay_img, place)
        return intersections, bounds, image_size, road_path, inter_path, overlay_path
    except Exception as e:
        print(f"Error generating maps: {str(e)}")
        return None, None, None, None, None, None

##########################################
# Graph Construction and Fire Functions  #
##########################################

def create_graph_from_streets(intersections, edges, geo_to_pixel, output_file="graph.json"):
    """
    Builds a graph from the given intersections and street edges. Each edge gets an initial weight of 0
    and an importance value based on its highway type.
    
    The graph is saved as JSON to output_file.
    """
    G = nx.Graph()
    intersection_array = np.array(intersections)
    
    # Add nodes
    for idx, (x, y) in enumerate(intersections):
        G.add_node(idx, pos=(int(x), int(y)))
    
    # Helper function to find the nearest intersection
    def find_nearest_intersection(pixel_coord):
        diffs = intersection_array - np.array(pixel_coord)
        dists = np.einsum('ij,ij->i', diffs, diffs)
        return np.argmin(dists)
    
    # Process each edge from OSMnx data
    for _, row in edges.iterrows():
        coords = [geo_to_pixel(lon, lat) for lon, lat in row.geometry.coords]
        node_indices = np.apply_along_axis(find_nearest_intersection, 1, np.array(coords))
        unique_edges = set(zip(node_indices[:-1], node_indices[1:]))
        
        highway_type = row.get("highway", "unclassified")
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        importance_value = ROAD_IMPORTANCE.get(highway_type, 0.5)
        
        for node1, node2 in unique_edges:
            if node1 != node2:
                G.add_edge(int(node1), int(node2), weight=0, importance=importance_value)
    
    # Save the graph JSON
    graph_dict = {
        "nodes": {n: {"pos": G.nodes[n]["pos"]} for n in G.nodes},
        "edges": [(u, v, {"weight": G.edges[u, v]["weight"], "importance": G.edges[u, v]["importance"]})
                  for u, v in G.edges]
    }
    with open(output_file, "w") as f:
        json.dump(graph_dict, f)
    print(f"Graph saved to {output_file}")
    return G

def update_edge_weights_with_fire(G, fire_mask, road_image_size=(800,800), fire_mask_size=(500,500), sigma=20, beta=1):
    """
    Updates the 'weight' attribute for each edge in graph G based on the distance from the edge's midpoint
    to active fire areas (where the fire_mask is white). The weight is computed as:
    
         weight = importance * (1 + beta * exp(-distance/sigma))
    
    Parameters:
      - G: networkx graph with an 'importance' attribute on each edge.
      - fire_mask: a 500x500 grayscale image (numpy array) where white (255) indicates fire.
      - road_image_size: dimensions of the road/intersection images.
      - fire_mask_size: dimensions of the fire mask.
      - sigma: controls the decay of influence with distance.
      - beta: controls how strongly the fire influences the weight.
    """
    # Compute scaling factors from road image to fire mask coordinates.
    scale_x = fire_mask_size[0] / road_image_size[0]
    scale_y = fire_mask_size[1] / road_image_size[1]
    
    # Convert fire mask to binary: assume fire > 127 indicates fire.
    fire_binary = (fire_mask > 127).astype(np.uint8) * 255
    # Invert so that fire areas are 0 (for distance transform)
    inverted_fire = 255 - fire_binary
    dt = cv2.distanceTransform(inverted_fire, cv2.DIST_L2, 5)
    
    for u, v, data in G.edges(data=True):
        pos_u = G.nodes[u]["pos"]
        pos_v = G.nodes[v]["pos"]
        mid_x = (pos_u[0] + pos_v[0]) / 2.0
        mid_y = (pos_u[1] + pos_v[1]) / 2.0
        # Convert midpoint to fire mask coordinates.
        fire_x = int(mid_x * scale_x)
        fire_y = int(mid_y * scale_y)
        fire_x = np.clip(fire_x, 0, fire_mask_size[0]-1)
        fire_y = np.clip(fire_y, 0, fire_mask_size[1]-1)
        distance = dt[fire_y, fire_x]
        road_importance = data["importance"]
        new_weight = road_importance * (1 + beta * np.exp(-distance / sigma))
        G.edges[u, v]["weight"] = new_weight
    print("Edge weights updated based on fire mask.")
    return G

def generate_fire_heatmap(G, road_image_size=(800,800), line_thickness=2):
    """
    Generates a heatmap image from graph G by drawing each edge with intensity
    corresponding to its weight. The resulting image is color-mapped and saved.
    
    Returns the color heatmap image.
    """
    heatmap_img = np.zeros((road_image_size[1], road_image_size[0]), dtype=np.float32)
    weights = [G.edges[e]["weight"] for e in G.edges]
    if not weights:
        print("No edges to create heatmap.")
        return None
    min_w, max_w = min(weights), max(weights)
    
    for u, v, data in G.edges(data=True):
        pos_u = G.nodes[u]["pos"]
        pos_v = G.nodes[v]["pos"]
        weight_val = data["weight"]
        normalized = 255 * (weight_val - min_w) / (max_w - min_w) if max_w != min_w else 127
        pt1 = (int(pos_u[0]), int(pos_u[1]))
        pt2 = (int(pos_v[0]), int(pos_v[1]))
        cv2.line(heatmap_img, pt1, pt2, color=float(normalized), thickness=line_thickness)
    
    heatmap_uint8 = np.clip(heatmap_img, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_path = "static/fire_heatmap.png"
    cv2.imwrite(heatmap_path, heatmap_color)
    print(f"Fire heatmap saved as {heatmap_path}")
    return heatmap_color

##########################################
#               Main Script              #
##########################################

if __name__ == "__main__":
    # Expect one or two command-line arguments: place name and optional fire mask file.
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: python combined_fire_map.py "Place Name" [fire_mask_file]'}))
        sys.exit(1)
    
    place_name = sys.argv[1].strip()
    fire_mask_file = sys.argv[2].strip() if len(sys.argv) >= 3 else None

    try:
        # Use OSMnx to geocode the place name.
        location_data = ox.geocode(place_name)
        lat, lon = location_data
    except Exception as e:
        print(json.dumps({'error': f'Failed to geocode place name: {str(e)}'}))
        sys.exit(1)

    start = time.time()
    
    # Create street and intersection maps.
    intersections, bounds, road_image_size, road_path, inter_path, overlay_path = create_street_and_intersection_maps(
        lat, lon, image_size=(800,800), dist=5000, intersection_radius=3, place=place_name.replace(" ", "_")
    )
    
    if intersections is None:
        print(json.dumps({'error': 'Failed to generate street maps.'}))
        sys.exit(1)
    
    # Fetch street network again for graph creation.
    edges, nodes, _ = fetch_street_network(lat, lon, dist=5000)
    geo_to_pixel = create_coordinate_transformer(bounds, road_image_size, scale=1.0)
    graph_json_file = os.path.join("static", f"{place_name.replace(' ', '_')}_graph.json")
    street_graph = create_graph_from_streets(intersections, edges, geo_to_pixel, output_file=graph_json_file)
    
    fire_heatmap_path = None
    # If a fire mask file was provided, update edge weights and generate heatmap.
    if fire_mask_file:
        fire_mask = cv2.imread(fire_mask_file, cv2.IMREAD_GRAYSCALE)
        if fire_mask is None:
            print(json.dumps({'error': f'Fire mask image not found: {fire_mask_file}'}))
            sys.exit(1)
        street_graph = update_edge_weights_with_fire(
            street_graph, fire_mask, road_image_size=road_image_size, fire_mask_size=(500,500),
            sigma=20, beta=1
        )
        heatmap_img = generate_fire_heatmap(street_graph, road_image_size=road_image_size, line_thickness=2)
        fire_heatmap_path = "static/fire_heatmap.png"
    
    end = time.time()
    execution_time = round(end - start, 2)
    
    # Prepare output JSON.
    output = {
        "place_name": place_name,
        "latitude": lat,
        "longitude": lon,
        "execution_time_sec": execution_time,
        "map_images": {
            "road_image": road_path,
            "intersection_image": inter_path,
            "overlay_image": overlay_path
        },
        "graph_json": graph_json_file
    }
    if fire_heatmap_path:
        output["fire_heatmap"] = fire_heatmap_path
    
    print(json.dumps(output))
