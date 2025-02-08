import osmnx as ox
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
import time
import cv2
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor

# Define road importance mapping
ROAD_IMPORTANCE = {
    "motorway": 5, "trunk": 4, "primary": 3, "secondary": 2,
    "tertiary": 1, "residential": 0.5, "unclassified": 0.5, "service": 0.3
}

def fetch_street_network(lat, lon, dist=100):
    G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive', simplify=True)
    edges = ox.graph_to_gdfs(G, nodes=False)
    nodes = ox.graph_to_gdfs(G, edges=False)
    return edges, nodes, edges.total_bounds

def create_coordinate_transformer(bounds, image_size, scale=1.0):
    min_x, min_y, max_x, max_y = bounds
    width, height = image_size
    
    def geo_to_pixel(lon, lat):
        x = int((lon - min_x) / (max_x - min_x) * (width - 1) * scale)
        y = int((max_y - lat) / (max_y - min_y) * (height - 1) * scale)
        return x, y
    
    return geo_to_pixel

def mark_intersections(nodes, geo_to_pixel, image_size, intersection_radius=5):
    intersection_img = np.zeros(image_size, dtype=np.uint8)
    intersections = [(geo_to_pixel(row.geometry.x, row.geometry.y)) for _, row in nodes.iterrows()]
    
    for x, y in intersections:
        # mark in red
        cv2.circle(intersection_img, (x, y), intersection_radius, color=(255,0,0), thickness=-1)
    
    return intersection_img, intersections

def create_graph_from_streets(intersections, edges, geo_to_pixel, output_file="graph.json"):
    G = nx.Graph()
    intersection_array = np.array(intersections)
    tree = KDTree(intersection_array)  # Fast nearest neighbor search

    for idx, (x, y) in enumerate(intersections):
        G.add_node(idx, pos=(int(x), int(y)))
    
    for _, row in edges.iterrows():
        coords = np.array([geo_to_pixel(lon, lat) for lon, lat in row.geometry.coords])
        _, node_indices = tree.query(coords)
        unique_edges = set(zip(node_indices[:-1], node_indices[1:]))
        highway_type = row.get("highway", "unclassified")
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        importance_value = ROAD_IMPORTANCE.get(highway_type, 0.5)
        
        for node1, node2 in unique_edges:
            if node1 != node2:
                G.add_edge(int(node1), int(node2), weight=0, importance=importance_value)
    
    with open(output_file, "w") as f:
        json.dump({
            "nodes": {n: {"pos": G.nodes[n]["pos"]} for n in G.nodes},
            "edges": [(u, v, {"importance": G.edges[u, v]["importance"]}) for u, v in G.edges]
        }, f)
    
    return G

def draw_streets(edges, geo_to_pixel, image_size):
    road_img = np.zeros(image_size, dtype=np.uint8)  # Single-channel grayscale
    overlay_img = np.zeros((*image_size, 4), dtype=np.uint8)  # RGBA (4 channels)

    for _, row in edges.iterrows():
        coords = np.array([geo_to_pixel(lon, lat) for lon, lat in row.geometry.coords], dtype=np.int32)

        # Ensure the image is in a valid format for OpenCV
        if road_img.ndim == 2:  # Convert grayscale to 3-channel before drawing (optional)
            road_img = cv2.cvtColor(road_img, cv2.COLOR_GRAY2BGR)

        # Draw the streets (white lines) on both images
        cv2.polylines(road_img, [coords], isClosed=False, color=(255, 255, 255), thickness=1)
        cv2.polylines(overlay_img[:, :, :3], [coords], isClosed=False, color=(255, 255, 255), thickness=1)

    # Convert black to transparent in the overlay
    gray = cv2.cvtColor(overlay_img[:, :, :3], cv2.COLOR_BGR2GRAY)
    overlay_img[:, :, 3] = np.where(gray > 0, 255, 0)  # Set alpha for non-black pixels

    return road_img, overlay_img



# def draw_streets(edges, geo_to_pixel, image_size):
#     road_img = np.zeros(image_size, dtype=np.uint8)
#     overlay_img = np.zeros((*image_size, 3), dtype=np.uint8)
    
#     for _, row in edges.iterrows():
#         coords = np.array([geo_to_pixel(lon, lat) for lon, lat in row.geometry.coords], dtype=np.int32)
#         cv2.polylines(road_img, [coords], isClosed=False, color=255, thickness=1)
#         cv2.polylines(overlay_img, [coords], isClosed=False, color=(255, 255, 255), thickness=1)
    
#     return road_img, overlay_img

def save_images(road_img, intersection_img, overlay_img, dir="static/", place=""):
    rpath = f"{dir}{place}_road.png"
    ipath = f"{dir}{place}_intersect.png"
    opath = f"{dir}{place}_overlay.png"

    cv2.imwrite(rpath, road_img)
    cv2.imwrite(ipath, intersection_img)

    # Save overlay with transparency
    if overlay_img.shape[-1] == 4:  # Ensure RGBA format
        cv2.imwrite(opath, overlay_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(opath, overlay_img[:, :, :3])  # Fallback for RGB

    return rpath, ipath, opath

# def save_images(road_img, intersection_img, overlay_img, dir="static/", place=""):
#     rpath = f"{dir}{place}_road.png"
#     ipath = f"{dir}{place}_intersect.png"
#     opath = f"{dir}{place}_overlay.png"
#     cv2.imwrite(rpath, road_img)
#     cv2.imwrite(ipath, intersection_img)
#     cv2.imwrite(opath, overlay_img)
#     return rpath, ipath, opath

def create_street_and_intersection_maps(lat, lon, image_size=(800, 800), dist=1000, scale=1.0, intersection_radius=5, place=""):
    try:
        edges, nodes, bounds = fetch_street_network(lat, lon, dist)
        geo_to_pixel = create_coordinate_transformer(bounds, image_size, scale)

        with ThreadPoolExecutor() as executor:
            road_future = executor.submit(draw_streets, edges, geo_to_pixel, image_size)
            intersection_future = executor.submit(mark_intersections, nodes, geo_to_pixel, image_size, intersection_radius)
            road_img, _ = road_future.result()
            intersection_img, intersections = intersection_future.result()

        overlay_img = cv2.addWeighted(road_img, 1, intersection_img, 1, 0)
        a, b, c = save_images(road_img, intersection_img, overlay_img, place=place)
        return intersections, a, b, c
    except Exception as e:
        print(f"Error generating maps: {str(e)}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(json.dumps({'error': 'Invalid arguments'}))
        sys.exit(1)

    place_name = sys.argv[1].strip()
    location_data = ox.geocode(place_name)
    lat, lon = location_data
    
    start = time.time()
    a = create_street_and_intersection_maps(lat, lon, intersection_radius=3, dist=5000)
    if not a:
        print(json.dumps({'error': 'something went wrong :('}))

    intersections = a[0]
    image_filenames = a[1], a[2], a[3]
    
    if intersections:
        edges, nodes, bounds = fetch_street_network(lat, lon, dist=5000)
        geo_to_pixel = create_coordinate_transformer(bounds, (800, 800), scale=1.0)
        street_graph = create_graph_from_streets(intersections, edges, geo_to_pixel, output_file="street_graph.json")
    end = time.time()
    # plot_color_graph(street_graph)
    # image_filenames.append("street_graph.png")
    
    print(json.dumps({'place_name': place_name, 'latitude': lat, 'longitude': lon, 'execution_time': round(end - start, 2), 'map_images': image_filenames}))
