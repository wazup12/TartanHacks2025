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

def draw_streets(edges, geo_to_pixel, image_size):
    road_img = np.zeros(image_size, dtype=np.uint8)
    for _, row in edges.iterrows():
        coords = np.array([geo_to_pixel(lon, lat) for lon, lat in row.geometry.coords], dtype=np.int32)
        cv2.polylines(road_img, [coords], isClosed=False, color=255, thickness=1)
    return road_img

def save_mask(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(image_path, mask)

def save_images(road_img, dir="static/", place=""):
    rpath = f"{dir}{place}_road.png"
    cv2.imwrite(rpath, road_img)
    save_mask(rpath)
    return rpath

def create_street_map(lat, lon, image_size=(800, 800), dist=1000, scale=1.0, place=""):
    try:
        edges, _, bounds = fetch_street_network(lat, lon, dist)
        geo_to_pixel = create_coordinate_transformer(bounds, image_size, scale)
        
        road_img = draw_streets(edges, geo_to_pixel, image_size)
        road_path = save_images(road_img, place=place)
        return road_path
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
    road_image = create_street_map(lat, lon, place=place_name)
    end = time.time()
    
    print(json.dumps({'place_name': place_name, 'latitude': lat, 'longitude': lon, 'execution_time': round(end - start, 2), 'map_image': road_image}))
