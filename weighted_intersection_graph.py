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

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib.cm as cm

def graph_to_heatmap_image(G, blur_kernel_size=(15, 15), blur_sigma=0):
    """
    Create a heatmap image from a graph by weighting edges according to their proximity
    to fire. Edges that come closer to fire (as indicated by the fire mask) will be colored
    with a “hotter” color. A Gaussian blur is applied to the entire image before returning.

    Parameters:
        G : networkx.Graph
            A graph where each node has a 'pos' attribute (a tuple (x, y)) indicating its pixel location.
        fire_mask : np.ndarray
            A binary NumPy array of shape (500, 500) with 1 indicating fire and 0 indicating no fire.
        blur_kernel_size : tuple (default (15, 15))
            The kernel size for the Gaussian blur (both values should be odd numbers).
        blur_sigma : float (default 0)
            The sigma for the Gaussian blur. If 0, OpenCV will calculate it from the kernel size.
            
    Returns:
        blurred_img : np.ndarray
            An image (a NumPy array) of the drawn heatmap with Gaussian blur applied.
    """
    import numpy as np
    from PIL import Image

    def extract_bw_array_from_jpg(image_path, threshold=128):
        """
        Load an image, convert it to grayscale, and then produce a binary
        numpy array (with values 1 and 0) based on a given threshold.

        Parameters:
            image_path (str): Path to the input JPG image.
            threshold (int): Grayscale threshold (0-255) to decide between black and white.
                            Default is 128.

        Returns:
            np.ndarray: 2D numpy array of 1s and 0s.
        """
        # Open the image and convert to grayscale ('L' mode)
        with Image.open(image_path) as img:
            gray_img = img.convert('L')
        
        # Convert the grayscale image to a numpy array
        gray_array = np.array(gray_img)
        
        # Apply threshold: pixels >= threshold become 1, else 0.
        bw_array = (gray_array >= threshold).astype(np.uint8)
        
        return bw_array

    fire_mask = extract_bw_array_from_jpg("goated_mask.jpg")

    # Compute the distance transform.
    # Since fire_mask==1 indicates fire, we pass (1 - fire_mask) so that fire pixels become 0.
    dt = distance_transform_edt(1 - fire_mask)
    dt_max = dt.max() if dt.max() > 0 else 1  # prevent division by zero

    # Get image dimensions from the fire_mask
    height, width = fire_mask.shape
    # Create a blank color image (black background)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Process each edge in the graph
    for u, v in G.edges():
        pos1 = G.nodes[u]['pos']  # expected as (x, y)
        pos2 = G.nodes[v]['pos']  # expected as (x, y)

        # Determine the number of sample points along the edge based on its length.
        num_samples = int(np.hypot(pos2[0] - pos1[0], pos2[1] - pos1[1])) + 1
        num_samples = max(num_samples, 2)  # ensure at least two points

        # Create linearly spaced points along the edge.
        x_coords = np.linspace(pos1[0], pos2[0], num_samples)
        y_coords = np.linspace(pos1[1], pos2[1], num_samples)

        # For each sample point, get the distance transform value.
        dt_values = []
        for x, y in zip(x_coords, y_coords):
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                dt_values.append(dt[yi, xi])  # note: dt is indexed as [row, col] i.e. [y, x]
            else:
                dt_values.append(dt_max)  # if outside, assume maximum distance

        # The edge’s proximity to fire is given by the smallest distance along the edge.
        d_min = min(dt_values)
        # Normalize the intensity so that an edge passing through fire (d_min=0) gets intensity 1.
        intensity = 1 - (d_min / dt_max)
        intensity = np.clip(intensity, 0, 1)

        # Use a colormap to convert the intensity into a color.
        # Here we use Matplotlib’s 'hot' colormap.
        cmap = cm.get_cmap("hot")
        r, g, b, _ = cmap(intensity)  # returns values in 0-1
        # Convert to 0-255 and switch to BGR order for OpenCV.
        color = (int(b * 255), int(g * 255), int(r * 255))

        # Define the endpoints of the edge as integer pixel positions.
        pt1 = (int(round(pos1[0])), int(round(pos1[1])))
        pt2 = (int(round(pos2[0])), int(round(pos2[1])))

        # Draw the edge on the image.
        cv2.line(img, pt1, pt2, color, thickness=2)

        # (Optional: You might also want to store the computed intensity as an edge attribute, e.g.:
        # G.edges[u, v]['fire_intensity'] = intensity)

    # Apply a Gaussian blur to the entire image.
    blurred_img = cv2.GaussianBlur(img, blur_kernel_size, blur_sigma)

    success = cv2.imwrite("test.jpg", blurred_img)
    if not success:
        raise IOError(f"Could not write image to")

    return blurred_img

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
    # Create empty images: one grayscale and one in color (BGR)
    road_img = np.zeros(image_size, dtype=np.uint8)
    overlay_img = np.zeros((*image_size, 3), dtype=np.uint8)
    
    # Define a thickness map (you can adjust these values)
    thickness_map = {
        "motorway": 4,
        "trunk": 3,
        "primary": 2,
        "secondary": 2,
        "tertiary": 1,
        "residential": 1,
        "unclassified": 1,
        "service": 1
    }
    
    # Define a color map for the overlay image (BGR format)
    color_map = {
        "motorway": (0, 0, 255),      # Red
        "trunk": (0, 165, 255),       # Orange
        "primary": (0, 255, 0),       # Green
        "secondary": (255, 255, 0),   # Cyan-ish
        "tertiary": (255, 0, 255),    # Magenta
        "residential": (200, 200, 200),  # Light gray
        "unclassified": (150, 150, 150), # Medium gray
        "service": (100, 100, 100)       # Dark gray
    }
    
    for _, row in edges.iterrows():
        # Convert geographic coordinates to pixel coordinates
        coords = np.array(
            [geo_to_pixel(lon, lat) for lon, lat in row.geometry.coords],
            dtype=np.int32
        )
        
        # Get the highway type (sometimes a list; take the first element if so)
        highway_type = row.get("highway", "unclassified")
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        
        # Look up thickness and color; use defaults if not found
        line_thickness = thickness_map.get(highway_type, thickness_map["unclassified"])
        line_color = color_map.get(highway_type, color_map["unclassified"])
        
        # For the grayscale image, convert the BGR color to a grayscale intensity.
        # One simple way is to take the average (or you could use a weighted conversion).
        gray_intensity = int(sum(line_color) / 3)
        
        # Draw on the grayscale image
        cv2.polylines(road_img, [coords], isClosed=False, color=gray_intensity, thickness=line_thickness)
        # Draw on the color overlay image
        cv2.polylines(overlay_img, [coords], isClosed=False, color=line_color, thickness=line_thickness)

    
    return road_img, overlay_img


def save_images(road_img, intersection_img, overlay_img, dir="static/", place=""):
    rpath = f"{dir}{place}_road.png"
    ipath = f"{dir}{place}_intersect.png"
    opath = f"{dir}{place}_overlay.png"
    cv2.imwrite(rpath, road_img)
    cv2.imwrite(ipath, intersection_img)
    cv2.imwrite(opath, overlay_img)
    return rpath, ipath, opath

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
