import osmnx as ox
import numpy as np
import cv2

def fetch_street_network(lat, lon, dist=1000):
    """
    Fetches the street network data for a given location.
    
    Parameters:
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        dist (int): The distance in meters to search around the point
        
    Returns:
        tuple: (edges GeoDataFrame, nodes GeoDataFrame, bounds)
    """
    G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive', simplify=True, retain_all=False)
    edges = ox.graph_to_gdfs(G, nodes=False)
    nodes = ox.graph_to_gdfs(G, edges=False)
    bounds = edges.total_bounds
    return edges, nodes, bounds

def create_coordinate_transformer(bounds, image_size, scale=1.0):
    """
    Creates a function to transform geographic coordinates to pixel coordinates.
    """
    def geo_to_pixel(lon, lat):
        x = int((lon - bounds[0]) / (bounds[2] - bounds[0]) * (image_size[0] - 1) * scale)
        y = int((bounds[3] - lat) / (bounds[3] - bounds[1]) * (image_size[1] - 1) * scale)
        return x, y
    return geo_to_pixel

def draw_streets(edges, geo_to_pixel, image_size):
    """
    Uses OpenCV to draw the street map for faster rendering.
    """
    road_img = np.zeros(image_size, dtype=np.uint8)

    for _, row in edges.iterrows():
        coords = np.array([geo_to_pixel(lon, lat) for lon, lat in row.geometry.coords], dtype=np.int32)
        cv2.polylines(road_img, [coords], isClosed=False, color=255, thickness=1)

    return road_img

def mark_intersections(nodes, geo_to_pixel, image_size, intersection_radius=5):
    """
    Uses vectorized NumPy operations to mark intersections.
    """
    intersection_img = np.zeros(image_size, dtype=np.uint8)
    coords = np.array([geo_to_pixel(row.geometry.x, row.geometry.y) for _, row in nodes.iterrows()])

    for x, y in coords:
        cv2.circle(intersection_img, (x, y), intersection_radius, 255, -1)

    return intersection_img, coords

def create_overlay(road_img, intersection_img):
    """
    Creates an overlay image combining roads and intersections.
    """
    overlay_img = cv2.merge([road_img, road_img, road_img])
    overlay_img[intersection_img > 0] = [0, 0, 255]  # Red for intersections
    return overlay_img

def save_images(road_img, intersection_img, overlay_img):
    """
    Saves the generated images using OpenCV.
    """
    cv2.imwrite("road_map.png", road_img)
    cv2.imwrite("intersection_map.png", intersection_img)
    cv2.imwrite("overlay.png", overlay_img)
    print("Saved road_map.png, intersection_map.png, and overlay.png successfully.")

def create_street_and_intersection_maps(lat, lon, image_size=(800, 800), dist=1000, scale=1.0, intersection_radius=5):
    """
    Generates and saves street, intersection, and overlay maps.
    """
    try:
        edges, nodes, bounds = fetch_street_network(lat, lon, dist)
        geo_to_pixel = create_coordinate_transformer(bounds, image_size, scale)

        road_img = draw_streets(edges, geo_to_pixel, image_size)
        intersection_img, intersections = mark_intersections(nodes, geo_to_pixel, image_size, intersection_radius)
        overlay_img = create_overlay(road_img, intersection_img)

        save_images(road_img, intersection_img, overlay_img)
        return intersections
        
    except Exception as e:
        print(f"Error generating maps: {str(e)}")
        return None

if __name__ == "__main__":
    lat, lon = 42.3143359, -71.0525965
    intersections = create_street_and_intersection_maps(lat, lon, intersection_radius=3, dist=10000)
    if intersections is not None:
        print(f"Extracted {len(intersections)} intersections.")
