import osmnx as ox
import numpy as np
from PIL import Image, ImageDraw

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
    G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
    nodes = ox.graph_to_gdfs(G, edges=False)
    bounds = edges.total_bounds
    return edges, nodes, bounds

def create_coordinate_transformer(bounds, image_size, scale=1.0):
    """
    Creates a function to transform geographic coordinates to pixel coordinates.
    
    Parameters:
        bounds: The geographic bounds [minx, miny, maxx, maxy]
        image_size (tuple): The target image size (width, height)
        scale (float): Scaling factor for the map
        
    Returns:
        function: A function that converts (lon, lat) to (x, y) pixels
    """
    def geo_to_pixel(lon, lat):
        x = int((lon - bounds[0]) / (bounds[2] - bounds[0]) * (image_size[0] - 1) * scale)
        y = int((bounds[3] - lat) / (bounds[3] - bounds[1]) * (image_size[1] - 1) * scale)
        return x, y
    return geo_to_pixel

def draw_streets(edges, geo_to_pixel, image_size):
    """
    Creates images with streets drawn on them.
    
    Parameters:
        edges: GeoDataFrame containing street data
        geo_to_pixel: Function to convert coordinates
        image_size (tuple): Size of the output images
        
    Returns:
        tuple: (road_img array, overlay_img array)
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
    Creates intersection map and marks intersections on overlay.
    
    Parameters:
        nodes: GeoDataFrame containing intersection data
        geo_to_pixel: Function to convert coordinates
        image_size (tuple): Size of the output images
        intersection_radius (int): Size of intersection markers
        
    Returns:
        tuple: (intersection_img array, list of intersection coordinates)
    """
    intersection_img = np.zeros(image_size, dtype=np.uint8)
    intersections = []
    
    for _, row in nodes.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        x, y = geo_to_pixel(lon, lat)
        intersections.append((x, y))
        
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            intersection_img[y, x] = 255
    
    return intersection_img, intersections

def draw_overlay_intersections(overlay_img, intersections, intersection_radius):
    """
    Draws intersection markers on the overlay image.
    
    Parameters:
        overlay_img: numpy array of the overlay image
        intersections: list of intersection coordinates
        intersection_radius (int): Size of intersection markers
        
    Returns:
        PIL.Image: Overlay image with intersection markers
    """
    overlay_pil = Image.fromarray(overlay_img)
    draw = ImageDraw.Draw(overlay_pil)
    
    for x, y in intersections:
        draw.ellipse(
            (x - intersection_radius, y - intersection_radius, 
             x + intersection_radius, y + intersection_radius),
            fill=(255, 0, 0), outline=(255, 0, 0)
        )
    
    return overlay_pil

def save_images(road_img, intersection_img, overlay_img):
    """
    Saves the generated images to files.
    
    Parameters:
        road_img: numpy array of road map
        intersection_img: numpy array of intersection map
        overlay_img: PIL Image of overlay
    """
    Image.fromarray(road_img).save("road_map.png")
    Image.fromarray(intersection_img).save("intersection_map.png")
    overlay_img.save("overlay.png")
    print("Saved road_map.png, intersection_map.png, and overlay.png successfully.")

def create_street_and_intersection_maps(lat, lon, image_size=(800, 800), dist=1000, scale=1.0, intersection_radius=5):
    """
    Creates and saves three images: road map, intersection map, and overlay map.
    
    Parameters:
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        image_size (tuple): Size of the output images (width, height)
        dist (int): Distance in meters to search around the point
        scale (float): Scaling factor for the map
        intersection_radius (int): Size of intersection markers in pixels
        
    Returns:
        list: List of intersection coordinates (pixel coordinates)
    """
    try:
        # Fetch network data
        edges, nodes, bounds = fetch_street_network(lat, lon, dist)
        
        # Create coordinate transformer
        geo_to_pixel = create_coordinate_transformer(bounds, image_size, scale)
        
        # Generate base images
        road_img, overlay_img = draw_streets(edges, geo_to_pixel, image_size)
        
        # Mark intersections
        intersection_img, intersections = mark_intersections(
            nodes, geo_to_pixel, image_size, intersection_radius
        )
        
        # Create final overlay
        overlay_pil = draw_overlay_intersections(
            overlay_img, intersections, intersection_radius
        )
        
        # Save all images
        save_images(road_img, intersection_img, overlay_pil)
        
        return intersections
        
    except Exception as e:
        print(f"Error generating maps: {str(e)}")
        return None

if __name__ == "__main__":
    # lat, lon = 40.4444, -79.9532  # Example coordinates (Pittsburgh)
    lat, lon = 34.0204789,-118.4117326
    intersections = create_street_and_intersection_maps(lat, lon, intersection_radius=3, dist=130000)
    if intersections:
        print(f"Extracted {len(intersections)} intersections.")