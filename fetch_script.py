import osmnx as ox
import numpy as np
from PIL import Image, ImageDraw

def create_street_and_intersection_maps(lat, lon, image_size=(800, 800), dist=1000, scale=1.0, intersection_radius=5):
    """
    Creates and saves three images:
    1. A road map where streets are white on a black background.
    2. An intersection map where intersections are white dots on a black background.
    3. An overlay map where streets are white, intersections are bright red and large.

    Parameters:
    lat (float): Latitude of the center point.
    lon (float): Longitude of the center point.
    image_size (tuple): The size of the output images (width, height).
    dist (int): The distance in meters to search around the point.
    scale (float): Scaling factor for the map.
    intersection_radius (int): The size of the intersection dots in pixels.

    Returns:
    list: List of intersection coordinates (pixel coordinates).
    """
    try:
        # Get the street network from the point
        G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
        
        # Get edges and nodes as GeoDataFrames
        edges = ox.graph_to_gdfs(G, nodes=False)
        nodes = ox.graph_to_gdfs(G, edges=False)

        # Get the bounding box
        bounds = edges.total_bounds  # [minx, miny, maxx, maxy]

        # Create blank images
        road_img = np.zeros(image_size, dtype=np.uint8)  # Black background for roads
        intersection_img = np.zeros(image_size, dtype=np.uint8)  # Black background for intersections
        overlay_img = np.zeros((*image_size, 3), dtype=np.uint8)  # RGB image for overlay

        # Convert geographic coordinates to pixel coordinates
        def geo_to_pixel(lon, lat):
            x = int((lon - bounds[0]) / (bounds[2] - bounds[0]) * (image_size[0] - 1) * scale)
            y = int((bounds[3] - lat) / (bounds[3] - bounds[1]) * (image_size[1] - 1) * scale)
            return x, y

        # Draw the streets on the road and overlay images
        for _, row in edges.iterrows():
            coords = row.geometry.coords
            pixel_coords = [geo_to_pixel(lon, lat) for lon, lat in coords]

            # Draw lines between consecutive points
            for i in range(len(pixel_coords) - 1):
                x1, y1 = pixel_coords[i]
                x2, y2 = pixel_coords[i + 1]

                # Create a smooth line between points
                num_points = max(abs(x2 - x1), abs(y2 - y1)) * 2
                x_coords = np.linspace(x1, x2, num=int(num_points))
                y_coords = np.linspace(y1, y2, num=int(num_points))

                # Clip to image boundaries
                x_coords = np.clip(x_coords.astype(int), 0, image_size[0] - 1)
                y_coords = np.clip(y_coords.astype(int), 0, image_size[1] - 1)

                # Draw roads in white
                road_img[y_coords, x_coords] = 255
                overlay_img[y_coords, x_coords] = [255, 255, 255]  # White roads on overlay

        # Convert overlay image to PIL for drawing
        overlay_pil = Image.fromarray(overlay_img)
        draw = ImageDraw.Draw(overlay_pil)

        # Extract intersections and mark them in the intersection and overlay images
        intersections = []
        for _, row in nodes.iterrows():
            lon, lat = row.geometry.x, row.geometry.y
            x, y = geo_to_pixel(lon, lat)
            intersections.append((x, y))

            # Draw a larger bright red dot for the intersection
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                intersection_img[y, x] = 255  # Mark intersection in white
                
                # Draw red circle for intersections
                draw.ellipse(
                    (x - intersection_radius, y - intersection_radius, x + intersection_radius, y + intersection_radius),
                    fill=(255, 0, 0), outline=(255, 0, 0)
                )

        # Convert numpy images to PIL images
        road_map = Image.fromarray(road_img)
        intersection_map = Image.fromarray(intersection_img)

        # Save images
        road_map.save("road_map.png")
        intersection_map.save("intersection_map.png")
        overlay_pil.save("overlay.png")  # Save the PIL image with red dots

        print("Saved road_map.png, intersection_map.png, and overlay.png successfully.")

        return intersections

    except Exception as e:
        print(f"Error generating maps: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    lat, lon = 40.4444, -79.9532  # Example coordinates (Pittsburgh)
    
    intersections = create_street_and_intersection_maps(lat, lon, intersection_radius=7)
    
    if intersections:
        print(f"Extracted {len(intersections)} intersections.")
