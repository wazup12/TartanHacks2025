import osmnx as ox
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_street_map(lat, lon, image_size=(800, 800), dist=10000):
    """
    Creates a binary image where streets are white and everything else is black.
    
    Parameters:
    lat (float): Latitude of the center point
    lon (float): Longitude of the center point
    image_size (tuple): The size of the output image (width, height)
    dist (int): The distance in meters to search around the point (creates a square with sides of 2*dist)
    
    Returns:
    PIL.Image: Binary image where streets are white (255) and background is black (0)
    """
    try:
        # Get the street network from the point
        G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
        
        # Get the edges coordinates
        edges = ox.graph_to_gdfs(G, nodes=False)
        
        # Get the bounding box
        bounds = edges.total_bounds
        
        # Create a blank image
        img = np.zeros(image_size, dtype=np.uint8)
        
        # Convert geographic coordinates to pixel coordinates
        def geo_to_pixel(lon, lat):
            x = int((lon - bounds[0]) / (bounds[2] - bounds[0]) * (image_size[0] - 1))
            y = int((bounds[3] - lat) / (bounds[3] - bounds[1]) * (image_size[1] - 1))
            return x, y
        
        # Draw the streets
        for _, row in edges.iterrows():
            coords = row.geometry.coords
            pixel_coords = []
            for lon, lat in coords:
                x, y = geo_to_pixel(lon, lat)
                pixel_coords.append((x, y))
            
            # Draw lines between consecutive points
            for i in range(len(pixel_coords) - 1):
                x1, y1 = pixel_coords[i]
                x2, y2 = pixel_coords[i + 1]
                
                # Use numpy's linspace to create a line
                num_points = max(abs(x2 - x1), abs(y2 - y1)) * 2
                x_coords = np.linspace(x1, x2, num=int(num_points))
                y_coords = np.linspace(y1, y2, num=int(num_points))
                
                # Convert to integers and clip to image boundaries
                x_coords = np.clip(x_coords.astype(int), 0, image_size[0] - 1)
                y_coords = np.clip(y_coords.astype(int), 0, image_size[1] - 1)
                
                # Draw the line
                img[y_coords, x_coords] = 255
        
        # Convert to PIL Image
        return Image.fromarray(img)
    
    except Exception as e:
        print(f"Error generating map: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    lat, lon = 40.4444, -79.9532
    
    street_map = create_street_map(lat, lon)
    
    if street_map:
        # Save the image
        street_map.save("street_map.png")
        
        # Display the image
        plt.imshow(street_map, cmap='binary')
        plt.axis('off')
        plt.show()