import sys
import json
import os
import osmnx as ox
import matplotlib.pyplot as plt

def get_coordinates_from_place(place_name):
    """Convert a place name to latitude and longitude using osmnx."""
    try:
        location = ox.geocode(place_name)
        return {'latitude': location[0], 'longitude': location[1]}
    except Exception as e:
        return {'error': str(e)}

def generate_osm_map(place_name, latitude, longitude, zoom=14, size=(600, 400)):
    """Generate a static map image using osmnx and save it as a file."""
    try:
        # Ensure the static directory exists
        static_dir = "static"
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # Generate the map
        graph = ox.graph_from_point((latitude, longitude), dist=1000, network_type='all')
        fig, ax = ox.plot_graph(graph, bgcolor='white', node_size=10, edge_linewidth=0.5, show=False, close=False)

        # Save map as an image
        map_filename = f"map_{latitude}_{longitude}.png"
        map_path = os.path.join(static_dir, map_filename)
        plt.savefig(map_path, dpi=100, bbox_inches='tight')
        plt.close()

        return map_filename  # Return filename (not full path)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(json.dumps({'error': 'Invalid arguments'}))
        sys.exit(1)

    place_name = sys.argv[1].strip()  # Remove extra spaces
    location_data = get_coordinates_from_place(place_name)

    if 'error' in location_data:
        print(json.dumps(location_data))
        sys.exit(1)

    latitude = location_data['latitude']
    longitude = location_data['longitude']
    map_filename = generate_osm_map(place_name, latitude, longitude)

    if map_filename.startswith("Error"):
        print(json.dumps({'error': map_filename}))
        sys.exit(1)

    result = {
        'place_name': place_name,
        'latitude': latitude,
        'longitude': longitude,
        'message': f'Coordinates found for {place_name}',
        'map_image': f"/static/{map_filename}"  # Corrected path for Flask
    }

    print(json.dumps(result))
