from flask import Flask, render_template, request, jsonify
import time
import traceback

# Uncomment and adjust this import once you have your actual processing function.
# from your_script import run_graph_generator

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/points')
def get_points():
    # Replace this dummy data with your actual simulation output.
    # Here each point has an id, latitude, and longitude.
    points = [
        {"id": 1, "lat": 40.7138, "lng": -74.0050},
        {"id": 2, "lat": 40.7118, "lng": -74.0070},
        {"id": 3, "lat": 40.7125, "lng": -74.0040}
    ]
    return jsonify(points)

@app.route('/process', methods=['POST'])
def process():
    try:
        # Get the JSON data from the request and convert coordinates to floats.
        data = request.get_json()
        lat = float(data.get('lat'))
        lng = float(data.get('lng'))
    except (TypeError, ValueError) as e:
        # Return an error response if the conversion fails.
        return jsonify({'error': 'Invalid input for latitude and/or longitude.'}), 400

    # Optionally, validate that coordinates are within valid ranges.
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return jsonify({'error': 'Coordinates out of range.'}), 400

    try:
        # If you have a long-running processing function, consider running it asynchronously.
        # For now, we simulate processing with a sleep call.
        # Replace the time.sleep() call with your actual processing function call.
        # For example: result = run_graph_generator(lat, lng)
        time.sleep(3)  # Simulate processing delay

        # Dummy result structure; update this with the actual output from your processing function.
        result = {
            'vertices': [
                {'id': 1, 'lat': lat + 0.001, 'lng': lng + 0.001},
                {'id': 2, 'lat': lat - 0.001, 'lng': lng - 0.001},
                # Add more vertices as needed...
            ],
            'edges': [
                {'source': 1, 'target': 2, 'weight': 1},
                # Add more edges as needed...
            ]
        }
        return jsonify(result)
    except Exception as e:
        # Log the error and return an error response.
        print("Error processing request:", e)
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during processing.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
