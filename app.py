from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
import subprocess
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-coordinates', methods=['POST'])
def get_coordinates():
    try:
        data = request.get_json()
        if not data or 'place_name' not in data:
            return jsonify({'error': 'Missing place name'}), 400

        place_name = data['place_name']

        # Run the subprocess
        result = subprocess.run(
            ['python', 'weighted_intersection_graph.py', place_name],
            capture_output=True,
            text=True
        )

        print("Subprocess Output (stdout):", result.stdout)  # Log stdout
        print("Subprocess Error (stderr):", result.stderr)    # Log stderr

        if result.returncode != 0:
            return jsonify({'error': 'Error processing location', 'details': result.stderr}), 500

        output_data = json.loads(result.stdout)

        if 'error' in output_data:
            return jsonify(output_data), 400

        return jsonify(output_data)

    except json.JSONDecodeError as e:
        return jsonify({'error': 'Invalid JSON output from subprocess', 'details': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<filename>')
def get_map_image(filename):
    response = make_response(send_from_directory("static", filename))
    # response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    # response.headers["Pragma"] = "no-cache"
    # response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
