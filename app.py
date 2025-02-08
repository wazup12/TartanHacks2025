from flask import Flask, request, jsonify, render_template, send_file
import subprocess
import json

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

        result = subprocess.run(
            ['python', 'process_coordinates.py', place_name],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return jsonify({'error': 'Error processing location', 'details': result.stderr}), 500

        output_data = json.loads(result.stdout)

        if 'error' in output_data:
            return jsonify(output_data), 400

        return jsonify(output_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<filename>')
def get_map_image(filename):
    return send_file(f"static/{filename}", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

