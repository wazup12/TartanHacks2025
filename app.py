#!/usr/bin/env python3
"""
Flask app for fire simulation.

This app accepts POST requests at /fire_sim with a JSON payload containing "lat" and "lon".
It then runs the fire simulation for the provided coordinates and returns a reduced simulation
result as a JSON response.
"""

import json
import numpy as np
import requests
import torch
from urllib3 import Retry
import cv2
from requests.adapters import HTTPAdapter
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------------------
# Global Model Loading (runs once at startup)
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "DPT_Hybrid"
print("Loading MiDaS model globally...")
global_midas = torch.hub.load("intel-isl/MiDaS", model_type)
global_midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
global_midas.to(device)
global_midas.eval()
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    global_transform = global_midas_transforms.dpt_transform
else:
    global_transform = global_midas_transforms.small_transform


# ------------------------------------------------------------------------------
# Step 1: Download Satellite Image (with retries)
# ------------------------------------------------------------------------------
def download_satellite_image_in_memory(lat, lon, zoom):
    """
    Downloads a 500x500 satellite image from Google Static Maps for the given coordinates.
    Returns the image as a NumPy array in BGR format.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(zoom),
        "size": "500x500",
        "maptype": "satellite",
        "key": "AIzaSyBk9BvEcF71-uEpdsrqML_0lqrddsGqoV0",
    }
    print(f"Requesting satellite image for lat: {lat:.6f}, lon: {lon:.6f}...")

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504, 429],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.get(base_url, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error downloading satellite image: {e}")

    image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode satellite image.")
    return image


# ------------------------------------------------------------------------------
# Step 2: Compute Depth (Gradient) Map via MiDaS
# ------------------------------------------------------------------------------
def compute_depth_map_from_image(img, midas, transform, device):
    """
    Given an image (as a NumPy array in BGR), compute its depth map using MiDaS.
    Returns the depth map as a NumPy array.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map


# ------------------------------------------------------------------------------
# Step 3: Fire Spread Simulation
# ------------------------------------------------------------------------------
def shift_with_zero_padding(arr, shift_row, shift_col):
    """
    Shifts a 2D array by (shift_row, shift_col) without wrapping (new areas filled with 0).
    """
    result = np.zeros_like(arr)
    if shift_row >= 0:
        row_start_src = 0
        row_end_src = arr.shape[0] - shift_row
        row_start_dest = shift_row
        row_end_dest = arr.shape[0]
    else:
        row_start_src = -shift_row
        row_end_src = arr.shape[0]
        row_start_dest = 0
        row_end_dest = arr.shape[0] + shift_row
    if shift_col >= 0:
        col_start_src = 0
        col_end_src = arr.shape[1] - shift_col
        col_start_dest = shift_col
        col_end_dest = arr.shape[1]
    else:
        col_start_src = -shift_col
        col_end_src = arr.shape[1]
        col_start_dest = 0
        col_end_dest = arr.shape[1] + shift_col
    result[row_start_dest:row_end_dest, col_start_dest:col_end_dest] = arr[
        row_start_src:row_end_src, col_start_src:col_end_src
    ]
    return result

def simulate_fire(
    wind_dir,
    wind_mag,
    slope_dir,
    gradient,
    grad_scale,
    steps=200,  # Increased from 100 to 200 for more time steps
    grid_size=500,
    p_base=0.5,  # Increased from 0.35 to 0.5 to make fire spread more aggressively
):
    """
    Runs the fire spread simulation on a grid using a cellular automata model.
    Returns:
      states: A list of 2D NumPy arrays (fire masks) for each time step.
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    grid[center, center] = 1

    neighbor_data = [
        {"offset": (-2, -2), "angle": 315},
        {"offset": (-2, 0), "angle": 0},
        {"offset": (-2, 2), "angle": 45},
        {"offset": (0, -2), "angle": 270},
        {"offset": (0, 2), "angle": 90},
        {"offset": (2, -2), "angle": 225},
        {"offset": (2, 0), "angle": 180},
        {"offset": (2, 2), "angle": 135},
    ]

    for d in neighbor_data:
        theta = d["angle"]
        wind_factor = 1 + wind_mag * np.cos(np.deg2rad(theta - wind_dir))
        slope_factor = 1 - grad_scale * gradient * np.cos(np.deg2rad(theta - slope_dir))
        p_dir = p_base * wind_factor * slope_factor
        d["p"] = np.clip(p_dir, 0, 1)

    states = [grid.copy()]
    for t in range(steps):
        prob_not_ignite = np.ones_like(grid, dtype=float)
        for d in neighbor_data:
            dr, dc = d["offset"]
            random_multiplier = np.random.uniform(0.3, 2.0, size=grid.shape)  # More randomness in fire spread
            effective_p_dir = np.clip(d["p"] * random_multiplier, 0, 1)
            shifted = shift_with_zero_padding(grid, dr, dc)
            factor = np.where(shifted == 1, 1 - effective_p_dir, 1.0)
            prob_not_ignite *= factor
        prob_ignite = 1 - prob_not_ignite
        random_vals = np.random.random(size=grid.shape)
        new_fires = (grid == 0) & (random_vals < prob_ignite)
        grid[new_fires] = 1
        states.append(grid.copy())
    return states


# def simulate_fire(
#     wind_dir,
#     wind_mag,
#     slope_dir,
#     gradient,
#     grad_scale,
#     steps=100,
#     grid_size=500,
#     p_base=0.35,
# ):
#     """
#     Runs the fire spread simulation on a grid using a cellular automata model.
#     Returns:
#       states: A list of 2D NumPy arrays (fire masks) for each time step.
#     """
#     grid = np.zeros((grid_size, grid_size), dtype=int)
#     center = grid_size // 2
#     grid[center, center] = 1

#     neighbor_data = [
#         {"offset": (-1, -1), "angle": 315},
#         {"offset": (-1, 0), "angle": 0},
#         {"offset": (-1, 1), "angle": 45},
#         {"offset": (0, -1), "angle": 270},
#         {"offset": (0, 1), "angle": 90},
#         {"offset": (1, -1), "angle": 225},
#         {"offset": (1, 0), "angle": 180},
#         {"offset": (1, 1), "angle": 135},
#     ]

#     for d in neighbor_data:
#         theta = d["angle"]
#         wind_factor = 1 + wind_mag * np.cos(np.deg2rad(theta - wind_dir))
#         slope_factor = 1 - grad_scale * gradient * np.cos(np.deg2rad(theta - slope_dir))
#         p_dir = p_base * wind_factor * slope_factor
#         d["p"] = np.clip(p_dir, 0, 1)

#     states = [grid.copy()]
#     for t in range(steps):
#         prob_not_ignite = np.ones_like(grid, dtype=float)
#         for d in neighbor_data:
#             dr, dc = d["offset"]
#             random_multiplier = np.random.uniform(0.5, 1.5, size=grid.shape)
#             effective_p_dir = np.clip(d["p"] * random_multiplier, 0, 1)
#             shifted = shift_with_zero_padding(grid, dr, dc)
#             factor = np.where(shifted == 1, 1 - effective_p_dir, 1.0)
#             prob_not_ignite *= factor
#         prob_ignite = 1 - prob_not_ignite
#         random_vals = np.random.random(size=grid.shape)
#         new_fires = (grid == 0) & (random_vals < prob_ignite)
#         grid[new_fires] = 1
#         states.append(grid.copy())
#     return states


# ------------------------------------------------------------------------------
# Generate Simulation JSON (with reduced output size)
# ------------------------------------------------------------------------------
def get_simulation_json_for_coords(lat, lon, wind_dir=45, wind_mag=0.5):
    """
    Takes in a latitude and longitude, runs the simulation for those coordinates,
    and returns a JSON string containing the new fire points at each time step for 20 steps.

    Modifications:
      - Runs for fewer time steps (steps=20) on a smaller grid (grid_size=200).
      - Outputs the new points at each time step (i.e. cells that switch from 0 to 1 relative to the previous step).
      - Only every 10th new point is included.
      - Each latitude and longitude is rounded to 6 decimal places.
    """
    print("Using global MiDaS model for simulation.")
    zoom = 18
    grid_size = 200
    steps = 20
    img = download_satellite_image_in_memory(lat, lon, zoom)

    print("Computing depth map...")
    depth_map = compute_depth_map_from_image(
        img, global_midas, global_transform, device
    )

    # Resize the depth map to match the simulation grid size.
    resized_gradient = cv2.resize(
        depth_map, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR
    )

    print("Running fire spread simulation...")
    slope_dir = 135
    grad_scale = 0.5
    p_base = 0.35
    states = simulate_fire(
        wind_dir,
        wind_mag,
        slope_dir,
        resized_gradient,
        grad_scale,
        steps,
        grid_size,
        p_base,
    )
    print(f"Simulation completed with {len(states)} time steps.")

    # Compute geographic resolution (meters per pixel).
    earth_radius = 6378137  # in meters
    resolution = (np.cos(np.deg2rad(lat)) * 2 * np.pi * earth_radius) / (
        256 * (2**zoom)
    )
    center = grid_size // 2

    time_series_positions = []
    # For each time step (1 to steps), compute the new fire points relative to the previous state.
    for t in range(1, len(states)):
        prev_state = states[t - 1]
        curr_state = states[t]
        new_fire_mask = (curr_state == 1) & (prev_state == 0)
        new_indices = np.nonzero(new_fire_mask)
        new_positions = []
        meter_to_deg_lat = resolution / 111320.0
        meter_to_deg_lon = resolution / (111320.0 * np.cos(np.deg2rad(lat)))
        for i, j in zip(new_indices[0], new_indices[1]):
            cell_lat = lat + (center - i) * meter_to_deg_lat
            cell_lon = lon + (j - center) * meter_to_deg_lon
            new_positions.append([round(cell_lat, 6), round(cell_lon, 6)])
        time_series_positions.append(new_positions)

    result = {
        "simulation_steps": len(states),
        "time_series_positions": time_series_positions,
    }
    return json.dumps(result, indent=2)


# ------------------------------------------------------------------------------
# Flask Endpoint
# ------------------------------------------------------------------------------
@app.route("/fire_sim", methods=["POST"])
def fire_sim():
    """
    Expects a JSON payload with "lat" and "lon".
    Returns the simulation result as a JSON response.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    try:
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        mag = float(data.get("mag"))
        ang = float(data.get("ang"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing latitude/longitude values"}), 400

    try:
        simulation_json = get_simulation_json_for_coords(lat, lon, wind_dir=ang, wind_mag=mag)
    except Exception as e:
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500

    return Response(simulation_json, mimetype="application/json")


# ------------------------------------------------------------------------------
# Default route (for testing purposes)
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return (
        "Fire Simulation API<br>"
        "POST a JSON payload with 'lat' and 'lon' to /fire_sim to run the simulation."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
