#!/usr/bin/env python3
"""
Full Pipeline:
  1. Downloads a satellite image (500x500) from Google Static Maps.
  2. Computes its depth/gradient map from the image using MiDaS.
  3. Runs a fire spread simulation (cellular automata) that uses the gradient as slope.
  4. Samples 5 fire masks from the simulation and overlays smoke onto the satellite image.
  
Each satellite image (num_sat_points) produces 5 data points:
   - The burned satellite image (training data)
   - The corresponding fire mask (label)
These outputs are saved in separate directories.
"""

import os
import numpy as np
import cv2
import torch
import requests
import argparse
from io import BytesIO
from PIL import Image
from scipy.ndimage import gaussian_filter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random

# -------------------------------
# Step 1: Download Satellite Image (with retries)
# -------------------------------
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
        "key": "AIzaSyBk9BvEcF71-uEpdsrqML_0lqrddsGqoV0"
    }
    print(f"Requesting satellite image for lat: {lat:.6f}, lon: {lon:.6f}...")

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504, 429],
        allowed_methods=["GET"]
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

# -------------------------------
# Step 2: Compute Depth (Gradient) Map via MiDaS
# -------------------------------
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

# -------------------------------
# Step 3: Fire Spread Simulation
# -------------------------------
def shift_with_zero_padding(arr, shift_row, shift_col):
    """
    Shifts a 2D array by (shift_row, shift_col) without wrapping.
    New areas are filled with zeros.
    """
    result = np.zeros_like(arr)
    # Determine the row slices.
    if shift_row >= 0:
        src_row_start = 0
        src_row_end = arr.shape[0] - shift_row
        dst_row_start = shift_row
        dst_row_end = arr.shape[0]
    else:
        src_row_start = -shift_row
        src_row_end = arr.shape[0]
        dst_row_start = 0
        dst_row_end = arr.shape[0] + shift_row

    # Determine the column slices.
    if shift_col >= 0:
        src_col_start = 0
        src_col_end = arr.shape[1] - shift_col
        dst_col_start = shift_col
        dst_col_end = arr.shape[1]
    else:
        src_col_start = -shift_col
        src_col_end = arr.shape[1]
        dst_col_start = 0
        dst_col_end = arr.shape[1] + shift_col

    # Copy the valid part of the array into the result.
    result[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
        arr[src_row_start:src_row_end, src_col_start:src_col_end]
    return result

def simulate_fire(wind_dir, wind_mag, slope_dir, gradient, grad_scale, steps=100, grid_size=500, p_base=0.35):
    """
    Runs the fire spread simulation on a grid using a cellular automata model.
    
    Returns:
      states    : A list of 2D NumPy arrays (fire masks) for each time step.
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    grid[center, center] = 1

    neighbor_data = [
        {"offset": (-1, -1), "angle": 315},
        {"offset": (-1,  0), "angle": 0},
        {"offset": (-1,  1), "angle": 45},
        {"offset": ( 0, -1), "angle": 270},
        {"offset": ( 0,  1), "angle": 90},
        {"offset": ( 1, -1), "angle": 225},
        {"offset": ( 1,  0), "angle": 180},
        {"offset": ( 1,  1), "angle": 135},
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
            random_multiplier = np.random.uniform(0.5, 1.5, size=grid.shape)
            effective_p_dir = np.clip(d["p"] * random_multiplier, 0, 1)
            shifted = shift_with_zero_padding(grid, dr, dc)
            factor = np.where(shifted == 1, 1 - effective_p_dir, 1.0)
            prob_not_ignite *= factor
        prob_ignite = 1 - prob_not_ignite
        random_vals = np.random.random(size=grid.shape)
        new_fires = ((grid == 0) & (random_vals < prob_ignite))
        grid[new_fires] = 1
        states.append(grid.copy())
    return states

# -------------------------------
# Step 4: Overlay Smoke on the Satellite Image
# -------------------------------
def overlay_smoke(sat_img, fire_mask, sigma=35.0, opacity=0.97, smoke_color="100,80,80"):
    """
    Overlays smoke artifacts on the satellite image using the provided fire mask.
    
    Returns:
      blended_img : A PIL Image of the satellite image with the smoke overlay.
    """
    sat_array = np.array(sat_img).astype(np.float32)
    fire_mask = fire_mask.astype(np.float32)
    if fire_mask.max() > 1:
        fire_mask = fire_mask / 255.0

    smoke_mask = gaussian_filter(fire_mask, sigma=sigma)
    if smoke_mask.max() > 0:
        smoke_mask = smoke_mask / smoke_mask.max()
    noise = np.random.uniform(0, 0.2, size=smoke_mask.shape)
    smoke_mask = np.clip(smoke_mask + noise, 0, 1)

    smoke_color_arr = np.array([int(c) for c in smoke_color.split(",")], dtype=np.float32).reshape((1, 1, 3))
    smoke_mask_3ch = np.stack([smoke_mask] * 3, axis=-1)
    alpha = opacity * smoke_mask_3ch

    blended_array = (1 - alpha) * sat_array + alpha * smoke_color_arr
    blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
    blended_img = Image.fromarray(blended_array)
    return blended_img

# -------------------------------
# Full Pipeline: Combining All Steps
# -------------------------------
def full_pipeline(lat, lon, zoom, wind_dir, wind_mag, slope_dir, grad_scale,
                  steps, sigma, opacity, smoke_color, midas, transform, device):
    """
    Runs the full data generation pipeline:
      - Downloads the satellite image.
      - Computes its gradient (depth) map.
      - Simulates fire spread (producing many fire masks).
      - Samples 5 fire masks from the simulation and overlays smoke onto the satellite image.
    
    Returns:
      burned_images : A list of 5 burned images (PIL Images).
      fire_masks    : A list of 5 corresponding fire masks (NumPy arrays).
    """
    sat_bgr = download_satellite_image_in_memory(lat, lon, zoom)
    sat_rgb = cv2.cvtColor(sat_bgr, cv2.COLOR_BGR2RGB)
    sat_pil = Image.fromarray(sat_rgb)

    gradient = compute_depth_map_from_image(sat_bgr, midas, transform, device)
    if gradient.shape != (500, 500):
        gradient = cv2.resize(gradient, (500, 500))
    
    fire_states = simulate_fire(wind_dir, wind_mag, slope_dir, gradient, grad_scale,
                                steps=steps, grid_size=500, p_base=0.35)
    indices = np.linspace(0, len(fire_states) - 1, 5, dtype=int)

    burned_images = []
    fire_masks = []
    for idx in indices:
        mask = fire_states[idx]
        burned = overlay_smoke(sat_pil, mask, sigma=sigma, opacity=opacity, smoke_color=smoke_color)
        burned_images.append(burned)
        fire_masks.append(mask)
    return burned_images, fire_masks

# -------------------------------
# Main: Set Up and Run the Pipeline
# -------------------------------
def main():
    # parser = argparse.ArgumentParser(
    #     description="Full Pipeline: Generate burned satellite images with simulated fire spread."
    # )
    # parser.add_argument("--num_sat_points", type=int, default=1,
    #                     help="Number of satellite points to process (each yields 5 data points; default: 1).")
    # parser.add_argument("--center_lat", type=float, default=40.4447605,
    #                     help="Center latitude (default: 40.4447605).")
    # parser.add_argument("--center_lon", type=float, default=-79.9426024,
    #                     help="Center longitude (default: -79.9426024).")
    # parser.add_argument("--zoom", type=int, default=18,
    #                     help="Zoom level for satellite images (default: 18).")
    # parser.add_argument("--wind_dir", type=float, default=90.0,
    #                     help="Wind direction in degrees relative to north (default: 90).")
    # parser.add_argument("--wind_mag", type=float, default=0.4,
    #                     help="Wind magnitude (default: 0.4).")
    # parser.add_argument("--slope_dir", type=float, default=0.0,
    #                     help="Slope (uphill) direction in degrees (default: 0).")
    # parser.add_argument("--grad_scale", type=float, default=0.0025,
    #                     help="Gradient scaling factor for slope effect (default: 0.0025).")
    # parser.add_argument("--steps", type=int, default=500,
    #                     help="Number of time steps for fire simulation (default: 500).")
    # parser.add_argument("--sigma", type=float, default=35.0,
    #                     help="Sigma for Gaussian blur in smoke overlay (default: 35.0).")
    # parser.add_argument("--opacity", type=float, default=0.97,
    #                     help="Global opacity for smoke overlay (default: 0.97).")
    # parser.add_argument("--smoke_color", type=str, default="100,80,80",
    #                     help="Smoke color as comma-separated R,G,B (default: 100,80,80).")
    # parser.add_argument("--output_prefix", type=str, default="output",
    #                     help="Prefix for the output file names (default: 'output').")
    # parser.add_argument("--data_dir", type=str, default="training_data",
    #                     help="Directory where burned (training) images will be saved (default: 'training_data').")
    # parser.add_argument("--labels_dir", type=str, default="labels",
    #                     help="Directory where label files (fire masks) will be saved (default: 'labels').")
    # args = parser.parse_args()

    



    num_sat_points = 25
    center_lat = 40.4447605
    center_lon = -79.9426024
    zoom = 17 
    # wind_dir = 270.0
    # wind_mag = 0.4
    slope_dir = 0.0 
    grad_scale = 0.0025 
    steps = 250
    sigma = 35.0 
    opacity = 0.97 
    smoke_color = "150,150,150"
    output_prefix = "output1"
    data_dir = "training"
    labels_dir = "labels"

    # Create the output directories if they don't exist.
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Load the MiDaS model and transforms (only once)
    model_type = "DPT_Hybrid"
    print("Loading MiDaS model...")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    for i in range(num_sat_points):
        lat_offset = np.random.uniform(-0.01, 0.01)
        lon_offset = np.random.uniform(-0.01, 0.01)
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        print(f"\nProcessing satellite point {i+1}/{num_sat_points} at lat: {lat:.6f}, lon: {lon:.6f}")
        
        wind_mag = random.uniform(0.0, 1.0)
        wind_dir = random.uniform(0.0, 360.0)
        burned_images, fire_masks = full_pipeline(
            lat, lon, zoom,
            wind_dir, wind_mag,
            slope_dir, grad_scale,
            steps, sigma, opacity,
            smoke_color,
            midas, transform, device
        )
        
        # Save burned images (training data) and fire masks (labels) in separate directories.
        for j, (burned_img, mask) in enumerate(zip(burned_images, fire_masks)):
            burned_filename = os.path.join(data_dir, f"{output_prefix}_sat_{i}_burned_{j}.jpg")
            mask_filename   = os.path.join(labels_dir, f"{output_prefix}_sat_{i}_fire_mask_{j}.npy")
            burned_img.save(burned_filename)
            np.save(mask_filename, mask)
            print(f"Saved burned image: {burned_filename}")
            print(f"Saved fire mask: {mask_filename}")

if __name__ == "__main__":
    main()
