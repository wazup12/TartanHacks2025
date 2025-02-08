#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
import requests
import argparse

def download_satellite_image(lat, lon, zoom, output_file):
    """
    Downloads a 500x500 satellite image from Google Static Maps centered at the specified coordinate.

    Parameters:
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center point.
        zoom (int): Zoom level.
        output_file (str): Filename to save the downloaded image.
        api_key (str): Your Google Maps API key.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(zoom),
        "size": "500x500",
        "maptype": "satellite",
        "key": "AIzaSyBk9BvEcF71-uEpdsrqML_0lqrddsGqoV0"
        # Optionally, you can add "scale": "2" for a higher resolution image.
    }
    
    print(f"Requesting image for lat: {lat:.6f}, lon: {lon:.6f}...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Image saved as '{output_file}'.")
    else:
        print("Error downloading image:")
        print(f"Status code: {response.status_code}")
        print(response.text)

def compute_depth_map(image_file, output_file, midas, transform, device):
    """
    Reads an image file, uses the MiDaS model to estimate depth, and saves the depth map as a .npy file.
    
    Parameters:
        image_file (str): The satellite image file.
        output_file (str): Filename to save the depth map (NumPy file).
        midas: The loaded MiDaS model.
        transform: The corresponding image transform.
        device: Torch device to run the model on.
    """
    # Load and prepare the image.
    img = cv2.imread(image_file)
    if img is None:
        print(f"Error: Could not read {image_file}.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply the transform and prepare the tensor.
    input_batch = transform(img).to(device)
    
    # Run the MiDaS model.
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert the prediction to a NumPy array.
    depth_map = prediction.cpu().numpy()
    
    print(depth_map)
    # Save the depth map.
    np.save(output_file, depth_map)
    print(f"Depth map saved as '{output_file}'.")
    return depth_map

def generate_datapoints(num_points, center_lat, center_lon, zoom,
                        image_prefix="satellite", gradient_prefix="gradient"):
    """
    Generates a set of data points (satellite images and corresponding gradient maps) by:
      - Varying the location slightly around a center point.
      - Downloading the satellite image.
      - Running the MiDaS model to produce the depth and gradient map.
    
    Parameters:
        num_points (int): Number of data points to generate.
        center_lat (float): Central latitude for generating coordinates.
        center_lon (float): Central longitude for generating coordinates.
        zoom (int): Zoom level for the satellite images.
        api_key (str): Google Maps API key.
        image_prefix (str): Prefix for saved satellite images.
        gradient_prefix (str): Prefix for saved gradient maps.
    """
    # Load the MiDaS model and its transforms once.
    model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    for i in range(num_points):
        # Create a small random offset (in degrees) for lat and lon.
        lat_offset = np.random.uniform(-0.01, 0.01)
        lon_offset = np.random.uniform(-0.01, 0.01)
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        
        # Define filenames for the satellite image and gradient map.
        image_file = f"{image_prefix}_{i}.jpg"
        gradient_file = f"{gradient_prefix}_{i}.npy"
        
        # Download the satellite image.
        download_satellite_image(lat, lon, zoom, image_file)
        
        # Compute the gradient map from the downloaded image.
        compute_depth_map(image_file, gradient_file, midas, transform, device)

def main():
    # parser = argparse.ArgumentParser(
    #     description="Generate datapoints (satellite images and corresponding gradient maps)."
    # )
    # parser.add_argument("--num_points", type=int, default=10,
    #                     help="Number of datapoints to generate (default: 10).")
    # parser.add_argument("--center_lat", type=float, default=40.4447605,
    #                     help="Center latitude (default: 40.4447605).")
    # parser.add_argument("--center_lon", type=float, default=-79.9426024,
    #                     help="Center longitude (default: -79.9426024).")
    # parser.add_argument("--zoom", type=int, default=18,
    #                     help="Zoom level for satellite images (default: 18).")
    # parser.add_argument("--api_key", type=str, required=True,
    #                     help="Your Google Maps API key.")
    
    # args = parser.parse_args()

    num_points = 5
    center_lat = 40.4447605
    center_lon = -79.9426024
    zoom = 17    
    generate_datapoints(num_points, center_lat, center_lon, zoom)

if __name__ == "__main__":
    main()
