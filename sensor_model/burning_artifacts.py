#!/usr/bin/env python3
"""
This script overlays smoke artifacts on a satellite image using a fire mask.
The fire mask (a 500x500 numpy array) is first diffused using a Gaussian blur to 
simulate smoke diffusion, and random noise is added to create a turbulent appearance.
Then, using a specified smoke color and opacity, the smoke is blended onto the satellite image.
"""

import numpy as np
import argparse
from PIL import Image
from scipy.ndimage import gaussian_filter

def main():
    # parser = argparse.ArgumentParser(
    #     description="Overlay smoke artifacts on a satellite image using a fire mask."
    # )
    # parser.add_argument("--satellite", type=str, required=True,
    #                     help="Path to the satellite image (e.g., jpg or png).")
    # parser.add_argument("--fire_mask", type=str, required=True,
    #                     help="Path to the fire mask numpy file (.npy).")
    # parser.add_argument("--sigma", type=float, default=5.0,
    #                     help="Sigma value for the Gaussian blur (default: 5.0).")
    # parser.add_argument("--opacity", type=float, default=0.5,
    #                     help="Global opacity for the smoke overlay (default: 0.5, between 0 and 1).")
    # parser.add_argument("--smoke_color", type=str, default="200,200,200",
    #                     help="Comma-separated RGB values for the smoke color (default: 200,200,200).")
    # parser.add_argument("--output", type=str, default="satellite_with_smoke.jpg",
    #                     help="Filename for the output image.")
    # args = parser.parse_args()

    satellite = "satellite_1.jpg"
    fire_mask = "fire_mask_200.npy"
    sigma = 35.0 
    opacity = 0.97
    smoke_color = "100,80,80"
    output = "burn_test1.jpg"


    # Load the satellite image and convert to RGB.
    sat_img = Image.open(satellite).convert("RGB")
    sat_array = np.array(sat_img).astype(np.float32)  # shape: (H, W, 3)

    # Load the fire mask (expected to be a 500x500 array).
    fire_mask = np.load(fire_mask).astype(np.float32)
    # If the mask has values above 1, normalize it (assume 0-255 scale).
    if fire_mask.max() > 1:
        fire_mask = fire_mask / 255.0

    # --- Create a smoke mask ---
    # Diffuse the fire mask to simulate smoke spread.
    smoke_mask = gaussian_filter(fire_mask, sigma=sigma)
    # Normalize so that the mask ranges between 0 and 1.
    if smoke_mask.max() > 0:
        smoke_mask = smoke_mask / smoke_mask.max()

    # Add random noise to simulate smoke turbulence.
    noise = np.random.uniform(0, 0.2, size=smoke_mask.shape)
    smoke_mask = np.clip(smoke_mask + noise, 0, 1)

    # --- Prepare the smoke overlay ---
    # Parse the smoke color string into an RGB array.
    smoke_color = np.array([int(c) for c in smoke_color.split(",")], dtype=np.float32)
    # Reshape to (1,1,3) so it can broadcast over the image.
    smoke_color = smoke_color.reshape((1, 1, 3))

    # Expand the 2D smoke mask to 3 channels.
    smoke_mask_3ch = np.stack([smoke_mask]*3, axis=-1)

    # Calculate per-pixel alpha (opacity) based on the smoke mask and global opacity factor.
    alpha = opacity * smoke_mask_3ch  # Values in [0, opacity]

    # Blend the satellite image with the smoke color.
    # For each pixel: blended = (1 - alpha) * satellite + alpha * smoke_color
    blended_array = (1 - alpha) * sat_array + alpha * smoke_color
    blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)

    # Save the final composited image.
    output_img = Image.fromarray(blended_array)
    output_img.save(output)
    print(f"Saved output image with smoke artifacts: {output}") 

if __name__ == "__main__":
    main()
