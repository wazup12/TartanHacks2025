#!/usr/bin/env python3
"""
animate_fire_growth_full_map.py

This script loads a terrain dataset from a GeoTIFF file,
simulates wildfire spread as a time series of binary masks,
and then animates how the fire grows over the full terrain map.

The animation overlays the fire (in semi‐transparent red) on top of the terrain (displayed in grayscale).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rasterio
from skimage.transform import resize
from scipy.ndimage import binary_dilation

# Parameters
GRID_HEIGHT = 64
GRID_WIDTH = 64
NUM_TIMESTEPS = 100  # Number of time steps in the simulation
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_terrain_data(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load terrain data from a GeoTIFF file and resize it to the target shape.

    Args:
        filepath (str): Path to the GeoTIFF file.
        target_shape (tuple): Desired output shape (height, width).

    Returns:
        terrain (np.ndarray): Normalized terrain data in the range [0, 1].
    """
    with rasterio.open(filepath) as src:
        terrain = src.read(1)  # Read the first band
    # Normalize terrain data to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    # Resize if necessary
    if terrain.shape != target_shape:
        terrain = resize(terrain, target_shape, anti_aliasing=True)
    return terrain.astype(np.float32)


def simulate_fire_spread(
    num_timesteps: int, grid_shape: tuple, wind: tuple = (1, 0)
) -> np.ndarray:
    """
    Simulate wildfire spread as a time series of binary masks.
    In your real application, replace this simulation with your real fire mask data.

    Args:
        num_timesteps (int): Number of time steps.
        grid_shape (tuple): (height, width) of the grid.
        wind (tuple): Wind offset (dy, dx) applied at each time step.

    Returns:
        fire_series (np.ndarray): Array of shape (num_timesteps, height, width) with binary fire masks.
    """
    height, width = grid_shape
    fire_series = np.zeros((num_timesteps, height, width), dtype=np.uint8)

    # Initialize fire: a small square in the center
    init_fire = np.zeros((height, width), dtype=np.uint8)
    center_y, center_x = height // 2, width // 2
    init_fire[center_y - 2 : center_y + 3, center_x - 2 : center_x + 3] = 1
    fire_series[0] = init_fire

    # Define a structuring element for dilation (4-connected neighbors)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    # Simulate fire spread by dilating and shifting the fire mask
    for t in range(1, num_timesteps):
        dilated = binary_dilation(fire_series[t - 1], structure=struct).astype(np.uint8)
        # Apply wind: roll the mask by the wind offset (wrap-around behavior)
        shifted = np.roll(dilated, shift=wind, axis=(0, 1))
        # Optionally add a bit of randomness (simulate occasional extinguishing)
        noise = np.random.rand(height, width) > 0.98
        new_fire = np.where(noise, 0, shifted)
        fire_series[t] = new_fire

    return fire_series


def animate_fire_on_map(
    terrain: np.ndarray, fire_series: np.ndarray, interval: int = 200
) -> None:
    """
    Animate the fire growth over the full terrain map.

    Args:
        terrain (np.ndarray): 2D array representing the terrain (grayscale).
        fire_series (np.ndarray): 3D array (time, height, width) of binary fire masks.
        interval (int): Delay between frames in milliseconds.
    """
    # Set up the figure with a larger size so the full map is visible
    fig, ax = plt.subplots(figsize=(8, 8))
    height, width = terrain.shape
    # Set extent to show the full map: [left, right, bottom, top]
    extent = [0, width, height, 0]
    # Display the terrain as the background with the defined extent
    ax.imshow(terrain, cmap="gray", interpolation="nearest", extent=extent)
    # Overlay the fire mask with a red colormap and transparency, using the same extent
    fire_overlay = ax.imshow(
        fire_series[0], cmap="Reds", alpha=0.5, interpolation="nearest", extent=extent
    )
    ax.set_title("Wildfire Growth on Terrain Map")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    def update(frame):
        fire_overlay.set_data(fire_series[frame])
        ax.set_xlabel(f"Time Step: {frame}")
        return [fire_overlay]

    ani = animation.FuncAnimation(
        fig, update, frames=fire_series.shape[0], interval=interval, blit=True
    )
    plt.show()


def main() -> None:
    # Specify the path to your terrain GeoTIFF file
    terrain_filepath = "output_GEBCOIceTopo.tif"  # <-- update this path to your file

    try:
        terrain = load_terrain_data(
            terrain_filepath, target_shape=(GRID_HEIGHT, GRID_WIDTH)
        )
        print("Terrain data loaded successfully.")
    except Exception as e:
        print(f"Error loading terrain data: {e}")
        # Fallback: use a flat terrain if loading fails
        terrain = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)

    # Simulate fire spread (replace with your real fire data if available)
    fire_series = simulate_fire_spread(
        NUM_TIMESTEPS, (GRID_HEIGHT, GRID_WIDTH), wind=(1, 0)
    )
    print(f"Fire series shape: {fire_series.shape}")

    # Animate the fire growing over the full terrain map
    animate_fire_on_map(terrain, fire_series, interval=200)


if __name__ == "__main__":
    main()
