#!/usr/bin/env python3
"""
display_3d_tif.py

This script loads a GeoTIFF file using rasterio and displays a 3D surface plot
of the terrain data. The plot uses the geospatial extent from the file and
assumes that the first band of the GeoTIFF represents elevation.

Dependencies:
  - numpy
  - matplotlib
  - rasterio
  - mpl_toolkits.mplot3d (comes with matplotlib)
"""

import rasterio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


def display_3d_tif(filepath: str) -> None:
    """
    Load a GeoTIFF file and display a 3D surface plot of its terrain data.

    Args:
        filepath (str): Path to the GeoTIFF file.
    """
    try:
        with rasterio.open(filepath) as src:
            # Read the first band (assuming it contains elevation data)
            image = src.read(1)
            # Get the geospatial bounds of the image
            bounds = src.bounds
            print("Image shape:", image.shape)
            print("Image bounds:", bounds)
    except Exception as e:
        print("Error loading GeoTIFF:", e)
        return

    # Derive coordinate arrays for the image.
    # image.shape returns (rows, cols)
    rows, cols = image.shape
    # Create x-coordinates from left to right and y-coordinates from bottom to top
    x = np.linspace(bounds.left, bounds.right, num=cols)
    y = np.linspace(bounds.bottom, bounds.top, num=rows)
    X, Y = np.meshgrid(x, y)

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface. Adjust rstride and cstride for performance if needed.
    surf = ax.plot_surface(X, Y, image, cmap="terrain", linewidth=0, antialiased=False)

    # Add a colorbar to map elevation values to colors.
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation")

    # Set axis labels and title
    ax.set_xlabel("X Coordinate (Longitude)")
    ax.set_ylabel("Y Coordinate (Latitude)")
    ax.set_zlabel("Elevation")
    ax.set_title("3D Terrain Visualization from GeoTIFF")

    plt.show()


def main() -> None:
    tif_filepath = "output_GEBCOIceTopo.tif"
    display_3d_tif(tif_filepath)


if __name__ == "__main__":
    main()
