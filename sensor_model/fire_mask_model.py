#!/usr/bin/env python3
"""
A simple synthetic fire spread simulator using a cellular automata model.
Each cell in a 500x500 grid starts as 0 (no fire), except for one source cell.
At each time step, any cell that is not burning may catch fire if one or more of its
8 neighboring cells is burning. The probability that a cell ignites is given by:

    P_ignite = 1 - ∏ (over burning neighbors i) (1 - p_dir_i)

where for each neighbor in direction i, the directional probability is

    p_dir(i,j) = p_base * f_wind * f_slope(i,j)

with
    f_wind = 1 + wind_mag * cos(deg2rad(theta_neighbor - wind_dir))
    f_slope(i,j) = 1 - gradient[i,j] * cos(deg2rad(theta_neighbor - slope_dir))

Here, the gradient is a 500x500 numpy array (loaded from a .npy file) containing
a steepness coefficient for each point in the image. The angles are measured in degrees
relative to north (0°).
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt

def shift_with_zero_padding(arr, shift_row, shift_col):
    """
    Shifts a 2D array by (shift_row, shift_col) without wrapping (new areas filled with 0).
    """
    result = np.zeros_like(arr)
    # Determine source and destination slices for the row dimension.
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

    # Similarly for the column dimension.
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

    result[row_start_dest:row_end_dest, col_start_dest:col_end_dest] = \
        arr[row_start_src:row_end_src, col_start_src:col_end_src]
    return result

def simulate_fire(wind_dir, wind_mag, slope_dir, gradient, grad_scale, steps=100, grid_size=500, p_base=0.35):
    """
    Runs the fire spread simulation.

    Parameters:
      - wind_dir: Wind direction in degrees relative to north.
      - wind_mag: Wind magnitude (a scaling factor; e.g. 0.2).
      - slope_dir: The direction of the steepest uphill (in degrees relative to north).
      - gradient: A 500x500 numpy array containing the steepness coefficient for each point.
      - steps: Number of time steps to simulate.
      - grid_size: The size of the grid (grid_size x grid_size).
      - p_base: Baseline ignition probability from a burning neighbor.
    """
    # Initialize the grid: 0 for not burning, 1 for burning.
    grid = np.zeros((grid_size, grid_size), dtype=int)
    # Set the initial burning cell (here, the center).
    center = grid_size // 2
    grid[center, center] = 1

    # Define the eight neighbor directions.
    # For each neighbor, we record its (row, col) offset and the corresponding direction in degrees.
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

    # For each neighbor direction, precompute the directional probability adjustments.
    # Note: The wind factor is a scalar per neighbor direction, whereas the slope factor is computed
    # locally using the gradient matrix.
    for d in neighbor_data:
        theta = d["angle"]
        # Wind adjustment (same for all cells in the grid).
        wind_factor = 1 + wind_mag * np.cos(np.deg2rad(theta - wind_dir))
        # Local slope adjustment: note that 'gradient' is a 2D array.
        slope_factor = 1 - grad_scale * gradient * np.cos(np.deg2rad(theta - slope_dir))
        # Compute the directional ignition probability for this neighbor, element-wise.
        p_dir = p_base * wind_factor * slope_factor
        # Ensure probabilities are between 0 and 1.
        p_dir = np.clip(p_dir, 0, 1)
        d["p"] = p_dir

    # Save each frame (the state at each time step).
    states = [grid.copy()]

    for t in range(steps):
        # For each cell, compute the probability that it does NOT ignite given contributions
        # from each burning neighbor.
        prob_not_ignite = np.ones_like(grid, dtype=float)
        # Loop over each neighbor direction.
        for d in neighbor_data:
            dr, dc = d["offset"]
            # p_dir = d["p"]  # This is an array of shape (grid_size, grid_size).
            # Shift the grid so that for each cell, the corresponding cell in 'shifted'
            # is the neighbor in this direction. (Cells that fall off are set to 0.)
            random_multiplier = np.random.uniform(0.5, 1.5, size=grid.shape)
            effective_p_dir = np.clip(d["p"] * random_multiplier, 0, 1)
            shifted = shift_with_zero_padding(grid, dr, dc)
            # For cells where the neighbor is burning, the chance of *not* igniting from that neighbor is (1 - p_dir).
            factor = np.where(shifted == 1, 1 - effective_p_dir, 1.0)
            prob_not_ignite *= factor

        # Overall ignition probability is the complement.
        prob_ignite = 1 - prob_not_ignite

        # noise = np.random.uniform(-0.001, 0.001, size=grid.shape)
        # prob_ignite = np.clip(prob_ignite + noise, 0, 1)

        # Ignite new cells (only if not already burning).
        random_vals = np.random.random(size=grid.shape)
        new_fires = ((grid == 0) & (random_vals < prob_ignite))
        grid[new_fires] = 1

        # Save the current grid state.
        states.append(grid.copy())
        # print(f"Time step {t+1}:")
        # print(grid)

        # Every 10 time steps, save a jpg image of the current grid state.
        if (t+1) % 100 == 0:
            filename = f"fire_step_{t+1:03d}.jpg"
            # Use a heatmap for visualization. The "hot" colormap highlights the fire intensity.
            plt.imsave(filename, grid, cmap='hot')
            print(f"Saved image: {filename}")

        if (t+1) % 100 == 0:
            filename = f"fire_mask_{t+1:03d}.npy"
            np.save(filename, grid)
            print(f"Saved fire mask as numpy array: {filename}")

    # Print summary parameters after the simulation.
    print("Simulation complete.")
    print(f"Wind direction: {wind_dir}°; Wind magnitude: {wind_mag}")
    print(f"Slope (uphill) direction: {slope_dir}°; using local gradient matrix for slope effect.")
    return states

if __name__ == "__main__":
    # Set up argparse so that the user can input parameters from the command line.
    parser = argparse.ArgumentParser(
        description="Simulate fire spread on a 500x500 grid using a probabilistic cellular automata model."
    )
    # parser.add_argument("--wind_dir", type=float, default=90.0,
    #                     help="Wind direction in degrees relative to north (default: 90)")
    # parser.add_argument("--wind_mag", type=float, default=0.2,
    #                     help="Wind magnitude (default: 0.2)")
    # parser.add_argument("--slope_dir", type=float, default=0.0,
    #                     help="Slope direction (angle of uphill) in degrees relative to north (default: 0)")
    # parser.add_argument("--gradient", type=str, required=True,
    #                     help="Path to .npy file containing the 500x500 gradient matrix for slope steepness")
    # parser.add_argument("--steps", type=int, default=100,
    #                     help="Number of time steps to simulate (default: 100)")
    
    
    # args = parser.parse_args()


    gradient = 'gradient_0.npy'
    wind_dir = 90.0
    wind_mag = 0.4
    slope_dir = 0.0
    steps = 500
    grad_scale = 0.0025


    # Load the gradient matrix.
    gradient = np.load(gradient)
    if gradient.shape != (500, 500):
    # Verify that the gradient matrix is the expected shape.
        raise ValueError("Gradient matrix must be of shape (500, 500)")

    # Run the simulation
    simulate_fire(wind_dir, wind_mag, slope_dir, gradient, grad_scale, steps=steps)
