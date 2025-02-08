#!/usr/bin/env python3
"""
Full synthetic training pipeline for simulating fire spread over terrain and training
a PointNet++ model to predict the next set of fire particle positions given local terrain gradients
and weather conditions.

Each simulation runs a cellular automata–based fire spread on a 500x500 grid using a synthetic
terrain gradient (e.g. smoothed random noise). For each time step, the positions (and gradient values)
of burning cells are extracted, augmented with weather features, and paired with the next step’s
burning points. A fixed number of points per sample is enforced by random sampling or zero–padding.
Finally, a simple PointNet–style model is trained to predict the next time step’s particle positions
using Chamfer Distance as the loss.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os

# For smoothing our synthetic terrain
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


#############################
# SIMULATION FUNCTIONS
#############################

def shift_with_zero_padding(arr, shift_row, shift_col):
    """
    Shifts a 2D array by (shift_row, shift_col) without wrapping.
    New areas are filled with 0.
    """
    result = np.zeros_like(arr)
    # Rows:
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
    # Columns:
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

def simulate_fire_simulation(wind_dir, wind_mag, slope_dir, gradient, grad_scale,
                             steps=100, grid_size=500, p_base=0.35):
    """
    Runs the fire spread simulation on a grid.
    
    Parameters:
      wind_dir   : Wind direction in degrees (relative to north)
      wind_mag   : Wind magnitude (a scaling factor)
      slope_dir  : Direction of uphill (degrees relative to north)
      gradient   : A (grid_size x grid_size) numpy array (terrain steepness)
      grad_scale : Scaling factor for the gradient effect
      steps      : Number of time steps
      grid_size  : Size of the grid (grid_size x grid_size)
      p_base     : Baseline ignition probability
    Returns:
      states     : A list of grid states (each a (grid_size x grid_size) array)
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    grid[center, center] = 1

    # Define the 8 neighbor directions
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
    # Precompute directional probability adjustments for each neighbor.
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

def generate_synthetic_gradient(grid_size=500):
    """
    Generates a synthetic terrain gradient using random noise smoothed with a Gaussian filter.
    """
    noise = np.random.rand(grid_size, grid_size)
    smooth_noise = gaussian_filter(noise, sigma=10)
    # Normalize to [0,1]
    smooth_noise = (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())
    return smooth_noise

def grid_to_point_cloud(grid, gradient):
    """
    Converts a grid state into a point cloud.
    Each burning cell becomes a point with features: (x, y, gradient_value).
    x and y are normalized coordinates in [0,1].
    """
    burning_indices = np.argwhere(grid == 1)
    if burning_indices.shape[0] == 0:
        return np.empty((0, 3))
    # Normalize indices
    points = burning_indices / (grid.shape[0] - 1)
    grad_vals = gradient[burning_indices[:, 0], burning_indices[:, 1]].reshape(-1, 1)
    point_cloud = np.concatenate([points, grad_vals], axis=1)
    return point_cloud


#############################
# DATASET DEFINITION
#############################

class SyntheticFireDataset(Dataset):
    """
    Generates synthetic training samples. For each simulation, we run a fire spread simulation
    over a synthetic terrain and random weather. For each pair of consecutive time steps, we
    extract the burning cells as a point cloud. Each point’s features consist of:
      - (x, y) normalized location,
      - local gradient value,
      - weather conditions (wind_dir, wind_mag, slope_dir) normalized.
    We then sample (or pad) the point cloud to a fixed size.
    """
    def __init__(self, num_simulations=100, steps_per_simulation=50, num_points=1024, grid_size=500):
        self.num_simulations = num_simulations
        self.steps_per_simulation = steps_per_simulation
        self.num_points = num_points
        self.grid_size = grid_size
        self.data = []  # List of tuples: (input_features, output_points)
        self.generate_dataset()
        
    def generate_dataset(self):
        for sim in range(self.num_simulations):
            # Random weather parameters
            wind_dir = random.uniform(0, 360)
            wind_mag = random.uniform(0.1, 0.5)
            slope_dir = random.uniform(0, 360)
            grad_scale = 0.0025

            # Generate a synthetic terrain gradient
            gradient = generate_synthetic_gradient(self.grid_size)
            # Run the fire simulation
            states = simulate_fire_simulation(wind_dir, wind_mag, slope_dir, gradient,
                                              grad_scale, steps=self.steps_per_simulation,
                                              grid_size=self.grid_size)
            # Normalize weather parameters (angles divided by 360)
            weather_features = np.array([wind_dir/360.0, wind_mag, slope_dir/360.0])
            # For each consecutive pair of states, form a training sample.
            for t in range(len(states)-1):
                input_grid = states[t]
                output_grid = states[t+1]
                input_pc = grid_to_point_cloud(input_grid, gradient)   # shape: (N_in, 3)
                output_pc = grid_to_point_cloud(output_grid, gradient) # shape: (N_out, 3)
                # Skip if one of the frames has no burning cells.
                if input_pc.shape[0] == 0 or output_pc.shape[0] == 0:
                    continue
                # Append the weather features to each input point (so each input point has 6 features)
                input_features = np.concatenate([input_pc, np.tile(weather_features, (input_pc.shape[0], 1))], axis=1)
                # Our prediction target will be the next step’s (x,y) positions.
                output_points = output_pc[:, :2]
                # Ensure fixed number of points per sample.
                input_features = self.sample_or_pad(input_features, self.num_points)
                output_points = self.sample_or_pad(output_points, self.num_points)
                self.data.append((input_features.astype(np.float32), output_points.astype(np.float32)))
                
    def sample_or_pad(self, points, num_points):
        """
        If the point cloud has more than num_points points, randomly sample num_points.
        Otherwise, pad with zeros.
        """
        N, D = points.shape
        if N >= num_points:
            indices = np.random.choice(N, num_points, replace=False)
            return points[indices]
        else:
            pad = np.zeros((num_points - N, D), dtype=points.dtype)
            return np.concatenate([points, pad], axis=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_features, output_points = self.data[idx]
        return input_features, output_points


#############################
# LOSS: CHAMFER DISTANCE
#############################

def chamfer_distance(pred, target):
    """
    Computes a simple Chamfer distance between two point clouds.
    pred: tensor of shape (B, N, 2)
    target: tensor of shape (B, M, 2)
    """
    B, N, _ = pred.size()
    _, M, _ = target.size()
    # Expand dims to compute pairwise distances.
    pred_expanded = pred.unsqueeze(2)    # (B, N, 1, 2)
    target_expanded = target.unsqueeze(1)  # (B, 1, M, 2)
    dist = torch.norm(pred_expanded - target_expanded, dim=3)  # (B, N, M)
    # For each point in pred, find min distance to target.
    min_pred_to_target, _ = torch.min(dist, dim=2)  # (B, N)
    # For each point in target, find min distance to pred.
    min_target_to_pred, _ = torch.min(dist, dim=1)  # (B, M)
    loss = torch.mean(min_pred_to_target) + torch.mean(min_target_to_pred)
    return loss


#############################
# SIMPLE PointNet++ MODEL
#############################

class PointNet(nn.Module):
    """
    A simplified PointNet/PointNet++–like network.
    Input: (B, N, 6) where each point has (x, y, gradient, wind_dir, wind_mag, slope_dir)
    Output: (B, N, 2) the predicted next (x, y) positions.
    """
    def __init__(self, input_dim=6, output_dim=2):
        super(PointNet, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        # Feature propagation / fusion layers
        self.fc1 = nn.Sequential(
            nn.Conv1d(64+128+256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc3 = nn.Conv1d(128, output_dim, 1)
    
    def forward(self, x):
        # x: (B, N, input_dim); transpose for conv layers: (B, input_dim, N)
        x = x.transpose(1, 2)
        x1 = self.mlp1(x)    # (B, 64, N)
        x2 = self.mlp2(x1)   # (B, 128, N)
        x3 = self.mlp3(x2)   # (B, 256, N)
        # Concatenate features from multiple layers.
        concat_features = torch.cat([x1, x2, x3], dim=1)  # (B, 64+128+256, N)
        x = self.fc1(concat_features)
        x = self.fc2(x)
        out = self.fc3(x)   # (B, output_dim, N)
        out = out.transpose(1, 2)  # (B, N, output_dim)
        return out


#############################
# TRAINING FUNCTION
#############################

def train_model(dataset, num_epochs=50, batch_size=8, learning_rate=0.001, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = PointNet(input_dim=6, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (input_features, target_points) in enumerate(dataloader):
            input_features = input_features.to(device)  # (B, N, 6)
            target_points = target_points.to(device)    # (B, N, 2)
            optimizer.zero_grad()
            pred_points = model(input_features)          # (B, N, 2)
            loss = chamfer_distance(pred_points, target_points)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    return model


#############################
# MAIN
#############################

def main():
    parser = argparse.ArgumentParser(description="Synthetic Fire Spread Training Pipeline with PointNet++")
    parser.add_argument('--num_simulations', type=int, default=50,
                        help="Number of simulation runs for dataset generation")
    parser.add_argument('--steps_per_simulation', type=int, default=50,
                        help="Number of steps per simulation")
    parser.add_argument('--num_points', type=int, default=1024,
                        help="Number of points per point cloud sample")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device for training (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()
    
    # Create the synthetic dataset.
    print("Generating synthetic dataset...")
    dataset = SyntheticFireDataset(num_simulations=args.num_simulations,
                                   steps_per_simulation=args.steps_per_simulation,
                                   num_points=args.num_points)
    print(f"Dataset generated with {len(dataset)} samples.")
    
    # Train the model.
    print("Training model...")
    model = train_model(dataset, num_epochs=args.epochs, batch_size=args.batch_size,
                        learning_rate=args.lr, device=args.device)
    
    # Save the trained model.
    model_path = "PointNet_fire_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved as {model_path}")

if __name__ == "__main__":
    main()
