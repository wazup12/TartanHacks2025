#!/usr/bin/env python3
"""
simulate_fire_update_graph_heatmap.py

This script demonstrates an end-to-end pipeline that:
  1. Loads and augments satellite images of the 2025 LA fires.
  2. Processes these images into binary fire masks using Otsu thresholding.
  3. Loads real hourly wind data for Los Angeles using Meteostat.
  4. Loads terrain and canopy data (GeoTIFFs) and computes the terrain slope.
  5. Constructs a dataset for an MLP where:
         Input  = [flattened current fire mask, wind vector (2),
                   flattened terrain, flattened canopy, flattened slope]
         Target = flattened fire mask at the next time step.
  6. Trains an MLP (using a weighted BCE loss) to predict fire spread.
  7. Uses the trained MLP to iteratively predict a fire series starting from a single ignition point.
  8. Aligns an existing JSON graph’s node positions to the terrain coordinate system.
  9. Updates the graph’s edge weights based on the fire density computed by counting the number of fire pixels within a specified radius around each node.
 10. Generates a heat map by “rasterizing” the edge weights of the updated graph and visualizes it.

Dependencies:
  - meteostat (pip install meteostat)
  - pandas
  - numpy
  - torch
  - matplotlib
  - rasterio
  - scikit-image
  - tqdm
  - scipy
  - networkx (pip install networkx)
  - json (standard library)
"""

from datetime import datetime, timedelta
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import rasterio
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, square
from tqdm import tqdm
from scipy.ndimage import binary_dilation, gaussian_filter
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.animation as animation
import networkx as nx

# Meteostat for wind data
from meteostat import Point, Hourly

# ----------------------------
# Global Parameters and File Paths (update these as needed)
# ----------------------------
GRID_HEIGHT = 64
GRID_WIDTH = 64

# Augmentation parameters
NUM_INTERMEDIATE = 2  # Number of intermediate images between consecutive frames
NOISE_SIGMA = 0.05  # Standard deviation of Gaussian noise

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 30  # Increased epochs for better training
LEARNING_RATE = 0.001
HIDDEN_DIM = 512  # Increased model capacity

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# File paths – update these to point to your actual data
SATELLITE_FOLDER = "FireSatelliteData"  # Folder containing satellite GeoTIFFs
TERRAIN_FILEPATH = "output_GEBCOIceTopo.tif"  # Example terrain file
CANOPY_FILEPATH = (
    "output_GEBCOIceTopo.tif"  # Using same file as placeholder (update if available)
)
GRAPH_JSON_FILE = "street_graph.json"  # Existing JSON graph file
UPDATED_GRAPH_JSON_FILE = "updated_street_graph.json"
WIND_START = datetime(2025, 1, 1)


# ----------------------------
# Utility Functions: Canopy, Slope, and Graph Alignment
# ----------------------------
def load_canopy_data(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load canopy data from a GeoTIFF, normalize to [0,1] (assuming values 0-100), and resize.
    """
    with rasterio.open(filepath) as src:
        canopy = src.read(1)
    canopy = canopy / 100.0
    if canopy.shape != target_shape:
        canopy = resize(canopy, target_shape, anti_aliasing=True)
    return canopy.astype(np.float32)


def compute_slope(terrain: np.ndarray) -> np.ndarray:
    """
    Compute the slope from the terrain using gradients and normalize to [0,1].
    """
    dy, dx = np.gradient(terrain)
    slope = np.sqrt(dx**2 + dy**2)
    slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)
    return slope_norm.astype(np.float32)


def align_graph_to_terrain(
    graph: dict, terrain_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> dict:
    """
    Align the graph's node positions to the terrain coordinate system.
    Scales and translates the node positions so they fall within [0, W-1] for x and [0, H-1] for y.
    """
    positions = []
    for node_id, node_info in graph["nodes"].items():
        positions.append(node_info["pos"])
    positions = np.array(positions)
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    H, W = terrain_shape
    scale_x = (W - 1) / (max_x - min_x) if (max_x - min_x) != 0 else 1
    scale_y = (H - 1) / (max_y - min_y) if (max_y - min_y) != 0 else 1
    scale = min(scale_x, scale_y)
    for node_id, node_info in graph["nodes"].items():
        old_pos = np.array(node_info["pos"])
        new_x = (old_pos[0] - min_x) * scale
        new_y = (old_pos[1] - min_y) * scale
        graph["nodes"][node_id]["pos"] = [new_x, new_y]
    return graph


# ----------------------------
# Functions for Loading and Augmenting Satellite Fire Masks
# ----------------------------
def load_satellite_fire_mask(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load a satellite image from a GeoTIFF and convert it into a binary fire mask using Otsu thresholding.
    """
    with rasterio.open(filepath) as src:
        image = src.read(1)
    if image.shape != target_shape:
        image = resize(image, target_shape, anti_aliasing=True)
    thresh = threshold_otsu(image)
    fire_mask = (image > thresh).astype(np.uint8)
    return fire_mask


def load_and_augment_satellite_fire_series(
    folder: str,
    num_intermediate: int = NUM_INTERMEDIATE,
    noise_sigma: float = NOISE_SIGMA,
    target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH),
) -> np.ndarray:
    """
    Load satellite images from a folder and augment them by generating intermediate images via linear interpolation and Gaussian noise.
    Assumes files are sorted in temporal order.
    """
    file_list = sorted(glob.glob(folder + "/*.tif"))
    if len(file_list) < 2:
        raise ValueError("Not enough images for augmentation.")
    original_series = []
    for f in file_list:
        fire_mask = load_satellite_fire_mask(f, target_shape)
        original_series.append(fire_mask.astype(np.float32))
    augmented_series = []
    for i in range(len(original_series) - 1):
        img1 = original_series[i]
        img2 = original_series[i + 1]
        augmented_series.append(img1)
        for j in range(1, num_intermediate + 1):
            alpha = j / (num_intermediate + 1)
            interpolated = (1 - alpha) * img1 + alpha * img2
            noisy = interpolated + np.random.normal(0, noise_sigma, interpolated.shape)
            noisy = np.clip(noisy, 0, 1)
            binary_noisy = (noisy > 0.5).astype(np.uint8)
            augmented_series.append(binary_noisy)
    augmented_series.append(original_series[-1])
    return np.array(augmented_series)


# ----------------------------
# Functions for Fetching Wind Data
# ----------------------------
def get_wind_data(start: datetime, num_steps: int) -> np.ndarray:
    """
    Fetch hourly wind data for Los Angeles starting at 'start' for 'num_steps' hours.
    Converts wind speed and direction into a vector (dy, dx).
    """
    end = start + timedelta(hours=num_steps)
    la_point = Point(34.0522, -118.2437)
    data = Hourly(la_point, start, end)
    df = data.fetch().iloc[:num_steps]
    wind_vectors = []
    for _, row in df.iterrows():
        wspd = row["wspd"]
        wdir = row["wdir"]
        effective_dir = (wdir + 180) % 360
        rad = np.deg2rad(effective_dir)
        dx = wspd * np.sin(rad)
        dy = wspd * np.cos(rad)
        wind_vectors.append([dy, dx])
    wind_data = np.array(wind_vectors, dtype=np.float32)
    print("Fetched wind data shape:", wind_data.shape)
    return wind_data


# ----------------------------
# Function for Loading Terrain Data
# ----------------------------
def load_terrain_data(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load terrain data from a GeoTIFF, normalize it to [0,1], and resize.
    """
    with rasterio.open(filepath) as src:
        terrain = src.read(1)
    terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
    if terrain.shape != target_shape:
        terrain = resize(terrain, target_shape, anti_aliasing=True)
    return terrain.astype(np.float32)


# ----------------------------
# Dataset Construction for the MLP
# ----------------------------
class FirePredictionDataset(Dataset):
    """
    Each sample is:
       Input = [flattened current fire mask, wind vector (2), flattened terrain, flattened canopy, flattened slope]
       Target = flattened fire mask at time t+1.
    """

    def __init__(
        self,
        fire_series: np.ndarray,
        terrain: np.ndarray,
        canopy: np.ndarray,
        slope: np.ndarray,
        wind_data: np.ndarray,
    ):
        self.inputs = []
        self.targets = []
        num_samples = fire_series.shape[0] - 1
        H, W = terrain.shape
        terrain_flat = terrain.flatten().astype(np.float32)
        canopy_flat = canopy.flatten().astype(np.float32)
        slope_flat = slope.flatten().astype(np.float32)
        for t in range(num_samples):
            fire_flat = fire_series[t].astype(np.float32).flatten()
            wind_vector = wind_data[t]
            input_features = np.concatenate(
                [fire_flat, wind_vector, terrain_flat, canopy_flat, slope_flat]
            )
            target_flat = fire_series[t + 1].astype(np.float32).flatten()
            self.inputs.append(input_features)
            self.targets.append(target_flat)
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


# ----------------------------
# MLP Definition and Training
# ----------------------------
class FireMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(FireMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


def train_model(
    model: nn.Module, dataloader: DataLoader, num_epochs: int, learning_rate: float
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Compute overall positive pixel ratio to weight the loss:
    all_targets = dataloader.dataset.targets
    pos_ratio = np.mean(all_targets)
    neg_ratio = 1 - pos_ratio
    pos_weight_val = neg_ratio / (pos_ratio + 1e-8)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32).to(device)
    print(f"Computed pos_weight: {pos_weight_val:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# ----------------------------
# Iterative Prediction Using the Trained MLP
# ----------------------------
def predict_fire_series(
    model: nn.Module,
    initial_fire: np.ndarray,
    terrain: np.ndarray,
    canopy: np.ndarray,
    slope: np.ndarray,
    wind_data: np.ndarray,
    num_timesteps: int,
) -> np.ndarray:
    """
    Starting from an initial fire mask, use the trained MLP iteratively to predict the fire evolution over num_timesteps.
    """
    H, W = terrain.shape
    predicted_series = []
    current_fire = initial_fire.copy()
    predicted_series.append(current_fire.copy())
    terrain_flat = terrain.flatten().astype(np.float32)
    canopy_flat = canopy.flatten().astype(np.float32)
    slope_flat = slope.flatten().astype(np.float32)
    device = next(model.parameters()).device
    for t in range(1, num_timesteps):
        fire_flat = current_fire.astype(np.float32).flatten()
        wind_vector = wind_data[t]
        input_features = np.concatenate(
            [fire_flat, wind_vector, terrain_flat, canopy_flat, slope_flat]
        )
        input_tensor = (
            torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            output_logits = model(input_tensor)
            output_probs = torch.sigmoid(output_logits).squeeze().cpu().numpy()
        new_fire = (output_probs > 0.5).astype(np.uint8).reshape(H, W)
        predicted_series.append(new_fire.copy())
        current_fire = new_fire
    return np.array(predicted_series)


# ----------------------------
# 3D Animation of the Predicted Fire Series
# ----------------------------
def animate_fire_on_3d(
    terrain: np.ndarray,
    fire_series: np.ndarray,
    interval: int = 200,
    z_offset: float = 0.0,
    marker_size: int = 5,
    max_points: int = 300,
    elevation_scale: float = 2.0,
):
    H, W = terrain.shape
    scaled_terrain = terrain * elevation_scale
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, scaled_terrain, cmap="terrain", alpha=0.7, linewidth=0, antialiased=True
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Scaled Elevation")

    def get_scatter_data(fire_mask):
        indices = np.where(fire_mask)
        xs = indices[1]
        ys = indices[0]
        zs = scaled_terrain[indices] + z_offset
        if len(xs) > max_points:
            sample_indices = np.random.choice(len(xs), size=max_points, replace=False)
            xs = xs[sample_indices]
            ys = ys[sample_indices]
            zs = zs[sample_indices]
        return xs, ys, zs

    fire0 = fire_series[0]
    xs, ys, zs = get_scatter_data(fire0)
    sc = ax.scatter(xs, ys, zs, c="r", marker="o", s=marker_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Scaled Elevation")
    ax.set_title("Predicted 3D Fire Spread over Scaled Terrain")

    def update(frame):
        fire_mask = fire_series[frame]
        xs, ys, zs = get_scatter_data(fire_mask)
        sc._offsets3d = (xs, ys, zs)
        ax.set_title(f"Predicted 3D Fire Spread - Time Step {frame}")
        return (sc,)

    ani = animation.FuncAnimation(
        fig, update, frames=fire_series.shape[0], interval=interval, blit=False
    )
    ax.set_box_aspect((W, H, np.ptp(scaled_terrain)))
    plt.show()


# ----------------------------
# Graph Edge Weight Update Based on Fire Density
# ----------------------------
def count_fire_density_at_node(
    fire_mask: np.ndarray, node_pos: list, radius: int = 1
) -> float:
    """
    Count the number of fire pixels in a circular area around a node.
    """
    H, W = fire_mask.shape
    x = int(round(node_pos[0]))
    y = int(round(node_pos[1]))
    x_min = max(0, x - radius)
    x_max = min(W, x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(H, y + radius + 1)
    sub_mask = fire_mask[y_min:y_max, x_min:x_max]
    yy, xx = np.meshgrid(
        np.arange(y_min, y_max), np.arange(x_min, x_max), indexing="ij"
    )
    distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    circular_mask = distances <= radius
    density = np.sum(sub_mask[circular_mask])
    return density


def update_graph_edges_from_fire_by_density(
    graph: dict, predicted_fire_mask: np.ndarray, radius: int = 5
) -> dict:
    """
    Update each edge's weight by counting fire pixels within a given radius around the source and target nodes.
    The new weight is set to -log(average_density + eps), optionally scaled by an "importance" factor.
    """
    eps = 1e-8
    for edge in graph.get("edges", []):
        # Assume each edge is a list: [source, target, { ... }]
        source_id = str(edge[0])
        target_id = str(edge[1])
        source_pos = np.array(graph["nodes"][source_id]["pos"])
        target_pos = np.array(graph["nodes"][target_id]["pos"])
        density_source = count_fire_density_at_node(
            predicted_fire_mask, source_pos, radius
        )
        density_target = count_fire_density_at_node(
            predicted_fire_mask, target_pos, radius
        )
        avg_density = (density_source + density_target) / 2.0
        new_weight = -np.log(avg_density + eps)
        if "importance" in edge[2]:
            new_weight *= edge[2]["importance"]
        edge[2]["weight"] = new_weight
    return graph


# ----------------------------
# Functions to Create and Visualize a Heat Map of Edge Weights
# ----------------------------
def graph_edges_heatmap(graph: dict, shape: tuple, thickness: int = 1) -> np.ndarray:
    """
    Rasterize the graph's edges into a heat map based on their weights.
    For each edge, draw a line between the aligned node positions and add the edge's weight to those pixels.
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    for edge in graph.get("edges", []):
        source_id = str(edge[0])
        target_id = str(edge[1])
        source_pos = graph["nodes"][source_id]["pos"]  # [x, y]
        target_pos = graph["nodes"][target_id]["pos"]  # [x, y]
        # Convert positions to integer pixel indices.
        x0, y0 = int(round(source_pos[0])), int(round(source_pos[1]))
        x1, y1 = int(round(target_pos[0])), int(round(target_pos[1]))
        rr, cc = np.array([]), np.array([])
        try:
            from skimage.draw import line

            rr, cc = line(y0, x0, y1, x1)
        except ImportError:
            # If skimage.draw.line is unavailable, use np.linspace.
            num = max(abs(x1 - x0), abs(y1 - y0)) + 1
            rr = np.linspace(y0, y1, num, dtype=int)
            cc = np.linspace(x0, x1, num, dtype=int)
        weight = edge[2]["weight"]
        for r, c in zip(rr, cc):
            r_min = max(0, r - thickness // 2)
            r_max = min(shape[0], r + thickness // 2 + 1)
            c_min = max(0, c - thickness // 2)
            c_max = min(shape[1], c + thickness // 2 + 1)
            heatmap[r_min:r_max, c_min:c_max] += weight
    return heatmap


def visualize_heatmap(heatmap: np.ndarray, sigma: float = 1.0):
    """
    Optionally smooth the heatmap with a Gaussian filter and display it.
    """
    smoothed = gaussian_filter(heatmap, sigma=sigma)
    plt.figure(figsize=(8, 6))
    plt.imshow(smoothed, cmap="hot", interpolation="nearest")
    plt.title("Graph Edge Weights Heatmap")
    plt.colorbar(label="Accumulated Edge Weight")
    plt.show()


# ----------------------------
# Main Function: Integration and Graph Update
# ----------------------------
def main():
    # --- User Settings ---
    satellite_folder = SATELLITE_FOLDER
    terrain_filepath = TERRAIN_FILEPATH
    canopy_filepath = CANOPY_FILEPATH
    graph_json_file = GRAPH_JSON_FILE
    updated_graph_json_file = UPDATED_GRAPH_JSON_FILE
    wind_start = WIND_START

    # --- Load Terrain and Canopy ---
    terrain = load_terrain_data(
        terrain_filepath, target_shape=(GRID_HEIGHT, GRID_WIDTH)
    )
    print("Terrain loaded. Shape:", terrain.shape)
    canopy = load_canopy_data(canopy_filepath, target_shape=(GRID_HEIGHT, GRID_WIDTH))
    print("Canopy data loaded. Shape:", canopy.shape)
    slope = compute_slope(terrain)
    print("Slope computed. Shape:", slope.shape)

    # --- Load and Augment Satellite Fire Series ---
    fire_series = load_and_augment_satellite_fire_series(
        satellite_folder,
        num_intermediate=NUM_INTERMEDIATE,
        noise_sigma=NOISE_SIGMA,
        target_shape=(GRID_HEIGHT, GRID_WIDTH),
    )
    print("Augmented satellite fire series loaded. Shape:", fire_series.shape)

    # --- Fetch Wind Data ---
    wind_data = get_wind_data(wind_start, fire_series.shape[0])
    print("Wind data fetched. Shape:", wind_data.shape)

    # --- Build Dataset and DataLoader ---
    dataset = FirePredictionDataset(fire_series, terrain, canopy, slope, wind_data[:-1])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    H, W = terrain.shape
    input_dim = H * W + 2 + H * W + H * W + H * W
    output_dim = H * W
    model = FireMLP(input_dim, HIDDEN_DIM, output_dim)
    print(model)

    # --- Train the MLP ---
    train_model(model, dataloader, NUM_EPOCHS, LEARNING_RATE)

    # --- Evaluate on One Sample (2D Visualization) ---
    model.eval()
    sample_input, sample_target = dataset[0]
    with torch.no_grad():
        sample_input = sample_input.unsqueeze(0)
        output_logits = model(sample_input)
        output_probs = torch.sigmoid(output_logits).squeeze().cpu().numpy()
    prediction_2d = (output_probs > 0.5).astype(np.uint8).reshape(H, W)
    ground_truth = sample_target.numpy().reshape(H, W)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(prediction_2d, cmap="Reds", interpolation="nearest")
    plt.title("Predicted Fire Mask (Next Step)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth, cmap="Reds", interpolation="nearest")
    plt.title("Ground Truth Fire Mask (Next Step)")
    plt.colorbar()
    plt.show()

    # --- Use the Trained MLP for Iterative Prediction ---
    initial_fire = np.zeros((H, W), dtype=np.uint8)
    initial_fire[H // 2, W // 2] = 1
    predicted_series = predict_fire_series(
        model, initial_fire, terrain, canopy, slope, wind_data, fire_series.shape[0]
    )
    print("Predicted fire series shape:", predicted_series.shape)

    # --- 3D Visualization of the Predicted Fire Series ---
    animate_fire_on_3d(
        terrain,
        predicted_series,
        interval=200,
        z_offset=0.0,
        marker_size=5,
        elevation_scale=2.0,
        max_points=300,
    )

    # --- Load Existing Graph JSON, Align to Terrain, and Update Edge Weights ---
    with open(graph_json_file, "r") as f:
        graph = json.load(f)
    graph_aligned = align_graph_to_terrain(
        graph, terrain_shape=(GRID_HEIGHT, GRID_WIDTH)
    )
    final_predicted_fire = predicted_series[-1]
    # Use the density-based update: count fire pixels within a radius at each node.
    updated_graph = update_graph_edges_from_fire_by_density(
        graph_aligned, final_predicted_fire, radius=1
    )
    with open(updated_graph_json_file, "w") as f:
        json.dump(updated_graph, f, indent=2)
    print(f"Updated graph saved to {updated_graph_json_file}")

    # --- Generate a Heat Map from the Graph Edges ---
    heatmap_shape = (GRID_HEIGHT, GRID_WIDTH)
    heatmap = graph_edges_heatmap(updated_graph, shape=heatmap_shape, thickness=1)
    visualize_heatmap(heatmap, sigma=1.0)


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
