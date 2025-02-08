#!/usr/bin/env python3
"""
simulate_fire_with_satellite_augmented_mlp_3d.py

This script demonstrates an end-to-end pipeline that:
  1. Loads satellite images of the 2025 LA fires from a folder.
  2. Augments the dataset by generating intermediate images (via linear interpolation and Gaussian noise).
  3. Processes the images to obtain binary fire masks using Otsu thresholding.
  4. Fetches real hourly wind data for Los Angeles using Meteostat.
  5. Loads terrain data from a GeoTIFF.
  6. Constructs a dataset for an MLP where:
         Input  = [flattened current fire mask, wind vector (2 features), flattened terrain]
         Target = flattened fire mask at the next time step.
  7. Trains an MLP to predict fire spread.
  8. Uses the trained MLP to iteratively predict a fire series.
  9. Animates the predicted fire evolution in 3D over the terrain.

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
"""

from datetime import datetime, timedelta
import glob
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
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.animation as animation

# Meteostat for wind data
from meteostat import Point, Hourly

# ----------------------------
# Parameters and Settings
# ----------------------------
GRID_HEIGHT = 128
GRID_WIDTH = 320

# Augmentation parameters
NUM_INTERMEDIATE = (
    2  # Number of intermediate images to generate between consecutive frames
)
NOISE_SIGMA = 0.05  # Standard deviation of Gaussian noise

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ----------------------------
# Step 1. Load and Augment Satellite Images
# ----------------------------
def load_satellite_fire_mask(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load a satellite image from a GeoTIFF file and convert it into a binary fire mask.
    Uses Otsu thresholding.
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
    Load satellite images from a folder and augment them by generating intermediate images
    via linear interpolation and adding Gaussian noise.
    Assumes the folder contains GeoTIFF files that can be sorted in temporal order.
    """
    file_list = sorted(glob.glob(folder + "/*.tif"))
    if len(file_list) < 2:
        raise ValueError("Not enough images for augmentation.")
    original_series = []
    for f in file_list:
        fire_mask = load_satellite_fire_mask(f, target_shape)
        original_series.append(fire_mask.astype(np.float32))
    augmented_series = []
    # For each pair of consecutive images, generate intermediate frames.
    for i in range(len(original_series) - 1):
        img1 = original_series[i]
        img2 = original_series[i + 1]
        augmented_series.append(img1)  # include the first image of the pair
        for j in range(1, num_intermediate + 1):
            alpha = j / (num_intermediate + 1)
            # Linear interpolation between img1 and img2
            interpolated = (1 - alpha) * img1 + alpha * img2
            # Add Gaussian noise
            noisy = interpolated + np.random.normal(0, noise_sigma, interpolated.shape)
            noisy = np.clip(noisy, 0, 1)
            # Binarize the result using a threshold of 0.5
            binary_noisy = (noisy > 0.5).astype(np.uint8)
            augmented_series.append(binary_noisy)
    # Append the last original image
    augmented_series.append(original_series[-1])
    return np.array(augmented_series)


# ----------------------------
# Step 2. Get Real Wind Data using Meteostat
# ----------------------------
def get_wind_data(start: datetime, num_steps: int) -> np.ndarray:
    """
    Fetch hourly wind data for Los Angeles starting at 'start' for 'num_steps' hours.
    Converts wind speed and direction into a 2D vector (dy, dx).
    """
    end = start + timedelta(hours=num_steps)
    la_point = Point(34.0522, -118.2437)
    data = Hourly(la_point, start, end)
    df = data.fetch().iloc[:num_steps]
    wind_vectors = []
    for _, row in df.iterrows():
        wspd = row["wspd"]  # wind speed in m/s
        wdir = row["wdir"]  # wind direction (from which wind comes, in degrees)
        # Convert to blowing direction by adding 180° (modulo 360)
        effective_dir = (wdir + 180) % 360
        rad = np.deg2rad(effective_dir)
        dx = wspd * np.sin(rad)
        dy = wspd * np.cos(rad)
        wind_vectors.append([dy, dx])
    wind_data = np.array(wind_vectors, dtype=np.float32)
    print("Fetched wind data shape:", wind_data.shape)
    return wind_data


# ----------------------------
# Step 3. Load Terrain Data
# ----------------------------
def load_terrain_data(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load terrain data from a GeoTIFF, normalize it to [0,1], and resize it.
    """
    with rasterio.open(filepath) as src:
        terrain = src.read(1)
    terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
    if terrain.shape != target_shape:
        terrain = resize(terrain, target_shape, anti_aliasing=True)
    return terrain.astype(np.float32)


# ----------------------------
# Step 4. Construct the Dataset for the MLP
# ----------------------------
class FirePredictionDataset(Dataset):
    """
    Each sample is:
       Input = [flattened current fire mask, wind vector (2 features), flattened terrain]
       Target = flattened fire mask at time t+1.
    """

    def __init__(
        self, fire_series: np.ndarray, terrain: np.ndarray, wind_data: np.ndarray
    ):
        self.inputs = []
        self.targets = []
        num_samples = fire_series.shape[0] - 1
        H, W = terrain.shape
        terrain_flat = terrain.flatten().astype(np.float32)
        for t in range(num_samples):
            fire_flat = fire_series[t].astype(np.float32).flatten()
            wind_vector = wind_data[t]  # dynamic wind vector for time step t
            input_features = np.concatenate([fire_flat, wind_vector, terrain_flat])
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
# Step 5. Define and Train the MLP
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
    criterion = nn.BCEWithLogitsLoss()
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
# Step 6. Iterative Prediction Using the Trained MLP
# ----------------------------
def predict_fire_series(
    model: nn.Module,
    initial_fire: np.ndarray,
    terrain: np.ndarray,
    wind_data: np.ndarray,
    num_timesteps: int,
) -> np.ndarray:
    """
    Starting from an initial fire mask, use the trained MLP iteratively to predict
    the fire evolution over num_timesteps. For each time step, the corresponding wind
    vector from wind_data is used.

    Returns:
        predicted_series: Array of shape (num_timesteps, H, W) with binary predictions.
    """
    H, W = terrain.shape
    predicted_series = []
    current_fire = initial_fire.copy()
    predicted_series.append(current_fire.copy())
    terrain_flat = terrain.flatten().astype(np.float32)
    device = next(model.parameters()).device

    for t in range(1, num_timesteps):
        fire_flat = current_fire.astype(np.float32).flatten()
        wind_vector = wind_data[t]  # use wind vector for time step t
        input_features = np.concatenate([fire_flat, wind_vector, terrain_flat])
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
# Step 7. 3D Animation of the Predicted Fire Series
# ----------------------------
def animate_fire_on_3d(
    terrain: np.ndarray,
    fire_series: np.ndarray,
    interval: int = 200,
    z_offset: float = 0.0,
    marker_size: int = 5,
    max_points: int = 300,
    elevation_scale: float = 200.0,
):
    """
    Create a 3D animation of the predicted fire spread over the terrain,
    scaling the elevation to make height differences more pronounced.

    Args:
        terrain (np.ndarray): 2D elevation array.
        fire_series (np.ndarray): Array of shape (num_timesteps, H, W) representing fire masks.
        interval (int): Delay between frames in milliseconds.
        z_offset (float): Vertical offset for the markers (set to 0.0 to place markers exactly on the terrain).
        marker_size (int): Size of the markers.
        max_points (int): Maximum number of fire points to display per frame.
        elevation_scale (float): Factor to multiply the terrain elevations by.
                                  (Values >1.0 exaggerate elevation differences.)
    """
    H, W = terrain.shape
    # Apply elevation scaling
    scaled_terrain = terrain * elevation_scale

    # Create coordinate arrays using integer indices.
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Plot the scaled terrain surface.
    surf = ax.plot_surface(
        X, Y, scaled_terrain, cmap="terrain", alpha=0.7, linewidth=0, antialiased=True
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Scaled Elevation")

    def get_scatter_data(fire_mask):
        indices = np.where(fire_mask)
        xs = indices[1]  # columns correspond to x
        ys = indices[0]  # rows correspond to y
        # Use the scaled terrain for the z-coordinate.
        zs = scaled_terrain[indices] + z_offset
        # Downsample if there are too many points.
        if len(xs) > max_points:
            sample_indices = np.random.choice(len(xs), size=max_points, replace=False)
            xs = xs[sample_indices]
            ys = ys[sample_indices]
            zs = zs[sample_indices]
        return xs, ys, zs

    # Initial fire overlay.
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

    # Set aspect ratio so that x and y dimensions match your data.
    ax.set_box_aspect((W, H, np.ptp(scaled_terrain)))  # Requires Matplotlib 3.3+

    plt.show()


# ----------------------------
# Main Function: Integration
# ----------------------------
def main():
    # --- User Settings ---
    # Folder containing satellite images (GeoTIFFs) of the 2025 LA fires.
    satellite_folder = "FireSatelliteData"
    # Path to terrain GeoTIFF.
    terrain_filepath = "output_GEBCOIceTopo.tif"
    # Start time for wind data.
    wind_start = datetime(2025, 1, 1)

    # --- Load Terrain ---
    terrain = load_terrain_data(
        terrain_filepath, target_shape=(GRID_HEIGHT, GRID_WIDTH)
    )
    print("Terrain loaded. Shape:", terrain.shape)

    # --- Load and Augment Satellite Fire Series ---
    fire_series = load_and_augment_satellite_fire_series(
        satellite_folder,
        num_intermediate=NUM_INTERMEDIATE,
        noise_sigma=NOISE_SIGMA,
        target_shape=(GRID_HEIGHT, GRID_WIDTH),
    )
    print("Augmented satellite fire series loaded. Shape:", fire_series.shape)

    # --- Fetch Wind Data ---
    # The number of wind steps should match the number of fire images.
    wind_data = get_wind_data(wind_start, fire_series.shape[0])
    print("Wind data fetched. Shape:", wind_data.shape)

    # --- Build Dataset and DataLoader ---
    # Use wind_data[0:-1] since each sample uses time t (input) and t+1 (target).
    dataset = FirePredictionDataset(fire_series, terrain, wind_data[:-1])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    H, W = terrain.shape
    input_dim = H * W + 2 + H * W  # flattened fire mask + wind (2) + flattened terrain
    output_dim = H * W  # next fire mask (flattened)
    hidden_dim = 256

    model = FireMLP(input_dim, hidden_dim, output_dim)
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
    initial_fire = fire_series[0].astype(np.uint8)
    predicted_series = predict_fire_series(
        model, initial_fire, terrain, wind_data, fire_series.shape[0]
    )
    print("Predicted fire series shape:", predicted_series.shape)

    # --- 3D Visualization of the Predicted Fire Series ---
    animate_fire_on_3d(
        terrain, predicted_series, interval=200, z_offset=0.0, marker_size=5
    )


if __name__ == "__main__":
    main()
