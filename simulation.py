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
from skimage.morphology import erosion, square
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
GRID_WIDTH = 256

# Augmentation parameters
NUM_INTERMEDIATE = (
    10  # number of intermediate images to generate between consecutive frames
)
NOISE_SIGMA = 0.1  # standard deviation of Gaussian noise

# Training parameters
BATCH_SIZE = 5
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ----------------------------
# New Functions for Canopy and Slope
# ----------------------------
def load_canopy_data(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load canopy data from a GeoTIFF file, normalize it to [0,1], and resize.
    Assumes the canopy values are given as percentages (0-100).
    """
    with rasterio.open(filepath) as src:
        canopy = src.read(1)
    canopy = canopy / 100.0
    if canopy.shape != target_shape:
        canopy = resize(canopy, target_shape, anti_aliasing=True)
    return canopy.astype(np.float32)


def compute_slope(terrain: np.ndarray) -> np.ndarray:
    """
    Compute the slope from the terrain data using gradients.
    Normalize the slope to [0,1] for use as an input feature.
    """
    dy, dx = np.gradient(terrain)
    slope = np.sqrt(dx**2 + dy**2)
    slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)
    return slope_norm.astype(np.float32)


# ----------------------------
# Step 1. Load and Augment Satellite Images (Fire Masks)
# ----------------------------
def load_satellite_fire_mask(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load a satellite image from a GeoTIFF file and convert it into a binary fire mask
    using Otsu thresholding.
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
            interpolated = (1 - alpha) * img1 + alpha * img2
            noisy = interpolated + np.random.normal(0, noise_sigma, interpolated.shape)
            noisy = np.clip(noisy, 0, 1)
            binary_noisy = (noisy > 0.5).astype(np.uint8)
            augmented_series.append(binary_noisy)
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
# Step 3. Load Terrain and Canopy Data; Compute Slope
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
       Input = [flattened current fire mask, wind vector (2), flattened terrain,
                flattened canopy, flattened slope]
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
    canopy: np.ndarray,
    slope: np.ndarray,
    wind_data: np.ndarray,
    num_timesteps: int,
) -> np.ndarray:
    """
    Starting from an initial fire mask, use the trained MLP iteratively to predict
    the fire evolution over num_timesteps. The input features include the static terrain,
    canopy, slope, and the dynamic wind vector.

    Returns:
        predicted_series: Array of shape (num_timesteps, H, W) with binary predictions.
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
        wind_vector = wind_data[t]  # use wind vector for time step t
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
# Step 7. 3D Animation of the Predicted Fire Series
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
    """
    Create a 3D animation of the predicted fire spread over the terrain,
    scaling the elevation to exaggerate differences.

    Args:
        terrain (np.ndarray): 2D elevation array.
        fire_series (np.ndarray): Array of shape (num_timesteps, H, W) representing fire masks.
        interval (int): Delay between frames in milliseconds.
        z_offset (float): Vertical offset for markers (0.0 places them exactly on the terrain).
        marker_size (int): Size of the markers.
        max_points (int): Maximum number of fire points to display per frame.
        elevation_scale (float): Factor to scale the terrain elevation.
    """
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
# Main Function: Integration
# ----------------------------
def main():
    # --- User Settings ---
    # Folder containing satellite images (GeoTIFFs) of the 2025 LA fires.
    satellite_folder = "FireSatelliteData"
    # Path to terrain GeoTIFF.
    terrain_filepath = "output_GEBCOIceTopo.tif"
    # Path to canopy GeoTIFF.
    canopy_filepath = "output_GEBCOIceTopo.tif"
    # Start time for wind data.
    wind_start = datetime(2025, 1, 1)

    # --- Load Terrain and Canopy ---
    terrain = load_terrain_data(
        terrain_filepath, target_shape=(GRID_HEIGHT, GRID_WIDTH)
    )
    print("Terrain loaded. Shape:", terrain.shape)

    canopy = load_canopy_data(canopy_filepath, target_shape=(GRID_HEIGHT, GRID_WIDTH))
    print("Canopy data loaded. Shape:", canopy.shape)

    # Compute slope from terrain.
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
    # Use wind_data[0:-1] since each sample uses time t (input) and t+1 (target).
    dataset = FirePredictionDataset(fire_series, terrain, canopy, slope, wind_data[:-1])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    H, W = terrain.shape
    # Input dimension: fire mask (H*W) + wind (2) + terrain (H*W) + canopy (H*W) + slope (H*W)
    input_dim = H * W + 2 + H * W + H * W + H * W
    output_dim = H * W
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
    # Instead of starting with the satellite image fire mask, we initialize with a single ignition point.
    initial_fire = np.zeros((H, W), dtype=np.uint8)
    # Set the center cell to 1 (or choose another cell as needed).
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
        z_offset=0.1,
        marker_size=5,
        elevation_scale=100.0,
        max_points=1000,
    )


if __name__ == "__main__":
    main()
