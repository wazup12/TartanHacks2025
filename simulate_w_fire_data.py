from datetime import datetime, timedelta
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

# Meteostat for wind data
from meteostat import Point, Hourly

# -----------------------------
# Parameters and Settings
# -----------------------------
GRID_HEIGHT = 64
GRID_WIDTH = 64
NUM_TIMESTEPS = 50  # Total number of time steps (images)
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -----------------------------
# Step 1. Load and Process Satellite Imagery for Fire Masks
# -----------------------------
def load_satellite_fire_mask(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load a satellite image from a GeoTIFF file and convert it into a binary fire mask.

    This example uses Otsu thresholding (from skimage) to separate fire from non-fire.
    (In practice, you might use more sophisticated indices such as NBR, NDVI, or thermal thresholds.)

    Args:
        filepath (str): Path to the satellite image GeoTIFF.
        target_shape (tuple): Desired output shape.

    Returns:
        fire_mask (np.ndarray): Binary mask with 1 indicating fire.
    """
    with rasterio.open(filepath) as src:
        image = src.read(
            1
        )  # Assume band 1 contains useful information for fire detection
    # Resize image if necessary
    if image.shape != target_shape:
        image = resize(image, target_shape, anti_aliasing=True)
    # Use Otsu thresholding to separate fire (assumed to have higher values) from non-fire.
    thresh = threshold_otsu(image)
    fire_mask = (image > thresh).astype(np.uint8)
    return fire_mask


def load_satellite_fire_series(
    base_filepath: str,
    num_timesteps: int,
    target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH),
) -> np.ndarray:
    """
    Load a series of satellite images (one per time step) and convert them to binary fire masks.

    The base_filepath should be a Python format string containing a placeholder for the time index.
    For example: "data/la_fires_2025_t{}.tif"

    Returns:
        fire_series (np.ndarray): Array of shape (num_timesteps, H, W) with binary masks.
    """
    fire_series = []
    for i in range(num_timesteps):
        filepath = base_filepath.format(i)
        fire_mask = load_satellite_fire_mask(filepath, target_shape)
        fire_series.append(fire_mask)
    return np.array(fire_series)


# -----------------------------
# Step 2. Get Real Wind Data for Los Angeles Using Meteostat
# -----------------------------
def get_wind_data(start: datetime, num_steps: int) -> np.ndarray:
    """
    Fetch hourly wind data (wind speed and direction) for Los Angeles using Meteostat.

    Converts meteorological wind data (where wind direction is the direction from which the wind comes)
    into a vector (dy, dx) indicating the blowing direction.

    Args:
        start (datetime): Start time.
        num_steps (int): Number of hourly steps.

    Returns:
        wind_data (np.ndarray): Array of shape (num_steps, 2) where each row is (dy, dx).
    """
    end = start + timedelta(hours=num_steps)
    la_point = Point(34.0522, -118.2437)
    data = Hourly(la_point, start, end)
    df = data.fetch().iloc[:num_steps]
    wind_vectors = []
    for _, row in df.iterrows():
        wspd = row["wspd"]  # wind speed in m/s
        wdir = row["wdir"]  # wind direction in degrees (from which wind comes)
        # Convert to blowing direction by adding 180° (modulo 360)
        effective_dir = (wdir + 180) % 360
        rad = np.deg2rad(effective_dir)
        # In our grid, assume:
        # dx = wspd * sin(rad) (eastward component)
        # dy = wspd * cos(rad) (southward component)
        dx = wspd * np.sin(rad)
        dy = wspd * np.cos(rad)
        wind_vectors.append([dy, dx])
    wind_data = np.array(wind_vectors, dtype=np.float32)
    print("Fetched wind data shape:", wind_data.shape)
    return wind_data


# -----------------------------
# Step 3. Load Terrain Data
# -----------------------------
def load_terrain_data(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load terrain data from a GeoTIFF, normalize to [0,1], and resize.
    """
    with rasterio.open(filepath) as src:
        terrain = src.read(1)
    terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
    if terrain.shape != target_shape:
        terrain = resize(terrain, target_shape, anti_aliasing=True)
    return terrain.astype(np.float32)


# -----------------------------
# Step 4. Construct the Dataset for the MLP
# -----------------------------
class FirePredictionDataset(Dataset):
    """
    Each sample is:
       Input = [flattened current fire mask, wind vector (2 features), flattened terrain]
       Target = flattened next fire mask.

    Assumes that fire_series is an array of shape (num_timesteps, H, W) and
    wind_data is an array of shape (num_timesteps, 2) (one per time step).
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
            wind_vector = wind_data[t]  # wind vector for this time step
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


# -----------------------------
# Step 5. Define and Train the MLP
# -----------------------------
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


def main():
    # --- User settings ---
    # Path to terrain data (GeoTIFF)
    terrain_filepath = "output_GEBCOIceTopo.tif"

    # Base filepath for satellite images of the 2025 LA fires.
    # The images should be named like "la_fires_2025_t0.tif", "la_fires_2025_t1.tif", ..., etc.
    satellite_base_filepath = (
        "path/to/your/la_fires_2025_t{}.tif"  # UPDATE with your path pattern
    )

    # Start time for wind data (adjust as needed)
    wind_start = datetime(2025, 1, 1)

    # --- Load Data ---
    terrain = load_terrain_data(
        terrain_filepath, target_shape=(GRID_HEIGHT, GRID_WIDTH)
    )
    print("Terrain loaded. Shape:", terrain.shape)

    fire_series = load_satellite_fire_series(
        satellite_base_filepath, NUM_TIMESTEPS, target_shape=(GRID_HEIGHT, GRID_WIDTH)
    )
    print("Satellite fire series loaded. Shape:", fire_series.shape)

    wind_data = get_wind_data(wind_start, NUM_TIMESTEPS)
    print("Wind data fetched. Shape:", wind_data.shape)

    # --- Build Dataset and DataLoader ---
    # We use wind_data[0:-1] since each sample corresponds to time t (input) and time t+1 (target).
    dataset = FirePredictionDataset(fire_series, terrain, wind_data[:-1])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    H, W = terrain.shape
    input_dim = (
        H * W + 2 + H * W
    )  # fire mask (flattened) + wind vector (2) + terrain (flattened)
    output_dim = H * W  # next fire mask (flattened)
    hidden_dim = 256

    model = FireMLP(input_dim, hidden_dim, output_dim)
    print(model)

    # --- Train the MLP ---
    train_model(model, dataloader, NUM_EPOCHS, LEARNING_RATE)

    # --- Evaluate on one sample (2D visualization) ---
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


if __name__ == "__main__":
    main()
