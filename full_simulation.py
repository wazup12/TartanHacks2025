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
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import matplotlib.animation as animation

# Import Meteostat modules
from meteostat import Point, Hourly

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Grid and simulation parameters
GRID_HEIGHT = 128
GRID_WIDTH = 128
NUM_TIMESTEPS = 240  # total simulation steps (hours)
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


#############################################
# Step 1. Get Real Wind Data Using Meteostat
#############################################
def get_wind_data(start: datetime, num_steps: int) -> np.ndarray:
    """
    Fetch hourly wind data (wind speed and direction) for Los Angeles
    for a period of num_steps hours starting at 'start'.

    Converts the meteorological wind direction (the direction from which
    the wind comes, in degrees) and wind speed (m/s) into a 2D vector (dy, dx)
    representing the blowing direction. In our simulation:
      - Positive dy means wind blowing toward increasing row indices (south).
      - Positive dx means wind blowing toward increasing column indices (east).

    Since meteorological wind direction is the direction from which the wind
    originates, we add 180° to obtain the direction the wind is blowing toward.

    Args:
        start (datetime): Start time for the data.
        num_steps (int): Number of hourly steps.

    Returns:
        wind_data (np.ndarray): Array of shape (num_steps, 2) where each row is (dy, dx).
    """
    end = start + timedelta(hours=num_steps)
    la_point = Point(34.0494, -118.5236)
    data = Hourly(la_point, start, end)
    df = data.fetch()

    # Ensure we have at least num_steps rows (if not, repeat or truncate accordingly)
    df = df.iloc[:num_steps]

    wind_vectors = []
    for _, row in df.iterrows():
        wspd = row["wspd"]  # wind speed in m/s
        wdir = row["wdir"]  # wind direction in degrees (from which wind comes)
        # Compute the effective blowing direction (in degrees): add 180°.
        effective_dir = (wdir + 180) % 360
        # Convert to radians.
        rad = np.deg2rad(effective_dir)
        # In our grid coordinate system, let:
        # dx = wind speed * sin(rad)  (eastward component)
        # dy = wind speed * cos(rad)  (southward component)
        dx = wspd * np.sin(rad)
        dy = wspd * np.cos(rad)
        wind_vectors.append([dy, dx])
    wind_data = np.array(wind_vectors, dtype=np.float32)
    print("Fetched wind data shape:", wind_data.shape)
    return wind_data


#############################################
# Step 2. Load Terrain Data from GeoTIFF
#############################################
def load_terrain_data(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load terrain data from a GeoTIFF, normalize it to [0,1],
    and resize it to the target shape.
    """
    with rasterio.open(filepath) as src:
        terrain = src.read(1)  # Read first band (assumed to be elevation)
    terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
    if terrain.shape != target_shape:
        terrain = resize(terrain, target_shape, anti_aliasing=True)
    return terrain.astype(np.float32)


#############################################
# Step 3. Simulate Fire Spread with Dynamic Wind Data
#############################################
def simulate_fire_on_terrain(
    terrain: np.ndarray,
    wind_data: np.ndarray,
    base_p: float = 0.2,
    alpha: float = 0.1,
    beta: float = 0.3,
) -> np.ndarray:
    """
    Simulate wildfire spread on terrain using time-varying wind data.

    The simulation starts with a 3x3 ignition region in the center. At each time step,
    the ignition probability for each cell is influenced by the burning neighbors,
    local elevation differences, and the wind vector for that time step.

    Args:
        terrain (np.ndarray): 2D elevation array.
        wind_data (np.ndarray): Array of shape (num_timesteps, 2) for each time step.
        base_p, alpha, beta: parameters for ignition probability.

    Returns:
        fire_series (np.ndarray): Boolean array of shape (num_timesteps, H, W).
    """
    num_timesteps = wind_data.shape[0]
    H, W = terrain.shape
    fire_series = []
    current_fire = np.zeros((H, W), dtype=bool)
    center_i, center_j = H // 2, W // 2
    current_fire[center_i - 1 : center_i + 2, center_j - 1 : center_j + 2] = True
    fire_series.append(current_fire.copy())

    # Offsets for 8-connected neighbors
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for t in range(1, num_timesteps):
        current_wind = wind_data[t]  # wind vector for this time step
        norm_wind = np.linalg.norm(current_wind)
        wind_norm = current_wind / norm_wind if norm_wind != 0 else np.array([0.0, 0.0])
        p_total = np.zeros((H, W), dtype=float)
        for di, dj in offsets:
            # Define slices for neighbor contribution
            if di < 0:
                src_rows = slice(0, H + di)
                tgt_rows = slice(-di, H)
            elif di > 0:
                src_rows = slice(di, H)
                tgt_rows = slice(0, H - di)
            else:
                src_rows = slice(0, H)
                tgt_rows = slice(0, H)
            if dj < 0:
                src_cols = slice(0, W + dj)
                tgt_cols = slice(-dj, W)
            elif dj > 0:
                src_cols = slice(dj, W)
                tgt_cols = slice(0, W - dj)
            else:
                src_cols = slice(0, W)
                tgt_cols = slice(0, W)

            neighbor_fire = current_fire[src_rows, src_cols]
            # Elevation difference: target cell minus neighbor cell elevation.
            dZ = terrain[tgt_rows, tgt_cols] - terrain[src_rows, src_cols]
            offset_vec = np.array([di, dj], dtype=float)
            norm_offset = np.linalg.norm(offset_vec)
            dot = (
                np.dot(offset_vec / norm_offset, wind_norm) if norm_offset != 0 else 0.0
            )
            p = base_p * (1 + alpha * dZ) * (1 + beta * dot)
            p = np.clip(p, 0, 1)
            p_effective = p * neighbor_fire.astype(float)
            p_total[tgt_rows, tgt_cols] = 1 - (1 - p_total[tgt_rows, tgt_cols]) * (
                1 - p_effective
            )
        random_vals = np.random.rand(H, W)
        new_ignitions = (random_vals < p_total) & (~current_fire)
        current_fire = current_fire | new_ignitions
        fire_series.append(current_fire.copy())

    return np.array(fire_series)  # shape: (num_timesteps, H, W)


#############################################
# Step 4. Create Dataset for Fire Prediction with Dynamic Wind
#############################################
class FirePredictionDataset(Dataset):
    """
    For each time step t, create a sample:
      Input: [flattened current fire mask, wind vector (2 features), flattened terrain]
      Target: flattened fire mask at time t+1.

    The wind vector is taken from wind_data[t]. Therefore, wind_data should be an array
    of shape (num_samples, 2) where num_samples = number of simulation steps - 1.
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
            wind_vector = wind_data[t]  # dynamic wind vector for this sample
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


#############################################
# Step 5. Define the MLP for Fire Prediction
#############################################
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
        for inputs, targets in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ):
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


#############################################
# Step 6. Iterative Prediction Using the Trained MLP
#############################################
def predict_fire_series(
    model: nn.Module,
    initial_fire: np.ndarray,
    terrain: np.ndarray,
    wind_data: np.ndarray,
    num_timesteps: int,
) -> np.ndarray:
    """
    Starting from an initial fire mask, use the trained MLP iteratively to predict
    the fire evolution. For each step, the corresponding wind vector is taken from wind_data.

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
        # Use wind_data[t] for this step. (Ensure wind_data has length >= num_timesteps.)
        wind_vector = wind_data[t]
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


#############################################
# Step 7. 3D Visualization of the Predicted Fire Series
#############################################
def animate_fire_on_3d(
    terrain: np.ndarray,
    fire_series: np.ndarray,
    interval: int = 200,
    z_offset: float = 0.0,
    marker_size: int = 5,
):
    """
    Create a 3D animation of the predicted fire spread over the terrain.

    Args:
        terrain (np.ndarray): 2D elevation array.
        fire_series (np.ndarray): Array of shape (num_timesteps, H, W).
        interval (int): Delay between frames in milliseconds.
        z_offset (float): Vertical offset (set to 0.0 to place markers exactly on the terrain).
        marker_size (int): Size of the markers.
    """
    H, W = terrain.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, terrain, cmap="terrain", alpha=0.7, linewidth=0, antialiased=True
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation")

    # Initial fire overlay
    fire0 = fire_series[0]
    fire_indices = np.where(fire0)
    xs = fire_indices[1]
    ys = fire_indices[0]
    zs = terrain[fire_indices] + z_offset
    sc = ax.scatter(xs, ys, zs, c="r", marker="o", s=marker_size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    ax.set_title("Predicted 3D Fire Spread over Terrain")

    def update(frame):
        fire_mask = fire_series[frame]
        fire_indices = np.where(fire_mask)
        xs = fire_indices[1]
        ys = fire_indices[0]
        zs = terrain[fire_indices] + z_offset
        sc._offsets3d = (xs, ys, zs)
        ax.set_title(f"Predicted 3D Fire Spread - Time Step {frame}")
        return (sc,)

    ani = animation.FuncAnimation(
        fig, update, frames=fire_series.shape[0], interval=interval, blit=False
    )
    plt.show()


#############################################
# MAIN FUNCTION: Integrate All Steps
#############################################
def main():
    tif_filepath = "output_GEBCOIceTopo.tif"
    # Start time for wind data (choose an arbitrary start date)
    wind_start = datetime(2025, 1, 7)

    # --- Load terrain ---
    terrain = load_terrain_data(tif_filepath, target_shape=(GRID_HEIGHT, GRID_WIDTH))
    print("Terrain loaded. Shape:", terrain.shape)

    # --- Get dynamic wind data ---
    wind_data = get_wind_data(wind_start, NUM_TIMESTEPS)

    # --- Simulate fire spread with dynamic wind ---
    fire_series = simulate_fire_on_terrain(
        terrain, wind_data, base_p=0.2, alpha=0.1, beta=0.3
    )
    print("Fire simulation complete. Fire series shape:", fire_series.shape)

    # --- Create dataset for MLP training ---
    # For training, we use fire_series[0:-1] as inputs and fire_series[1:] as targets.
    # Accordingly, use wind_data[0:-1] for each sample.
    dataset = FirePredictionDataset(fire_series, terrain, wind_data[:-1])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    H, W = terrain.shape
    fire_size = H * W
    input_dim = fire_size + 2 + fire_size  # current fire mask + wind (2) + terrain
    output_dim = fire_size  # next fire mask
    hidden_dim = 256

    model = FireMLP(input_dim, hidden_dim, output_dim)
    print(model)

    # --- Train the MLP ---
    train_model(model, dataloader, NUM_EPOCHS, LEARNING_RATE)

    # --- (Optional) Evaluate on one sample in 2D ---
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

    # --- Use the trained MLP for iterative prediction ---
    initial_fire = fire_series[0].astype(np.uint8)
    predicted_series = predict_fire_series(
        model, initial_fire, terrain, wind_data, NUM_TIMESTEPS
    )
    print("Predicted fire series shape:", predicted_series.shape)

    # --- 3D Visualization of the predicted fire spread ---
    animate_fire_on_3d(
        terrain, predicted_series, interval=200, z_offset=0.1, marker_size=5
    )


if __name__ == "__main__":
    main()
