import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import rasterio
from skimage.transform import resize
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.animation as animation

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Grid and simulation parameters
GRID_HEIGHT = 64
GRID_WIDTH = 64
NUM_TIMESTEPS = 50  # For simulation & training
PREDICT_TIMESTEPS = 50  # For iterative prediction after training
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


#############################################
# STEP 1. Load Terrain Data from GeoTIFF
#############################################
def load_terrain_data(
    filepath: str, target_shape: tuple = (GRID_HEIGHT, GRID_WIDTH)
) -> np.ndarray:
    """
    Load terrain data from a GeoTIFF, normalize it to [0,1],
    and resize it to the target shape.
    """
    with rasterio.open(filepath) as src:
        terrain = src.read(1)  # Read first band (assumed elevation)
    # Normalize terrain values to [0, 1]
    terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
    # Resize if necessary
    if terrain.shape != target_shape:
        terrain = resize(terrain, target_shape, anti_aliasing=True)
    return terrain.astype(np.float32)


#############################################
# STEP 2. Simulate Fire Spread on Terrain
#############################################
def simulate_fire_on_terrain(
    terrain: np.ndarray,
    num_timesteps: int,
    wind: tuple = (0, 1),
    base_p: float = 0.2,
    alpha: float = 0.1,
    beta: float = 0.3,
) -> np.ndarray:
    """
    Simulate wildfire spread on terrain.

    The model ignites cells based on burning neighbors, local elevation differences
    (fire tends to spread uphill), and the alignment with a constant wind vector.

    Returns:
        fire_series: Boolean array of shape (num_timesteps, H, W)
    """
    H, W = terrain.shape
    fire_series = []
    # Initialize fire: a 3x3 square at the center
    current_fire = np.zeros((H, W), dtype=bool)
    center_i, center_j = H // 2, W // 2
    current_fire[center_i - 1 : center_i + 2, center_j - 1 : center_j + 2] = True
    fire_series.append(current_fire.copy())

    wind_vec = np.array(wind, dtype=float)
    norm_wind = np.linalg.norm(wind_vec)
    wind_norm = wind_vec / norm_wind if norm_wind != 0 else np.array([0.0, 0.0])

    # Offsets for 8-connected neighbors
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for t in range(1, num_timesteps):
        p_total = np.zeros((H, W), dtype=float)
        for di, dj in offsets:
            # Define slices to avoid index errors
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
            # Elevation difference: target cell minus neighbor cell elevation
            dZ = terrain[tgt_rows, tgt_cols] - terrain[src_rows, src_cols]
            # Alignment: dot product of normalized neighbor offset with wind direction
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

    return np.array(fire_series)  # Shape: (num_timesteps, H, W)


#############################################
# STEP 3. Create Dataset for Fire Prediction
#############################################
class FirePredictionDataset(Dataset):
    """
    Each sample:
      Input: [flattened current fire mask, wind (2 features), flattened terrain]
      Target: flattened next fire mask.
    """

    def __init__(self, fire_series: np.ndarray, terrain: np.ndarray, wind: tuple):
        self.inputs = []
        self.targets = []
        num_samples = fire_series.shape[0] - 1
        H, W = terrain.shape
        self.fire_size = H * W
        terrain_flat = terrain.flatten().astype(np.float32)
        wind_array = np.array(wind, dtype=np.float32)
        for t in range(num_samples):
            fire_flat = fire_series[t].astype(np.float32).flatten()
            input_features = np.concatenate([fire_flat, wind_array, terrain_flat])
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
# STEP 4. Define the MLP for Fire Prediction
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
            loss = criterion(outputs, targets) * 1000
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


#############################################
# STEP 5. Iterative Prediction Using the Trained MLP
#############################################
def predict_fire_series(
    model: nn.Module,
    initial_fire: np.ndarray,
    terrain: np.ndarray,
    wind: tuple,
    num_timesteps: int,
) -> np.ndarray:
    """
    Starting from an initial fire mask, use the trained MLP iteratively to predict
    the fire evolution over a number of time steps.

    Returns:
        predicted_series: Array of shape (num_timesteps, H, W) of binary predictions.
    """
    H, W = terrain.shape
    predicted_series = []
    current_fire = initial_fire.copy()  # binary mask
    predicted_series.append(current_fire.copy())
    terrain_flat = terrain.flatten().astype(np.float32)
    wind_array = np.array(wind, dtype=np.float32)
    device = next(model.parameters()).device

    for t in range(1, num_timesteps):
        fire_flat = current_fire.astype(np.float32).flatten()
        input_features = np.concatenate([fire_flat, wind_array, terrain_flat])
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
# STEP 6. 3D Visualization of the Predicted Fire Series
#############################################
def animate_fire_on_3d(
    terrain: np.ndarray,
    fire_series: np.ndarray,
    interval: int = 200,
    z_offset: float = 0.01,
    marker_size: int = 5,
):
    """
    Create a 3D animation of the fire spread over the terrain.

    The terrain is rendered as a 3D surface, and predicted fire cells are overlaid as small red markers.
    The markers are placed very close to the terrain surface (using a small z_offset).

    Args:
        terrain (np.ndarray): 2D elevation array.
        fire_series (np.ndarray): Array of shape (num_timesteps, H, W) representing the fire masks.
        interval (int): Delay between frames in milliseconds.
        z_offset (float): Minimal vertical offset added to fire markers.
        marker_size (int): Size of the markers representing fire cells.
    """
    H, W = terrain.shape
    # Create coordinate arrays using integer indices so they exactly match the grid indices.
    x = np.arange(W)  # 0, 1, ..., W-1
    y = np.arange(H)  # 0, 1, ..., H-1
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the terrain surface.
    surf = ax.plot_surface(
        X, Y, terrain, cmap="terrain", alpha=0.7, linewidth=0, antialiased=True
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation")

    # Plot the initial fire overlay.
    fire0 = fire_series[0]
    fire_indices = np.where(fire0)
    xs = fire_indices[1]  # column indices → x
    ys = fire_indices[0]  # row indices → y
    zs = terrain[fire_indices] + z_offset  # add a very small offset
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
    # Update this path to your terrain GeoTIFF file
    tif_filepath = "output_GEBCOIceTopo.tif"
    terrain = load_terrain_data(tif_filepath, target_shape=(GRID_HEIGHT, GRID_WIDTH))
    print("Terrain loaded. Shape:", terrain.shape)

    # Define constant wind vector (dy, dx); e.g., wind blowing east
    wind = (0, 1)

    # Step 2: Simulate fire spread (used for training)
    fire_series = simulate_fire_on_terrain(
        terrain, NUM_TIMESTEPS, wind=wind, base_p=0.2, alpha=0.1, beta=0.3
    )
    print("Fire simulation complete. Fire series shape:", fire_series.shape)

    # Step 3: Create dataset for training
    dataset = FirePredictionDataset(fire_series, terrain, wind)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Compute dimensions
    H, W = terrain.shape
    fire_size = H * W
    input_dim = fire_size + 2 + fire_size  # current fire + wind (2) + terrain
    output_dim = fire_size  # next fire mask
    hidden_dim = 256

    # Step 4: Define and train the MLP
    model = FireMLP(input_dim, hidden_dim, output_dim)
    print(model)
    train_model(model, dataloader, NUM_EPOCHS, LEARNING_RATE)

    # (Optional) Evaluate on one sample in 2D
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

    # Step 5: Use the trained MLP to iteratively predict a fire series.
    # Use the same initial fire mask as in simulation.
    initial_fire = fire_series[0].astype(np.uint8)
    predicted_series = predict_fire_series(
        model, initial_fire, terrain, wind, PREDICT_TIMESTEPS
    )
    print("Predicted fire series shape:", predicted_series.shape)

    # Step 6: Visualize the predicted fire series in 3D over the terrain.
    animate_fire_on_3d(terrain, predicted_series, interval=200, z_offset=0.1)


if __name__ == "__main__":
    main()
