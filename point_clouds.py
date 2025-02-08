#!/usr/bin/env python3
"""
wildfire_prediction_with_progress.py

This script simulates wildfire spread as a time series of binary matrices,
creates a dataset where each sample consists of a flattened fire mask plus
weather features, and trains a Multi-Layer Perceptron (MLP) to predict the
next time step's fire mask. Progress tracking is added using tqdm.

In a real project you would replace the simulation code with data ingestion
and processing for satellite imagery, terrain, and weather data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar library

# Constants
GRID_HEIGHT = 64
GRID_WIDTH = 64
NUM_TIMESTEPS = 100  # number of time steps in the simulation
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def simulate_fire_spread(num_timesteps: int, grid_shape: tuple, wind: tuple = (1, 0)):
    """
    Simulate wildfire spread as a time series of binary matrices.

    Args:
        num_timesteps (int): Number of time steps to simulate.
        grid_shape (tuple): (height, width) of the grid.
        wind (tuple): (dy, dx) wind vector applied as an offset each timestep.

    Returns:
        fire_series (np.ndarray): Array of shape (num_timesteps, height, width) containing binary fire masks.
        weather_series (np.ndarray): Array of shape (num_timesteps, 2) containing weather features (wind dy, wind dx).
                                     For simplicity, we use constant wind in this simulation.
    """
    height, width = grid_shape
    fire_series = np.zeros((num_timesteps, height, width), dtype=np.uint8)
    weather_series = np.zeros((num_timesteps, 2), dtype=np.float32)

    # Initialize fire: a small square in the center
    init_fire = np.zeros((height, width), dtype=np.uint8)
    center_y, center_x = height // 2, width // 2
    init_fire[center_y - 2 : center_y + 3, center_x - 2 : center_x + 3] = (
        1  # 5x5 square
    )
    fire_series[0] = init_fire

    for t in range(num_timesteps):
        weather_series[t] = wind

    # Define structuring element for dilation (neighbors connectivity)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    # Use tqdm to track progress during simulation
    for t in tqdm(range(1, num_timesteps), desc="Simulating fire spread", leave=False):
        dilated_fire = binary_dilation(fire_series[t - 1], structure=struct).astype(
            np.uint8
        )
        shifted_fire = np.roll(dilated_fire, shift=wind, axis=(0, 1))
        noise = (
            np.random.rand(height, width) > 0.98
        )  # small chance to extinguish a pixel
        new_fire = np.where(noise, 0, shifted_fire)
        fire_series[t] = new_fire

    return fire_series, weather_series


class WildfireDataset(Dataset):
    """
    PyTorch Dataset for wildfire simulation.

    Each sample consists of:
      - Input: the flattened current fire mask concatenated with weather features.
      - Target: the flattened next time step fire mask.
    """

    def __init__(self, fire_series: np.ndarray, weather_series: np.ndarray):
        """
        Args:
            fire_series (np.ndarray): Array of shape (T, H, W) containing binary fire masks.
            weather_series (np.ndarray): Array of shape (T, 2) containing weather features.
        """
        # Create pairs: (fire[t], weather[t]) -> fire[t+1]
        self.inputs = []
        self.targets = []
        num_samples = fire_series.shape[0] - 1
        self.grid_size = fire_series.shape[1] * fire_series.shape[2]

        for t in range(num_samples):
            fire_flat = fire_series[t].flatten().astype(np.float32)
            weather_features = weather_series[t]
            input_features = np.concatenate([fire_flat, weather_features])
            target_flat = fire_series[t + 1].flatten().astype(np.float32)
            self.inputs.append(input_features)
            self.targets.append(target_flat)
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


class FireMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) for predicting wildfire spread.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(FireMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_model(
    model: nn.Module, dataloader: DataLoader, num_epochs: int, learning_rate: float
) -> None:
    """
    Train the MLP model using BCEWithLogitsLoss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        # Wrap the dataloader with tqdm to show progress for each epoch
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


def visualize_fire_mask(fire_mask: np.ndarray, title: str = "Fire Mask") -> None:
    """
    Visualize a fire mask using matplotlib.
    """
    plt.imshow(fire_mask, cmap="hot", interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.show()


def main() -> None:
    # Simulate wildfire spread data
    fire_series, weather_series = simulate_fire_spread(
        NUM_TIMESTEPS, (GRID_HEIGHT, GRID_WIDTH), wind=(1, 0)
    )
    print(f"Simulated fire series shape: {fire_series.shape}")
    print(f"Simulated weather series shape: {weather_series.shape}")

    # Visualize the initial and final fire masks
    visualize_fire_mask(fire_series[0], title="Initial Fire Mask")
    visualize_fire_mask(fire_series[-1], title="Final Fire Mask")

    # Create dataset and dataloader
    dataset = WildfireDataset(fire_series, weather_series)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define model dimensions:
    # Input: flattened fire mask (GRID_HEIGHT x GRID_WIDTH) + 2 weather features
    input_dim = GRID_HEIGHT * GRID_WIDTH + 2
    hidden_dim = 256
    output_dim = GRID_HEIGHT * GRID_WIDTH  # predicting the next fire mask (flattened)

    model = FireMLP(input_dim, hidden_dim, output_dim)
    print(model)

    # Train the model
    train_model(model, dataloader, NUM_EPOCHS, LEARNING_RATE)

    # Test prediction on a sample from the dataset
    model.eval()
    sample_input, sample_target = dataset[0]
    with torch.no_grad():
        sample_input = sample_input.unsqueeze(0)
        output_logits = model(sample_input)
        output_probs = torch.sigmoid(output_logits).squeeze().cpu().numpy()
    # Threshold probabilities to obtain a binary prediction
    prediction = (output_probs > 0.5).astype(np.uint8).reshape(GRID_HEIGHT, GRID_WIDTH)
    ground_truth = sample_target.numpy().reshape(GRID_HEIGHT, GRID_WIDTH)

    visualize_fire_mask(prediction, title="Predicted Fire Mask (Next Time Step)")
    visualize_fire_mask(ground_truth, title="Ground Truth Fire Mask (Next Time Step)")


if __name__ == "__main__":
    main()
