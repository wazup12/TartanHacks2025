import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for GIF generation
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from scipy.spatial.distance import euclidean
from scipy.fftpack import fft2, fftshift
import argparse
import time
import os
import sys
import imageio

###############################################################################
#  DP-GMM and TSP functions (as before)
###############################################################################
class DirichletGMM:
    def __init__(self, max_components=10, weight_threshold=1e-3, plot_gaussians=False):
        self.max_components = max_components
        self.weight_threshold = weight_threshold
        self.plot_gaussians = plot_gaussians
        self.model = BayesianGaussianMixture(
            n_components=self.max_components,
            weight_concentration_prior_type='dirichlet_process',
            covariance_type='full'
        )

    def fit(self, info_map):
        height, width = info_map.shape
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        data = np.column_stack((X.ravel(), Y.ravel(), info_map.ravel()))
        data = data[data[:, 2] > 0]  # use only nonzero points

        num_samples = min(5000, data.shape[0])
        sampled_indices = np.random.choice(
            len(data), size=num_samples, p=data[:, 2] / np.sum(data[:, 2])
        )
        sampled_data = data[sampled_indices, :2]
        self.model.fit(sampled_data)
        gaussians = self.get_gaussians()
        if self.plot_gaussians:
            self.plot_results(info_map, gaussians, info_map.shape[0], info_map.shape[1])
        return gaussians

    def get_gaussians(self):
        gaussians = []
        for i in range(self.model.n_components):
            if self.model.weights_[i] > self.weight_threshold:
                gaussians.append({
                    'weight': self.model.weights_[i],
                    'mean': self.model.means_[i],
                    'covariance': self.model.covariances_[i]
                })
        return gaussians

    def fourier_magnitude(self, gaussian, size=50):
        """
        Compute the normalized 2D Fourier transform magnitude of a Gaussian kernel.
        (This acts as the spectral signature of the component.)
        """
        mean = gaussian['mean']
        cov = gaussian['covariance']
        X, Y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        pos = np.dstack((X, Y))
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        diff = pos - mean
        exponent = -0.5 * np.einsum('ijk,kl,ijl->ij', diff, inv_cov, diff)
        gaussian_kernel = np.exp(exponent)
        fft_magnitude = np.abs(fftshift(fft2(gaussian_kernel)))
        return fft_magnitude / np.linalg.norm(fft_magnitude)

    def compute_similarity(self, g1, g2):
        """Compute similarity between two Gaussians using their Fourier magnitudes."""
        mag1 = self.fourier_magnitude(g1)
        mag2 = self.fourier_magnitude(g2)
        return np.dot(mag1.ravel(), mag2.ravel()) / (np.linalg.norm(mag1) * np.linalg.norm(mag2))

    def tsp_path(self, gaussians):
        """
        Compute a TSP ordering of the Gaussian nodes using a weighted cost function.
        The cost between two nodes is:
            cost = Euclidean_distance * (1 - Fourier_similarity)
        (Here we use a nearest-neighbor heuristic.)
        """
        n = len(gaussians)
        cost_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidean(gaussians[i]['mean'], gaussians[j]['mean'])
                similarity = self.compute_similarity(gaussians[i], gaussians[j])
                cost_matrix[i, j] = cost_matrix[j, i] = d * (1 - similarity)
        path = [0]
        unvisited = set(range(1, n))
        while unvisited:
            last = path[-1]
            next_node = min(unvisited, key=lambda node: cost_matrix[last, node])
            unvisited.remove(next_node)
            path.append(next_node)
        return path

    def plot_results(self, info_map, gaussians, height, width):
        # (For debugging: a one-time plot of the overall density.)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(info_map, cmap='plasma', origin='lower')
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        gaussian_map = np.zeros((height, width))
        for g in gaussians:
            weight = g['weight']
            mean = g['mean']
            cov = g['covariance']
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)
            diff = np.dstack((X - mean[0], Y - mean[1]))
            exponent = -0.5 * np.einsum('ijk,kl,ijl->ij', diff, inv_cov, diff)
            gaussian_map += weight * np.exp(exponent)
        levels = np.linspace(np.min(gaussian_map), np.max(gaussian_map), 15)
        ax.contour(X, Y, gaussian_map, levels=levels, colors='black', linewidths=1.0)
        ax.set_title("DP-GMM Overall Density Contours")
        plt.show()

###############################################################################
#  Agent and Simulation helper functions
###############################################################################
class Agent:
    def __init__(self, agent_type, sigma, pos, node_index):
        self.agent_type = agent_type    # e.g. 'aircraft' or 'truck'
        self.sigma = sigma              # sensing footprint std deviation
        self.node_index = node_index    # current assigned node index
        self.pos = np.array(pos, dtype=float)   # absolute position on the map
        self.local_offset = np.array([0.0, 0.0])  # offset from node center
        # Transition-related fields:
        self.transitioning = False      # whether the agent is currently transitioning
        self.start_pos = None           # starting position (when a transition is initiated)
        self.target_pos = None          # target position (the mean of the new component)
        self.target_node = None         # the new node index (assigned at transition)

def agent_fourier(sigma, size=50):
    """
    Compute the normalized Fourier magnitude of a Gaussian sensing footprint.
    (An agent with a Gaussian footprint of variance sigma^2.)
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    gauss = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    fft_mag = np.abs(fftshift(fft2(gauss)))
    return fft_mag / np.linalg.norm(fft_mag)

def initialize_agents(nodes, agent_counts, agent_sigmas):
    """
    Randomly initialize agents (of each type) over the map and assign each to its nearest node.
    """
    agents = []
    all_means = np.array([node['mean'] for node in nodes])
    map_min = np.min(all_means, axis=0)
    map_max = np.max(all_means, axis=0)
    for agent_type, count in agent_counts.items():
        sigma = agent_sigmas[agent_type]
        for i in range(count):
            pos = np.random.uniform(map_min, map_max)
            distances = [np.linalg.norm(pos - node['mean']) for node in nodes]
            node_index = np.argmin(distances)
            agent = Agent(agent_type, sigma, pos, node_index)
            agent.local_offset = pos - nodes[node_index]['mean']
            agents.append(agent)
    return agents

def compute_desired_allocation(nodes, agent_counts, agent_sigmas, dgmm):
    """
    For each node and each agent type, compute a spectral match score between the node’s
    Fourier signature and the agent’s sensing footprint. Then, normalize over nodes
    to obtain a desired allocation (i.e. number of agents of that type) per node.
    """
    node_scores = {}
    for i, node in enumerate(nodes):
        fft_node = dgmm.fourier_magnitude(node, size=50)
        node_scores[i] = {}
        for agent_type in agent_counts.keys():
            fft_agent = agent_fourier(agent_sigmas[agent_type], size=50)
            sim = np.dot(fft_node.ravel(), fft_agent.ravel())
            node_scores[i][agent_type] = sim
    desired_allocation = {i: {} for i in range(len(nodes))}
    for agent_type, count in agent_counts.items():
        total_score = sum(node_scores[i][agent_type] for i in range(len(nodes)))
        if total_score == 0:
            # Fall back to a uniform distribution if the total score is 0.
            for i in range(len(nodes)):
                desired_allocation[i][agent_type] = count / len(nodes)
        else:
            for i in range(len(nodes)):
                desired_allocation[i][agent_type] = (node_scores[i][agent_type] / total_score) * count
    return desired_allocation

def get_neighbors(tsp_order):
    """
    Given a TSP ordering (a closed cycle of node indices), return a dictionary
    mapping each node to its two neighbors.
    """
    neighbors = {}
    n = len(tsp_order)
    for i, node_index in enumerate(tsp_order):
        prev_node = tsp_order[(i - 1) % n]
        next_node = tsp_order[(i + 1) % n]
        neighbors[node_index] = [prev_node, next_node]
    return neighbors

def local_random_walk(agent, step_size=5.0):
    """
    Perform a discrete random walk for an agent.
    The probability of a given move is proportional to the Gaussian sensing footprint.
    (Step size increased to 5.0 for faster movement.)
    """
    moves = [np.array([0, 0]),
             np.array([1, 0]), np.array([-1, 0]),
             np.array([0, 1]), np.array([0, -1]),
             np.array([1, 1]), np.array([-1, 1]),
             np.array([1, -1]), np.array([-1, -1])]
    probs = np.array([np.exp(-np.linalg.norm(m)**2 / (2 * agent.sigma**2)) for m in moves])
    probs /= np.sum(probs)
    chosen_move = moves[np.random.choice(len(moves), p=probs)]
    return step_size * chosen_move

# --- Helper function for explicit overall-boundary checking ---
def check_bounds(pos, width, height):
    """
    Ensure that pos is within [0, width-1] x [0, height-1].
    """
    x, y = pos[0], pos[1]
    if x < 0:
        x = 0
    elif x > width - 1:
        x = width - 1
    if y < 0:
        y = 0
    elif y > height - 1:
        y = height - 1
    return np.array([x, y])

# --- Helper functions for component-specific boundaries ---
def get_component_bounds(node, width, height, k=3, min_std=1.0):
    """
    For a given Gaussian node, compute a bounding box.
    We use a k-sigma rule on the node's mean and standard deviations,
    with a minimum standard deviation (min_std) to avoid degenerate boxes.
    The resulting bounds are also clipped to the overall map bounds.
    """
    mu = node['mean']
    cov = node['covariance']
    std_x = max(np.sqrt(cov[0, 0]), min_std)
    std_y = max(np.sqrt(cov[1, 1]), min_std)
    x_min = mu[0] - k * std_x
    x_max = mu[0] + k * std_x
    y_min = mu[1] - k * std_y
    y_max = mu[1] + k * std_y
    # Clip to overall map bounds:
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, width - 1)
    y_max = min(y_max, height - 1)
    return (x_min, x_max, y_min, y_max)

def check_bounds_component(pos, bounds):
    """
    Ensure that pos is within the given component bounds (x_min, x_max, y_min, y_max).
    """
    x_min, x_max, y_min, y_max = bounds
    x, y = pos[0], pos[1]
    if x < x_min:
        x = x_min
    elif x > x_max:
        x = x_max
    if y < y_min:
        y = y_min
    elif y > y_max:
        y = y_max
    return np.array([x, y])

###############################################################################
#  Simulation Loop for Transparent GIF Generation
###############################################################################
def simulate_and_save_gif(info_map, nodes, agents, desired_allocation, tsp_order, dgmm, neighbors,
                          num_time_steps=200, output_file='simulation.gif'):
    """
    Run the simulation for a fixed number of time steps and save as a transparent GIF.
    Only the agents are drawn on a transparent background.
    """
    frames = []
    # Create a figure with a transparent background.
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='none')
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    
    height, width = info_map.shape
    transition_speed = 3.0  # Increased transition speed

    for t in range(num_time_steps):
        # --- Update transitioning agents ---
        transitioning_agents = [agent for agent in agents if agent.transitioning]
        if transitioning_agents:
            for agent in transitioning_agents:
                delta = agent.target_pos - agent.pos
                dist = np.linalg.norm(delta)
                if dist < 0.5 or transition_speed >= dist:
                    agent.pos = agent.target_pos.copy()
                    agent.node_index = agent.target_node
                    agent.local_offset = np.array([0.0, 0.0])
                    agent.transitioning = False
                else:
                    step = (delta / dist) * transition_speed
                    agent.pos = agent.pos + step
                    agent.pos = check_bounds(agent.pos, width, height)
            for agent in agents:
                if not agent.transitioning:
                    delta = local_random_walk(agent, step_size=3)
                    new_offset = agent.local_offset + delta
                    base = nodes[agent.node_index]['mean']
                    new_pos = base + new_offset
                    comp_bounds = get_component_bounds(nodes[agent.node_index], width, height)
                    new_pos = check_bounds_component(new_pos, comp_bounds)
                    agent.local_offset = new_pos - base
                    agent.pos = new_pos
        else:
            # --- No agents are transitioning; compute current allocation ---
            current_allocation = {i: {atype: 0 for atype in desired_allocation[i].keys()} for i in range(len(nodes))}
            for agent in agents:
                if not agent.transitioning:
                    current_allocation[agent.node_index][agent.agent_type] += 1

            # --- Initiate global transitions (batch) for nodes with surplus ---
            moves = []
            for i in range(len(nodes)):
                for agent_type in desired_allocation[i].keys():
                    curr = current_allocation[i][agent_type]
                    desired = desired_allocation[i][agent_type]
                    if curr > desired:
                        neighbor_defs = {}
                        total_deficit = 0
                        for nb in neighbors[i]:
                            neighbor_count = sum(1 for agent in agents if (not agent.transitioning) and agent.node_index == nb and agent.agent_type == agent_type)
                            deficit = max(0, desired_allocation[nb][agent_type] - neighbor_count)
                            neighbor_defs[nb] = deficit
                            total_deficit += deficit
                        if total_deficit > 0:
                            p_move = min(1, total_deficit / curr)
                            for agent in [a for a in agents if (not a.transitioning) and a.node_index == i and a.agent_type == agent_type]:
                                if np.random.rand() < p_move:
                                    nb_choice = np.random.choice(list(neighbor_defs.keys()),
                                                                 p=np.array(list(neighbor_defs.values())) / total_deficit)
                                    moves.append((agent, nb_choice))
            for agent, new_node in moves:
                agent.transitioning = True
                agent.start_pos = agent.pos.copy()
                agent.target_pos = np.array(nodes[new_node]['mean'])
                agent.target_node = new_node

            # --- For non-transitioning agents, update via local random walk within their component ---
            for agent in agents:
                if not agent.transitioning:
                    delta = local_random_walk(agent, step_size=3)
                    new_offset = agent.local_offset + delta
                    base = nodes[agent.node_index]['mean']
                    new_pos = base + new_offset
                    comp_bounds = get_component_bounds(nodes[agent.node_index], width, height)
                    new_pos = check_bounds_component(new_pos, comp_bounds)
                    agent.local_offset = new_pos - base
                    agent.pos = new_pos

        # --- Update Plot: Only agents are drawn on a transparent background ---
        ax.clear()
        ax.set_facecolor('none')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')
        # Draw agents (red for aircraft, blue for trucks).
        for agent in agents:
            color = 'red' if agent.agent_type == 'aircraft' else 'blue'
            ax.plot(agent.pos[0], agent.pos[1], marker='o', color=color, markersize=6)

        # Render the canvas and capture the frame with RGBA channels.
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # Get the ARGB buffer, then reshape and convert to RGBA.
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape((h, w, 4))
        frame = frame[:, :, [1, 2, 3, 0]]  # Convert ARGB to RGBA.
        frames.append(frame)

    plt.close(fig)
    imageio.mimsave(output_file, frames, duration=0.1)
    print(f"Saved transparent simulation GIF to {output_file}")

###############################################################################
#  New Function: Generate a Transparent GIF from an information map over 50 time steps
###############################################################################
def generate_simulation_gif(info_map, num_time_steps=50, output_file='simulation.gif'):
    """
    Given an information map (a 2D numpy array), this function:
      - Computes DP-GMM nodes,
      - Determines a TSP ordering and neighbor relationships,
      - Sets up default agent counts and sensing parameters,
      - Computes the desired allocation for each node,
      - Initializes agents on the map,
      - Runs the simulation for num_time_steps (default is 50),
      - And saves the resulting transparent GIF to the provided output_file.
      
    Returns the output file name.
    """
    # Compute DP-GMM nodes from the info_map.
    if isinstance(info_map, str):  # Only read if it's a filepath
        info_map = plt.imread(info_map)

    dgmm = DirichletGMM(max_components=10, weight_threshold=1e-3, plot_gaussians=False)
    nodes = dgmm.fit(info_map)

    # Compute TSP ordering and derive neighbor relationships (closed cycle).
    tsp_order = dgmm.tsp_path(nodes)
    neighbors = get_neighbors(tsp_order)

    # Set up default agent counts and sensing parameters.
    agent_counts = {
        'aircraft': 5,
        'truck': 10
    }
    agent_sigmas = {
        'aircraft': 10.0,
        'truck': 3.0
    }

    # Compute the desired allocation per node (via spectral matching).
    desired_allocation = compute_desired_allocation(nodes, agent_counts, agent_sigmas, dgmm)

    # Initialize agents (each assigned to its nearest node initially).
    agents = initialize_agents(nodes, agent_counts, agent_sigmas)

    # Run the simulation for a fixed number of steps and save as a transparent GIF.
    simulate_and_save_gif(info_map, nodes, agents, desired_allocation, tsp_order, dgmm, neighbors,
                            num_time_steps=num_time_steps, output_file=output_file)
    return output_file

###############################################################################
#  Main function (with command-line arguments)
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Agent Simulation on DP-GMM Nodes with Transparent GIF Output")
    parser.add_argument('--aircraft', type=int, default=5, help="Number of aircraft agents")
    parser.add_argument('--truck', type=int, default=10, help="Number of truck agents")
    parser.add_argument('--aircraft_sigma', type=float, default=10.0, help="Sensing sigma for aircraft")
    parser.add_argument('--truck_sigma', type=float, default=3.0, help="Sensing sigma for trucks")
    parser.add_argument('--map_path', type=str, default=None,
                        help="Path to an image file to use as the map. If not provided, a default Gaussian map is used.")
    parser.add_argument('--steps', type=int, default=200, help="Number of simulation time steps")
    parser.add_argument('--output', type=str, default='simulation.gif', help="Output GIF file name")
    args = parser.parse_args()

    # If a map path is provided and valid, load the image; otherwise, use the default generator.
    if args.map_path is not None and os.path.exists(args.map_path):
        info_map = plt.imread(args.map_path)
        # If the image is colored (has 3 or 4 channels), convert to grayscale.
        if info_map.ndim == 3:
            info_map = np.mean(info_map, axis=2)
        # Normalize the image (if needed)
        info_map = info_map.astype(np.float32)
        if info_map.max() > 1:
            info_map = info_map / 255.0
        print(f"Using map image from {args.map_path}", flush=True)
    else:
        print("No valid map image provided; using default generated map.", flush=True)
        return

    # To generate a transparent GIF using only 50 time steps via our new helper function:
    output_file = generate_simulation_gif(info_map, num_time_steps=50, output_file=args.output)
    print(f"Transparent simulation GIF generated: {output_file}")

if __name__ == "__main__":
    main()
