# problem_evaluation_solution_v2.py
import os
from os import path
import numpy as np
from collections import deque
from typing import Callable, List, Any
import matplotlib.pyplot as plt


# =============================================================================
# 1. DATA CLASSES (MODIFIED FOR SOLVABILITY)
# =============================================================================

class Data_Network:
    """
    Holds problem-specific data for the network.
    *** MODIFIED FOR EASIER SOLVABILITY ***
    """

    def __init__(self, sn_pos_file: str):
        try:
            basepath = os.path.dirname(os.path.abspath(__file__))
            sn_data = np.load(path.join(basepath, sn_pos_file))
            self.sn_positions = sn_data
            print(f"Successfully loaded SN positions from '{sn_pos_file}'.")
        except FileNotFoundError:
            print(f"'{sn_pos_file}' not found. Generating easier random SN positions.")
            self.sn_positions = np.random.rand(100, 2) * 40 + 5

        self.num_cn = 50
        self.num_sn = 200
        self.capacity = 15
        self.cn_con_r = 20.0
        self.gama = 2.5
        self.sn_lowerest = -85.0
        self.beta_initial = 55.0


class Data_Algorithm:
    """Holds algorithm-specific parameters."""

    def __init__(self, data_net: Data_Network):
        self.SearchAgents = 50
        self.MaxIter = 100
        self.dim = data_net.num_cn * 3
        self.ub = np.tile([50, 50, 30], data_net.num_cn)
        self.lb = np.zeros(self.dim)
        self.S = 2.5  # Initial value for rg


# =============================================================================
# 2. HEURISTICS FUNCTION (MATCHING THE REQUESTED SIGNATURE)
# =============================================================================

def pso_inspired_heuristic(Positions: np.ndarray, Best_pos: np.ndarray, Best_score: float, rg: float) -> np.ndarray:
    """
    An example heuristic function that updates agent positions.
    This function strictly follows the signature: (Positions, Best_pos, Best_score, rg).

    It uses a simplified PSO-like attraction mechanism where 'rg' controls the
    step size towards the global best solution. A larger rg encourages exploration,
    while a smaller rg promotes exploitation.

    Args:
        Positions (np.ndarray): Current positions of all search agents.
        Best_pos (np.ndarray): The best position found so far in the entire population.
        Best_score (float): The fitness of the best position (not used in this simple version, but part of the interface).
        rg (float): A dynamic parameter, typically decreasing over iterations, used to control search behavior.

    Returns:
        np.ndarray: The new positions of the search agents.
    """
    # The 'Best_score' parameter is available but not used in this specific simple implementation.
    # It could be used in more complex heuristics (e.g., for acceptance criteria in Simulated Annealing).

    # Simple attraction towards the best known solution
    # The random factor ensures diversity in movement
    attraction_step = rg * np.random.rand(*Positions.shape) * (Best_pos - Positions)

    # Update positions
    new_positions = Positions + attraction_step

    return new_positions


# =============================================================================
# 3. MAIN EVALUATION CLASS (STRICTLY FOLLOWING THE FINAL REQUESTED STRUCTURE)
# =============================================================================

class NetworkEvaluation:
    """
    Manages the WSN optimization.
    """

    def __init__(self, sn_pos_file='sn_pos.npy', **kwargs):
        self.data = Data_Network(sn_pos_file)
        self.data_al = Data_Algorithm(self.data)

    def _is_connected_graph(self, adj_matrix: np.ndarray) -> bool:
        # (Unchanged from previous version)
        num_nodes = adj_matrix.shape[0]
        if num_nodes == 0: return True
        visited = np.zeros(num_nodes, dtype=bool)
        q = deque([0])
        visited[0] = True
        count = 1
        while q:
            node = q.popleft()
            neighbors = np.where(adj_matrix[node] > 0)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append(neighbor)
                    count += 1
        return count == num_nodes

    def f_obj(self, X: np.ndarray) -> float:
        # (Unchanged from previous version)
        state_sn = np.reshape(X, (self.data.num_cn, 3))
        pos_cn_x, pos_cn_y, pos_cn_p = state_sn[:, 0], state_sn[:, 1], state_sn[:, 2]

        cn_pos_matrix = np.stack([pos_cn_x, pos_cn_y], axis=1)
        dist_matrix = np.sqrt(
            np.sum((self.data.sn_positions[:, np.newaxis, :] - cn_pos_matrix[np.newaxis, :, :]) ** 2, axis=-1))

        beta_x = np.where((self.data.sn_positions[:, 0:1] - 25) * (cn_pos_matrix[:, 0] - 25) < 0, 8.0, 0.0)
        beta_y = np.where((self.data.sn_positions[:, 1:2] - 25) * (cn_pos_matrix[:, 1] - 25) < 0, 20.0, 0.0)
        beta_matrix = beta_x + beta_y

        path_loss_matrix = self.data.beta_initial + 10 * self.data.gama * np.log10(dist_matrix + 1e-9) + beta_matrix

        cn_capacity = np.full(self.data.num_cn, self.data.capacity, dtype=int)
        is_sn_covered = np.zeros(self.data.num_sn, dtype=bool)
        sorted_cn_indices = np.argsort(path_loss_matrix, axis=1)

        for sn_idx in range(self.data.num_sn):
            for cn_choice_rank in range(self.data.num_cn):
                best_cn_idx = sorted_cn_indices[sn_idx, cn_choice_rank]
                if cn_capacity[best_cn_idx] > 0:
                    loss = path_loss_matrix[sn_idx, best_cn_idx]
                    if pos_cn_p[best_cn_idx] - loss >= self.data.sn_lowerest:
                        is_sn_covered[sn_idx] = True
                        cn_capacity[best_cn_idx] -= 1
                        break

        uncovered_sn = self.data.num_sn - np.sum(is_sn_covered)

        cn_dist_matrix = np.sqrt(
            np.sum((cn_pos_matrix[:, np.newaxis, :] - cn_pos_matrix[np.newaxis, :, :]) ** 2, axis=-1))
        link_cn_cn = np.where(cn_dist_matrix <= self.data.cn_con_r, 1, 0)
        np.fill_diagonal(link_cn_cn, 0)

        is_connected = self._is_connected_graph(link_cn_cn)
        power_std_over = max(np.std(pos_cn_p) - 1.0, 0)

        penalty_coverage = uncovered_sn * 10
        penalty_connectivity = (0 if is_connected else 1) * 1000
        penalty_power_std = power_std_over * 100
        true_fitness = np.sum(10 ** (pos_cn_p / 10))

        fitness = true_fitness + penalty_coverage + penalty_connectivity + penalty_power_std
        return fitness

    def evaluate(self, priority: Callable) -> List[float]:
        """
        Main optimization process. The call to the priority function
        now strictly matches the requested signature.
        """
        SearchAgents_no = self.data_al.SearchAgents
        Max_iter = self.data_al.MaxIter
        dim = self.data_al.dim
        lb = self.data_al.lb
        ub = self.data_al.ub

        Positions = lb + np.random.rand(SearchAgents_no, dim) * (ub - lb)

        # --- MODIFICATION ---: Initialize rg and its decay as per the original structure
        rg = self.data_al.S
        decay_rate = self.data_al.S / Max_iter

        Best_pos = np.zeros(dim)
        Best_score = np.inf
        Convergence_curve = []

        print(f"Starting optimization for {Max_iter} iterations...")
        for t in range(Max_iter):
            for i in range(Positions.shape[0]):
                # Boundary checking
                Positions[i] = np.clip(Positions[i], lb, ub)

                # Fitness evaluation
                fitness = self.f_obj(Positions[i, :])

                # Update global best
                if fitness < Best_score:
                    Best_score = fitness
                    Best_pos = Positions[i, :].copy()

            if (t + 1) % 10 == 0:
                is_feasible = "Yes" if Best_score < 1000 else "No"
                print(f"Iter {t + 1}/{Max_iter}, Best Score: {Best_score:.2f}, rg: {rg:.2f}, Feasible: {is_feasible}")

            Convergence_curve.append(Best_score)

            # --- MODIFICATION ---: Call the priority function with the new, correct signature
            Positions = priority(Positions, Best_pos, Best_score, rg)

            # --- MODIFICATION ---: Update rg for the next iteration
            rg = max(0.1, rg - decay_rate)

        print("Optimization finished.")
        return Convergence_curve


# =============================================================================
# 4. PLOTTING AND EXECUTION
# =============================================================================

def plot_convergence(curve):
    """绘制适应度收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(curve, linewidth=2)
    plt.title('Fitness Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Score')
    plt.axhline(y=1000, color='r', linestyle='--', label='Feasibility Threshold (1000)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


if __name__ == '__main__':
    if not os.path.exists('sn_pos.npy'):
        print("Error: 'sn_pos.npy' not found. Please run 'generate_sn_pos.py' first.")
    else:
        evaluator = NetworkEvaluation(sn_pos_file='sn_pos.npy')
        convergence_curve = evaluator.evaluate(priority=pso_inspired_heuristic)

        final_score = convergence_curve[-1]
        print(f"\nFinal Best Score: {final_score:.4f}")
        if final_score < 1000:
            print("A feasible solution was found!")
            print(f"The minimum total power consumption found is {final_score:.2f} mW.")
        else:
            print("Failed to find a feasible solution within the given iterations.")

        # plot_convergence(convergence_curve)