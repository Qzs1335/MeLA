import numpy as np
import numpy as np

def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]
    
    # Initialize bounds
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    # Boundary handling
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    # Calculate fitness-based weights
    fitness_weights = 1 / (1 + np.abs(Best_score - np.random.rand(SearchAgents_no, 1) * Best_score))
    
    # Select random donors
    donor_indices = np.random.choice(SearchAgents_no, (SearchAgents_no, 3), replace=True)
    donors = Positions[donor_indices]
    
    # Mutation operation
    mutation = (donors[:, 0] - donors[:, 1] + Best_pos - Positions) * fitness_weights
    
    # Adaptive exploration control
    adjusted_rg = rg * (0.5 + np.random.exponential(0.5, (SearchAgents_no, 1)))
    exploration_factor = 1 / (1 + np.exp(-adjusted_rg))
    
    # Position update
    Positions = np.where(
        np.random.rand(*Positions.shape) < exploration_factor,
        Positions + adjusted_rg * mutation,
        Best_pos * (1 - exploration_factor) + Positions * exploration_factor
    )
    #EVOLVE-END
    
    return Positions