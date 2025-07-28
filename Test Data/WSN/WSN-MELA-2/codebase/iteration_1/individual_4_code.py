import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)

    #EVOLVE-START
    # Adaptive exploration-exploitation balance
    progress = 1 - rg  # Normalized progress
    w = 0.9 - 0.5 * progress  # Decreasing inertia weight
    
    # Memory-based perturbation
    memory_effect = 0.3 * np.random.randn(*Positions.shape) * (Best_pos - Positions)
    
    # Neighborhood search
    radius = 0.1 * progress
    neighborhood = Positions + radius * np.random.randn(*Positions.shape)
    
    # Fitness-based mutation
    mutation_prob = 0.1 * (1 - progress)
    mutation_mask = np.random.rand(*Positions.shape) < mutation_prob
    mutation = mutation_mask * np.random.randn(*Positions.shape)
    
    Positions = w * Positions + memory_effect + mutation
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.7, 
                        neighborhood, Positions)
    #EVOLVE-END       
    return Positions