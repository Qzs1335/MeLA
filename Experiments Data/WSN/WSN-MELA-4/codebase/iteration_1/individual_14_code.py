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
    progress = 1 - rg  # Normalized [0,1] where 1 is start
    w = 0.9 * progress + 0.1  # Decreasing inertia
    
    # Opposition-based learning for diversity
    if np.random.rand() < 0.3*progress:
        opposition_pos = lb_array + ub_array - Positions
        Positions = np.where(np.random.rand(*Positions.shape) < 0.5, opposition_pos, Positions)
    
    # Fitness-guided mutation
    mutation_strength = 0.1 * (1 - progress)
    fitness_weights = np.exp(-np.linspace(0, 1, SearchAgents_no))  # Rank-based
    mutation = mutation_strength * np.random.randn(*Positions.shape) * fitness_weights.reshape(-1,1)
    Positions = np.clip(Positions + mutation, lb_array, ub_array)
    #EVOLVE-END       

    return Positions