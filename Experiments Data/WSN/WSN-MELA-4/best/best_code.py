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
    # Fitness-based adaptive parameters
    progress = 1 - (Best_score / 1000)  # Normalized progress
    w = 0.4 + 0.4 * progress  # Decreasing inertia
    c1 = 2.0 * progress  # Decreasing cognitive
    c2 = 2.0 - c1  # Increasing social
    
    # Velocity update with boundary reflection
    velocity = w * np.random.randn(*Positions.shape) + \
               c1 * np.random.rand() * (Best_pos - Positions) + \
               c2 * np.random.rand() * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Position update with reflection boundary handling
    Positions = Positions + velocity * rg
    overflow = Positions > ub_array
    Positions = np.where(overflow, 2*ub_array - Positions, Positions)
    underflow = Positions < lb_array
    Positions = np.where(underflow, 2*lb_array - Positions, Positions)
    #EVOLVE-END       
    
    return Positions