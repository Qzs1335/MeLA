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
    # Enhanced adaptive parameters
    progress = 1 / (1 + np.exp(-Best_score/200 + 3))  # Sigmoid progress
    w = 0.9 - 0.5 * progress  # Smooth inertia decay
    c1 = 2.5 * (1 - progress)  # Non-linear cognitive
    c2 = 1.5 + progress  # Balanced social
    
    # Hybrid velocity update
    rp = np.random.permutation(SearchAgents_no)
    velocity = w * np.random.randn(*Positions.shape) + \
              c1 * np.random.rand(*Positions.shape) * (Best_pos - Positions) + \
              c2 * np.random.rand(*Positions.shape) * (Positions[rp] - Positions)
    
    # Adaptive position update
    Positions = Positions + velocity * (rg * (0.5 + progress/2))
    boundary_violation = (Positions < lb_array) | (Positions > ub_array)
    reflection = np.where(Positions > ub_array, 2*ub_array - Positions, 
                       2*lb_array - np.where(Positions < lb_array, Positions, lb_array))
    Positions = np.where(boundary_violation, 
                       reflection * (0.8 + 0.2*np.random.rand(*Positions.shape)),
                       Positions)
    #EVOLVE-END       
    
    return Positions