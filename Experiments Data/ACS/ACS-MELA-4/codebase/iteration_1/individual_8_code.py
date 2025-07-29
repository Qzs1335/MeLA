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
    # Adaptive parameter control
    t = 1 - (1/(1+np.exp(-5*(30-rg)/30)))  # Sigmoid decay
    
    # Chaotic exploration
    chaotic_map = 3.7 * Positions * (1-Positions) * (rg < 15)
    
    # Opposition-based learning
    opposite_pos = 1 - Positions
    mask = np.random.rand(*Positions.shape) < t
    Positions = np.where(mask, opposite_pos, Positions) + chaotic_map
    
    # Elite guidance
    elite_mask = np.random.rand(*Positions.shape) < 0.5*t
    Positions = np.where(elite_mask, 0.5*(Positions + Best_pos), Positions)
    #EVOLVE-END       

    return Positions