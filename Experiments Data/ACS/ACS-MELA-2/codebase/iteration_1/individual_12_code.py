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
    # Chaotic exploration
    chaotic_map = 4.0 * Positions * (1 - Positions)
    
    # Adaptive gradient following
    adaptive_scale = 0.9 - 0.5 * (Best_score / (Best_score + 1))
    gradient = np.abs(Best_pos - Positions)
    
    # Neighborhood interaction
    neighbors = np.random.choice(SearchAgents_no, size=(SearchAgents_no,3), replace=True)
    mentor = Positions[neighbors].mean(axis=1)
    
    # Hybrid update
    r1, r2 = np.random.rand(2)
    Positions = r1*chaotic_map + (1-r1)*(adaptive_scale*gradient) + r2*mentor
    #EVOLVE-END       

    return Positions