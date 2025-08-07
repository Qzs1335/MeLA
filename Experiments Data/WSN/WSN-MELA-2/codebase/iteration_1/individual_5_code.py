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
    adaptive_factor = 0.5 * (1 + np.cos(np.pi * rg))
    
    # Opposition-based learning
    opposite_pos = 1 - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.3, 
                        opposite_pos, Positions)
    
    # Chaotic perturbation using logistic map
    chaos = 3.9 * Positions * (1 - Positions)
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < adaptive_factor,
                        Best_pos + rg * (chaos - 0.5),
                        Positions)
    #EVOLVE-END       
    
    return Positions