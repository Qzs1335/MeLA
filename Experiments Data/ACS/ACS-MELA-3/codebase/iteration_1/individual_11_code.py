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
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(*Positions.shape) < 0.5, opposite_pos, Positions)
    
    # Adaptive parameter control
    rg = rg * (0.9 + 0.1 * np.random.rand())
    
    # Differential evolution mutation
    a, b, c = np.random.choice(SearchAgents_no, 3, replace=False)
    mutant = Positions[a] + 0.5 * (Positions[b] - Positions[c])
    
    # Fitness-distance balance
    dist = np.linalg.norm(Positions - Best_pos, axis=1).reshape(-1,1)
    fdb = 0.5 + Best_score / (Best_score + dist + 1e-10)
    Positions = fdb * Positions + (1-fdb) * mutant
    
    # Boundary check
    Positions = np.clip(Positions, lb_array, ub_array)
    #EVOLVE-END   
    
    return Positions