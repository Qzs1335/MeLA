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
    w = 0.9 - (0.9-0.4)*rg
    neighbor_size = max(3, int(SearchAgents_no*(0.3 - 0.2*rg)))
    dists = np.linalg.norm(Positions - Best_pos, axis=1)
    neighbors = np.argsort(dists)[:neighbor_size]
    
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = (1.2 + 0.6*rg) * r1 * (Best_pos - Positions)
    social = (1.2 + 0.6*rg) * r2 * (Positions[neighbors].mean(axis=0) - Positions)
    
    Positions = w*Positions + cognitive + social
    mutate_mask = np.random.rand(*Positions.shape) < 0.1*rg
    Positions = np.where(mutate_mask, Positions + rg*(np.random.rand(*Positions.shape)-0.5), Positions)
    #EVOLVE-END       

    return Positions