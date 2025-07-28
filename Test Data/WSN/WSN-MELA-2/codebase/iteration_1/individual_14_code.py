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
    # Dynamic weights based on iteration progress
    w = 0.9 - (0.5 * rg)  
    c1 = 1.5 * rg
    c2 = 2.0 - rg
    
    # Hybrid velocity update
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = c1 * r1 * (Best_pos - Positions)
    social = c2 * r2 * (Best_pos[np.random.randint(0, SearchAgents_no)] - Positions)
    
    # Adaptive perturbation
    perturbation = 0.1 * rg * np.random.randn(SearchAgents_no, dim)
    
    Positions = w * Positions + cognitive + social + perturbation
    #EVOLVE-END       
    
    return Positions