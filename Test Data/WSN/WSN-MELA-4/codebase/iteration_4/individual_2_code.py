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
    # Adaptive parameters with cosine modulation
    progress = np.cos(0.5*np.pi*(1 - Best_score/1000))
    w = 0.3 + 0.5*progress
    c1 = 1.5 + np.cos(np.pi*progress)
    c2 = 2.5 - c1
    
    # Hybrid velocity update
    r1 = np.random.rand(SearchAgents_no, 1)
    r2 = np.random.rand(SearchAgents_no, 1)
    cognitive = c1*r1*(Best_pos - Positions)
    social = c2*r2*(Positions[np.random.permutation(SearchAgents_no)] - Positions)
    velocity = w*(0.5*np.random.randn(*Positions.shape) + cognitive + social)
    
    # Enhanced position update
    new_pos = Positions + velocity*rg*(0.5 + 0.5*progress)
    reflect_mask = (new_pos < lb_array) | (new_pos > ub_array)
    Positions = np.where(reflect_mask, Positions + 0.5*(Best_pos - Positions)*np.random.rand(), new_pos)
    #EVOLVE-END       
    
    return Positions