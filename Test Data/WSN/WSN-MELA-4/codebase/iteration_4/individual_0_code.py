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
    # Adaptive parameters with sigmoid scaling
    progress = 1 / (1 + np.exp(0.1*(Best_score-800)))
    w = 0.9 - 0.5*progress
    c1 = 2.5*(1-progress)
    c2 = 2.0 + 0.5*progress
    
    # Hybrid velocity update
    cos_factor = np.cos(2*np.pi*np.random.rand(*Positions.shape))
    velocity = w*np.random.randn(*Positions.shape) + \
              c1*np.random.rand()*(Best_pos - Positions)*cos_factor + \
              c2*np.random.rand()*(Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Soft boundary reflection
    Positions = Positions + velocity*rg
    overflow = Positions > ub_array
    underflow = Positions < lb_array
    Positions = np.where(overflow, ub_array - 0.5*(Positions-ub_array), 
                        np.where(underflow, lb_array + 0.5*(lb_array-Positions), Positions))
    #EVOLVE-END       
    
    return Positions