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
    progress = 1 / (1 + np.exp(-Best_score/200))  # Sigmoid scaling
    w = 0.9 - 0.5*progress
    c1 = 2.5*(1-progress)
    c2 = 1.5 + progress
    
    # Hybrid velocity update
    r_perm = np.random.permutation(SearchAgents_no)
    velocity = w*np.random.randn(*Positions.shape) + \
              c1*np.random.rand(*Positions.shape)*(Best_pos-Positions) + \
              c2*np.random.rand(*Positions.shape)*(Positions[r_perm]-Positions)
    
    # Clamped position update
    Positions = np.clip(Positions + velocity*rg*(0.5+progress), lb_array, ub_array)
    reflect_prob = np.random.rand(*Positions.shape)
    Positions = np.where(reflect_prob<0.1, lb_array + ub_array - Positions, Positions)
    #EVOLVE-END       
    
    return Positions