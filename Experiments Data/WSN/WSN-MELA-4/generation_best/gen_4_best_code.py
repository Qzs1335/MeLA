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
    norm_score = np.clip(Best_score/1000, 0.1, 0.9)
    w = 0.9 - 0.5*norm_score
    c1 = 2.5*(1-norm_score)
    c2 = 2.5 - c1
    
    # Hybrid velocity update
    r1 = np.random.rand(SearchAgents_no, 1)
    r2 = np.random.rand(SearchAgents_no, 1)
    cognitive = c1*r1*(Best_pos - Positions)
    social = c2*r2*(Positions[np.random.permutation(SearchAgents_no)] - Positions)
    velocity = w*np.random.randn(*Positions.shape) + cognitive + social
    
    # Smart boundary handling
    Positions = np.clip(Positions + velocity*rg, lb_array-0.1, ub_array+0.1)
    overflow = Positions > ub_array
    underflow = Positions < lb_array
    Positions = np.where(overflow|underflow, 
                        Best_pos + 0.1*np.random.randn(*Positions.shape), 
                        Positions)
    #EVOLVE-END       
    return Positions