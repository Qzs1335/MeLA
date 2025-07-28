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
    w = 0.9 * progress + 0.1
    c1 = 1.5 * (1 - progress)
    c2 = 1.5 * progress
    
    # Lévy flights for exploration
    levy = np.random.normal(0, 1, Positions.shape) * (rg/(1+Best_score))**0.5
    cognitive = c1 * np.random.rand() * (Best_pos - Positions)
    social = c2 * np.random.rand() * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Dynamic boundary handling
    velocity = w * levy + cognitive + social
    Positions = Positions + velocity
    overflow = Positions > ub_array
    Positions = np.where(overflow, ub_array - (Positions - ub_array)*0.5, Positions)
    underflow = Positions < lb_array
    Positions = np.where(underflow, lb_array + (lb_array - Positions)*0.5, Positions)
    #EVOLVE-END       
    
    return Positions