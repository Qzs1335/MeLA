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
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    w = 0.9 - 0.5*progress
    c1 = 1.5*(1-progress)
    c2 = 1.5 + 0.5*progress
    
    # Hybrid velocity update
    r1, r2 = np.random.rand(2)
    elite_mask = np.random.rand(SearchAgents_no, dim) < 0.2
    velocity = (w * np.random.randn(*Positions.shape) + 
               c1*r1*(Best_pos - Positions) + 
               c2*r2*(Positions[np.random.permutation(SearchAgents_no)] - Positions))
    
    # Smart boundary handling
    new_pos = Positions + velocity*rg
    reflect_prob = np.random.rand(*Positions.shape)
    Positions = np.where((new_pos < lb_array) | (new_pos > ub_array),
                       np.where(reflect_prob < 0.7, 
                               np.clip(new_pos, lb_array, ub_array),
                               np.where(new_pos < lb_array, 
                                       2*lb_array - new_pos, 
                                       2*ub_array - new_pos)),
                       new_pos)
    #EVOLVE-END       
    return Positions