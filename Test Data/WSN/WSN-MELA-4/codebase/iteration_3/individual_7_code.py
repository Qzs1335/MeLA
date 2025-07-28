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
    # Hybrid adaptive parameters
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    w = 0.9 - 0.5*progress
    c1 = 1.5*(1-progress)
    c2 = 1.5*progress
    
    # Hybrid PSO-cosine velocity
    r1, r2 = np.random.rand(2)
    cos_term = np.cos(np.pi*rg)*progress
    velocity = w*(np.random.randn(*Positions.shape) + cos_term) + \
              c1*r1*(Best_pos - Positions) + \
              c2*r2*(Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Adaptive boundary reflection
    Positions = Positions + velocity*rg*(0.5 + progress/2)
    reflect_scale = 1 + progress
    Positions = np.where(Positions > ub_array, ub_array - reflect_scale*(Positions - ub_array), Positions)
    Positions = np.where(Positions < lb_array, lb_array + reflect_scale*(lb_array - Positions), Positions)
    #EVOLVE-END       
    
    return Positions