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
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    cos_weight = 0.5*(1 + np.cos(np.pi*progress))
    
    # Hybrid velocity update
    velocity = (cos_weight * np.random.randn(*Positions.shape) + 
               (1-progress) * np.random.rand() * (Best_pos - Positions) +
               progress * np.random.rand() * (Positions[np.random.permutation(SearchAgents_no)] - Positions))
    
    # Adaptive boundary reflection
    Positions += velocity * rg
    reflect_coeff = 1 + progress * np.random.rand()
    Positions = np.where(Positions > ub_array, ub_array - reflect_coeff*(Positions-ub_array), Positions)
    Positions = np.where(Positions < lb_array, lb_array + reflect_coeff*(lb_array-Positions), Positions)
    #EVOLVE-END       
    
    return Positions