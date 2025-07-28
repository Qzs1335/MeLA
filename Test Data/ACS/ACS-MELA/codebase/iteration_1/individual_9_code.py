import numpy as np
import numpy as np 
def heuristics_v2(data_al, data_pb, Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)

    #EVOLVE-START
    p = 360 * np.random.rand()
    dynamic_r = rg * (0.5 + np.random.rand()/2)
    beta = 0.1 + 0.4 * np.random.rand()

    cos_p = np.cos(np.deg2rad(p))**2
    perturb = dynamic_r * np.random.normal(0, 1, Positions.shape)
            
    mask = np.random.rand(*Positions.shape) < 0.7
    Positions = np.where(mask, 
                        Best_pos + beta * (Best_pos - Positions) + cos_p * perturb,
                        Positions * (1.0 - 0.5*np.random.rand()))
    #EVOLVE-END 
    
    return Positions