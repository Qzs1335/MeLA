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
    momentum = 0.8 * (Positions - Best_pos[np.newaxis, :])
    guided_noise = 0.2 * (Best_score - np.linalg.norm(Positions, axis=1)[:, np.newaxis]) * np.random.randn(*Positions.shape)
    Positions = Positions + momentum + guided_noise
    
    scale = 0.5 + 0.4 * np.sin(rg * np.pi/2)
    Positions = Best_pos + scale * (Positions - Best_pos)
    #EVOLVE-END       
    return Positions