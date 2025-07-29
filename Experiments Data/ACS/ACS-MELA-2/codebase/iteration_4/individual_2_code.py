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
    beta = 1.5
    sigma = 0.6966  # Precomputed for beta=1.5
    scale = rg * 0.01
    levy_step = scale * sigma * np.random.randn(SearchAgents_no,1) / (np.abs(np.random.randn(SearchAgents_no,1))**beta)
    
    norm_dist = np.linalg.norm(Positions-Best_pos, axis=1, keepdims=True)
    learn_prob = 0.5 + 0.4*(Best_score - norm_dist)/Best_score
    mask = np.random.rand(*Positions.shape) < learn_prob
    scaling = 1 + (Best_score/(norm_dist+1e-6) - 0.5)*0.1
    Positions = np.where(mask, Best_pos + levy_step*(Positions-Best_pos), Positions*scaling)
    #EVOLVE-END       
    return Positions