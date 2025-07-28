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
    # Improved chaotic map with logistic mapping
    chaos = 4.0 * rg * (1.0 - rg)
    chaos = np.clip(chaos, 1e-8, 0.9999)
    
    # Enhanced opposition learning with dynamic probability
    opp_prob = 0.3 + 0.4 * (1 - Best_score/np.max([Best_score, 1e-8]))
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, opposite_pos, Positions)
    
    # Adaptive cosine search with elite guidance
    t = np.pi * (0.1 + 0.9*np.random.rand(SearchAgents_no, 1))
    w = (0.5 * (1 + np.cos(t)) * chaos).reshape(-1,1)
    elite_guide = Best_pos + chaos*(Best_pos - Positions.mean(axis=0))
    new_pos = elite_guide*(1-w) + Positions*w
    
    # Dynamic scaling with adaptive factor
    scale = 0.5 + 0.5*chaos*np.random.rand(SearchAgents_no, dim)
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.7, new_pos*scale, new_pos)
    #EVOLVE-END
    
    return Positions