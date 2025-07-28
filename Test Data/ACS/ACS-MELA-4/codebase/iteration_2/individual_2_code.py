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
    # Enhanced chaotic map
    chaos = 4 * rg * (1 - rg) * (1 - 0.5*np.random.rand())
    
    # Dynamic opposition learning
    opp_prob = 0.3 + 0.4*(Best_score - Positions.mean())/Best_score
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_prob, opposite_pos, Positions)
    
    # Elite-guided adaptive search
    elite = Positions[np.argmin(Positions, axis=0)]
    w = chaos * np.exp(-np.arange(SearchAgents_no).reshape(-1,1)/SearchAgents_no
    new_pos = (1-w)*elite + w*Best_pos
    
    # Dimension-wise scaling
    scale = 1 - np.random.rand(dim) * chaos
    Positions = new_pos * scale
    #EVOLVE-END
    
    return Positions