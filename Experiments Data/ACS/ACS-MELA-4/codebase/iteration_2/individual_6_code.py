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
    # Enhanced chaotic map (logistic)
    chaos = 4 * rg * (1 - rg) * (3.9 * rg * (1 - rg))
    
    # Dynamic opposition learning
    opp_prob = 0.3 + 0.4 * (1 - np.exp(-0.01 * Best_score))
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_prob, opposite_pos, Positions)
    
    # Elite-guided adaptive search
    elite = Positions[np.argsort(Best_score)[:max(1, SearchAgents_no//5)]]
    w = chaos * np.random.rand(SearchAgents_no, 1)
    new_pos = Best_pos + w * (elite.mean(0) - Positions)
    
    # Chaotic scaling
    scale = 0.5 + chaos * np.random.rand(SearchAgents_no, dim)
    Positions = np.where(chaos > 0.5, new_pos * scale, new_pos)
    #EVOLVE-END
    
    return Positions