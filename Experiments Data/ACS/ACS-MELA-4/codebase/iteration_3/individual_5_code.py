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
    # Enhanced logistic chaotic map with adaptive control
    chaos = 3.9 * rg * (1 - rg) * (1 - 0.5*np.random.rand())
    
    # Adaptive opposition learning with elite influence
    opp_prob = 0.6 - 0.4*(Best_score/10000) 
    elite_opp = Best_pos + (lb_array + ub_array - 2*Best_pos)*chaos
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_prob, 
                        np.where(np.random.rand(SearchAgents_no,1) < 0.5, elite_opp, lb_array + ub_array - Positions), 
                        Positions)
    
    # Dimension-aware adaptive search
    w = np.exp(-chaos*np.linspace(0,1,SearchAgents_no)).reshape(-1,1)
    scale = 0.8 + 0.4*chaos*np.random.rand(SearchAgents_no, dim)
    Positions = (w*Best_pos + (1-w)*Positions) * scale
    #EVOLVE-END
    
    return Positions