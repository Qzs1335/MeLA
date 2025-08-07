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
    # Hybrid chaotic map
    chaos = 4*rg*(1-rg)*(1-np.random.rand()) if np.random.rand() > 0.5 else 1-2*np.abs(rg-0.5)
    
    # Adaptive elite opposition
    opp_prob = 0.7 - 0.3*(Best_score/10000)
    elite_opp = Best_pos + (lb_array + ub_array - 2*Best_pos)*np.random.rand(SearchAgents_no, dim)
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_prob, elite_opp, Positions)
    
    # Dimension-wise mutation
    mut_prob = 0.1 + 0.4*chaos*np.random.rand(dim)
    w = np.exp(-chaos*np.arange(1, SearchAgents_no+1)/SearchAgents_no).reshape(-1,1)*(1-mut_prob)
    
    # Enhanced position update
    scale = 0.8 + 0.4*chaos*np.random.rand(SearchAgents_no, dim)
    Positions = (Best_pos*w + Positions*(1-w)) * scale
    #EVOLVE-END
    
    return Positions