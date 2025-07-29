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
    # Dynamic chaotic control
    decay = 0.9 * (1 - Best_score/25000)
    chaos = 3.8 * rg * decay * (1 - rg * decay)
    
    # Adaptive opposition with elite bias
    opp_prob = 0.7 * np.exp(-Best_score/1500)
    elite_opp = (Best_pos + lb_array + ub_array - Positions)/2
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, elite_opp, Positions)
    
    # Nonlinear elite guidance
    w = np.exp(-0.7*chaos*np.arange(1, SearchAgents_no+1)**0.5/SearchAgents_no).reshape(-1,1)
    Positions = Best_pos*w + Positions*(1-w) + chaos*np.random.randn(SearchAgents_no, dim)
    #EVOLVE-END
    
    return Positions