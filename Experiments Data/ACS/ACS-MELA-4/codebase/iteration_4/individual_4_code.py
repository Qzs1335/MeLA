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
    # Smooth chaotic control
    decay = 0.5 * (1 + np.cos(Best_score/40000*np.pi))
    chaos = 4 * rg * decay * (1 - rg * decay)
    
    # Dynamic opposition learning
    opp_prob = 1 / (1 + np.exp(0.002*(Best_score-10000)))
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, (opposite_pos + chaos*Positions)/(1+chaos), Positions)
    
    # Dimension-adaptive elite guidance
    elite = Best_pos * (1 + chaos*np.random.randn(SearchAgents_no, dim))
    w = np.exp(-0.5*chaos*np.linspace(0,1,SearchAgents_no)).reshape(-1,1) * np.random.rand(1,dim)
    Positions = elite*w + Positions*(1-w)
    #EVOLVE-END
    
    return Positions