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
    # Logistic chaotic mapping with adaptive control
    chaos = 4.0 * rg * (1.0 - rg) * (1.0 - Best_score/25000)
    
    # Dynamic opposition learning
    opp_prob = 0.7 * np.exp(-Best_score/15000)
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(*Positions.shape) < opp_prob
    Positions = np.where(mask, 0.6*opposite_pos + 0.4*Positions, Positions)
    
    # Elite guidance with adaptive exponential weights
    elite = Best_pos * (1 + chaos*np.random.randn(*Positions.shape))
    w = np.exp(-0.8*chaos*np.linspace(0,1,SearchAgents_no)).reshape(-1,1)
    Positions = w*elite + (1-w)*Positions
    #EVOLVE-END
    
    return Positions