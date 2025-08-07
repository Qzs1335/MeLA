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
    # Adaptive chaotic control
    decay = np.clip(1 - (Best_score/25000), 0.1, 0.9)
    chaos = 3.8 * decay * (1 - decay) * (1 + 0.2*np.random.randn())
    
    # Dynamic opposition learning
    opp_prob = 0.5 * np.exp(-Best_score/15000)
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(*Positions.shape) < opp_prob
    Positions = np.where(mask, (opposite_pos + 0.8*Positions), Positions)
    
    # Elite guidance with adaptive exponential weights
    elite = Best_pos * (1 + chaos*np.random.rand(*Positions.shape))
    w = np.exp(-0.4*chaos*np.linspace(0,1,SearchAgents_no)).reshape(-1,1)
    Positions = elite*w + Positions*(1-w)
    #EVOLVE-END
    
    return Positions