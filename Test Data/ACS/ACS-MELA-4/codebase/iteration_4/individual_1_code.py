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
    chaos = 4 * decay * (1 - decay) * (0.5 + 0.5*np.random.rand())
    
    # Probabilistic opposition learning
    opp_prob = 0.4 * np.exp(-Best_score/15000)
    mask = np.random.rand(*Positions.shape) < opp_prob
    Positions = np.where(mask, (lb_array + ub_array - Positions)*0.7 + Positions*0.3, Positions)
    
    # Elite guidance with dynamic weights
    w = np.exp(-0.5*chaos*np.linspace(0,1,SearchAgents_no)).reshape(-1,1)
    elite = Best_pos * (1 + chaos*np.random.randn(*Positions.shape))
    Positions = elite*w + Positions*(1-w)
    #EVOLVE-END
    
    return Positions