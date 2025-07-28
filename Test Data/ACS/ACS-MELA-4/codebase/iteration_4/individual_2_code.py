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
    decay = np.clip(1 - (Best_score/18000), 0.1, 0.9)
    chaos = 3.9 * decay * (1 - decay) * (1 + np.sin(rg))
    
    # Adaptive opposition with elite bias
    opp_prob = 0.7 * np.exp(-0.015*Best_score/800)
    opposite_pos = 0.7*(lb_array + ub_array - Positions) + 0.3*Best_pos
    Positions = np.where(np.random.rand(SearchAgents_no, dim)<opp_prob, 
                        (opposite_pos + chaos*Positions)/(1+chaos), Positions)
    
    # Nonlinear elite guidance
    w = np.exp(-0.7*chaos*np.linspace(0,1,SearchAgents_no)**2).reshape(-1,1)
    Positions = w*Best_pos*(1+0.3*chaos*np.random.randn(SearchAgents_no,dim)) + (1-w)*Positions
    #EVOLVE-END
    
    return Positions