import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    #EVOLVE-START
    # Adaptive chaotic control
    chaos = rg * (2 - Best_score/10000) * np.random.rand(SearchAgents_no, 1)
    
    # Logistic opposition learning
    opp_prob = 1/(1 + np.exp(0.002*Best_score - 5))
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(*Positions.shape) < opp_prob, 
                        (opposite_pos + chaos*Positions)/(1+chaos), 
                        Positions)
    
    # Dynamic elite guidance
    w = np.linspace(0.9, 0.1, SearchAgents_no).reshape(-1,1) * (1 - chaos)
    Positions = w*Best_pos + (1-w)*Positions
    
    # Boundary handling
    Positions = np.clip(Positions, lb_array, ub_array)
    #EVOLVE-END
    
    return Positions