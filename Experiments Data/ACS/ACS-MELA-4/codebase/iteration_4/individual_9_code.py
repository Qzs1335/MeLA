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
    # Simplified chaotic control
    chaos = 4 * rg * (1 - rg)
    
    # Dynamic opposition learning
    opp_prob = 1/(1+np.exp(0.002*(Best_score-8000))) 
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, lb_array + ub_array - Positions, Positions)
    
    # Cosine-based elite guidance
    t = np.linspace(0, np.pi, SearchAgents_no)
    w = (np.cos(t)+1).reshape(-1,1)/2
    Positions = w*Best_pos + (1-w)*Positions
    
    # Boundary reflection
    Positions = np.where(Positions < lb_array, 2*lb_array-Positions, 
                        np.where(Positions > ub_array, 2*ub_array-Positions, Positions))
    #EVOLVE-END
    
    return Positions