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
    # Opposition-based learning
    opp_pos = lb_array + ub_array - Positions
    fitness = np.random.rand(SearchAgents_no, 1)
    Positions = np.where(fitness < 0.5, opp_pos, Positions)
    
    # Adaptive cosine movement
    t = 2*np.pi*np.random.rand(SearchAgents_no, 1)
    w = 0.9 - (0.5 * (Best_score/10000)) # Dynamic weight
    Positions = w*Positions + (1-w)*Best_pos + np.cos(t)*rg*np.random.randn(*Positions.shape)
    #EVOLVE-END       
    return Positions