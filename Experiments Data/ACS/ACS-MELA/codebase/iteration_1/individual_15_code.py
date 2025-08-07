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
    if np.random.rand() < 0.3:
        opposite_pos = lb_array + ub_array - Positions
        opposite_fitness = Positions.min()
        if opposite_fitness < Best_score:
            Positions[-1] = opposite_pos.mean(axis=0)
            
    # Adaptive exploration/exploitation
    w = 0.7 - 0.4*np.arange(100)[:SearchAgents_no]/100
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = 1.5 * r1 * (Best_pos - Positions)
    social = 1.5 * r2 * (Best_pos.mean(axis=0) - Positions)
    Positions += w.reshape(-1,1)*(cognitive + social)
    #EVOLVE-END       
    
    return Positions