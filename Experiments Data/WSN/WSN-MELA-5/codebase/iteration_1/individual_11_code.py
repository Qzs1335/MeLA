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
    # Dynamic weights based on rg (decreasing over iterations)
    w = 0.9 * (rg / 2.28)  # 2.28 is initial rg from history
    
    # Opposition-based learning for 30% of population
    opp_mask = np.random.rand(SearchAgents_no) < 0.3
    opposite_pos = lb_array + ub_array - Positions
    Positions[opp_mask] = opposite_pos[opp_mask]
    
    # Hybrid movement: PSO + DE mutation
    r1, r2 = np.random.randint(0, SearchAgents_no, 2)
    F = 0.5 + 0.5*np.random.rand()
    Positions = w*Positions + (1-w)*(Best_pos + F*(Positions[r1] - Positions[r2]))
    #EVOLVE-END       
    
    return Positions