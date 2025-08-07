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
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.5, Positions, opposite_pos)
    
    # Adaptive movement
    a = 2 * (1 - (rg/2.28))  # 2.28 is initial rg from history
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    A = 2 * a * r1 - a
    C = 2 * r2
    D = np.abs(C * Best_pos - Positions)
    Positions = Best_pos - A * D * np.cos(2*np.pi*np.random.rand(SearchAgents_no, dim))
    #EVOLVE-END       
    return Positions