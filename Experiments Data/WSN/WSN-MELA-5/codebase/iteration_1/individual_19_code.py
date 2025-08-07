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
    # Adaptive search based on rg (decreasing radius)
    w = 0.9 * (rg / 2.28)  # Weight decays with rg (initial rg=2.28)
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid mutation combining best and random directions
    cognitive = r1 * (Best_pos - Positions)
    social = r2 * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    Positions = Positions + w * (cognitive + social)
    
    # Periodic random jumps to escape local optima
    if np.random.rand() < 0.1:
        jumpers = np.random.choice(SearchAgents_no, size=int(0.1*SearchAgents_no), replace=False)
        Positions[jumpers] = lb_array[jumpers] + (ub_array[jumpers]-lb_array[jumpers])*np.random.rand(len(jumpers), dim)
    #EVOLVE-END       
    return Positions