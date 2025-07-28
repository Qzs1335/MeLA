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
    # Adaptive inertia weight
    w = 0.9 - (0.5 * rg)
    
    # Cognitive and social factors
    c1 = 1.5 * np.random.rand()
    c2 = 2.0 * np.random.rand()
    
    # Hybrid velocity update
    velocity = w * Positions + c1 * np.random.rand() * (Best_pos - Positions) \
               + c2 * np.random.rand() * (Best_pos[np.random.randint(SearchAgents_no)] - Positions)
    
    # Local search around best solution
    if rg < 0.5:
        elite_mask = np.random.rand(*Positions.shape) < 0.1
        Positions = np.where(elite_mask, Best_pos + 0.1*rg*np.random.randn(*Positions.shape), velocity)
    else:
        Positions = velocity
    #EVOLVE-END       

    return Positions