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
    # Dynamic inertia weight
    w = 0.9 - (0.9-0.4)*(rg/2.28) 
    
    # Cognitive and social components
    c1 = 1.5 * np.random.rand()
    c2 = 1.5 * np.random.rand()
    
    # Velocity update
    velocity = w * np.random.randn(*Positions.shape) + \
               c1 * np.random.rand() * (Best_pos - Positions) + \
               c2 * np.random.rand() * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Position update with adaptive step
    Positions = Positions + velocity * (0.1 + 0.9*rg/2.28)
    
    # Local search around best solution
    if rg < 0.5:
        mask = np.random.rand(*Positions.shape) < 0.3
        Positions = np.where(mask, Best_pos + 0.1*rg*np.random.randn(*Positions.shape), Positions)
    #EVOLVE-END
    
    return Positions