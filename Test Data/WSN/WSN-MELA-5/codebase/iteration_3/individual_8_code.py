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
    # Adaptive parameters
    w = 0.9 - 0.5*rg  # Linear decay
    neighbor_size = max(2, int(SearchAgents_no*(0.3-0.2*rg)))
    
    # Cosine exploration factor
    cos_factor = np.cos(rg*np.pi/2)  
    
    for i in range(SearchAgents_no):
        # Dynamic neighborhood
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, replace=False)
        dists = np.linalg.norm(Positions[neighbors]-Best_pos, axis=1)
        local_best = Positions[neighbors[np.argmin(dists)]]
        
        # Hybrid update
        r1, r2 = np.random.rand(2)
        cognitive = 1.5*r1*(Best_pos - Positions[i])
        social = 1.5*r2*(local_best - Positions[i])
        Positions[i] = w*Positions[i] + (1-w)*(cognitive + cos_factor*social)
    #EVOLVE-END       

    return Positions