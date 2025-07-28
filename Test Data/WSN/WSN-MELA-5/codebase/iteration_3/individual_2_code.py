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
    # Enhanced dynamic inertia
    w = 0.9 * (1 - rg**2)
    
    # Fitness-proportional neighborhood
    neighbor_size = max(3, int(SearchAgents_no*0.25))
    dists = np.linalg.norm(Positions - Best_pos, axis=1)
    probs = 1/(1+dists)
    probs /= probs.sum()
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=probs, replace=False)
        local_best = Positions[neighbors[np.argmin(dists[neighbors])]]
        
        # Adaptive hybrid update
        r1, r2 = np.random.rand(2)
        cognitive = (1.5 - rg) * r1 * (Best_pos - Positions[i]) 
        social = (1.0 + rg) * r2 * (local_best - Positions[i])
        Positions[i] = w*Positions[i] + cognitive + social
    #EVOLVE-END       

    return Positions