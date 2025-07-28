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
    w = 0.9 * (1 - rg**2)  # Non-linear decay
    
    neighbor_probs = np.exp(-np.linalg.norm(Positions - Best_pos, axis=1))
    neighbor_probs /= neighbor_probs.sum()
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, min(5,SearchAgents_no-1), p=neighbor_probs, replace=False)
        local_best = Positions[neighbors[np.argmin(np.linalg.norm(Positions[neighbors] - Best_pos, axis=1))]]
        
        r1, r2, r3 = np.random.rand(3)
        cognitive = 1.7*r1*(Best_pos - Positions[i])
        social = 1.7*r2*(local_best - Positions[i])
        elite = 0.3*r3*(Best_pos - Positions[i])
        Positions[i] = w*Positions[i] + cognitive + social + elite
        
        # Boundary check
        Positions[i] = np.clip(Positions[i], lb_array[i], ub_array[i])
    #EVOLVE-END       

    return Positions