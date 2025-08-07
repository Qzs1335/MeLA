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
    w = 0.5*(1 + np.cos(rg*np.pi))  # Non-linear inertia
    
    # Fitness-proportional neighbors
    dists = np.linalg.norm(Positions - Best_pos, axis=1)
    probs = 1/(1+dists)
    probs /= probs.sum()
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, 3, p=probs, replace=False)
        local_best = Positions[neighbors[np.argmin(dists[neighbors])]]
        
        r1, r2 = np.random.rand(2)
        Positions[i] = w*Positions[i] + 1.5*r1*(Best_pos-Positions[i]) + 1.5*r2*(local_best-Positions[i])
    #EVOLVE-END       

    return Positions