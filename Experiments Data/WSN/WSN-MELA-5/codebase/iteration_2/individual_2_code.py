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
    # Enhanced dynamic parameters
    w = 0.9 * (1 - rg**2)  # Non-linear decay
    elite = Positions[np.argmin([np.linalg.norm(p-Best_pos) for p in Positions])]
    
    # Fitness-proportional neighborhood
    dists = np.array([np.linalg.norm(p-Best_pos) for p in Positions])
    probs = 1/(1+dists)
    probs /= probs.sum()
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, size=3, p=probs, replace=False)
        local_best = Positions[neighbors[np.argmin(dists[neighbors])]]
        
        # Clamped hybrid update
        r1, r2 = np.random.rand(2)
        cognitive = (1.5 - rg) * r1 * (Best_pos - Positions[i])
        social = (0.5 + rg) * r2 * (local_best - Positions[i])
        Positions[i] = np.clip(w*Positions[i] + cognitive + social + 0.1*(elite-Positions[i]), 0, 1)
    #EVOLVE-END       

    return Positions