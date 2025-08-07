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
    # Enhanced dynamic components
    w = 0.9 * (1 - 0.5*(rg**2))  # Non-linear decay
    cos_perturb = np.cos(2*np.pi*np.random.rand(SearchAgents_no,1)) * rg
    
    # Fitness-proportional neighborhood
    dist_to_best = np.linalg.norm(Positions - Best_pos, axis=1)
    prob = 1/(1+dist_to_best)
    prob /= prob.sum()
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, 3, p=prob, replace=False)
        local_best = Positions[neighbors[np.argmin(dist_to_best[neighbors])]]
        
        # Elite-guided hybrid update
        r1, r2 = np.random.rand(2)
        cognitive = 1.7 * r1 * (Best_pos - Positions[i])
        social = 1.3 * r2 * (local_best - Positions[i]) 
        Positions[i] = w*Positions[i] + cognitive + social + cos_perturb[i]*np.random.randn(dim)
    #EVOLVE-END       

    return Positions