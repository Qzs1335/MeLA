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
    w = 0.9 - (0.9-0.4)*rg
    
    # Adaptive neighborhood
    neighbor_size = max(3, int(SearchAgents_no*(0.1 + 0.1*rg)))
    fitness = np.array([np.linalg.norm(p-Best_pos) for p in Positions])
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=fitness/fitness.sum(), replace=False)
        local_best = Positions[neighbors[np.argmin(fitness[neighbors])]]
        
        # Hybrid update with perturbation
        r1, r2 = np.random.rand(2)
        cognitive = 1.5 * r1 * (Best_pos - Positions[i])
        social = 1.5 * r2 * (local_best - Positions[i])
        perturbation = 0.1*rg*(np.random.rand(dim)-0.5)
        Positions[i] = w*Positions[i] + cognitive + social + perturbation
    #EVOLVE-END       

    return Positions