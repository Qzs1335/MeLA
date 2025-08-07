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
    w = 0.9 - (0.9-0.2)*rg
    c1 = 1.7 - rg*0.7
    c2 = 1.7 - rg*0.7
    
    # Fitness-weighted neighborhood
    neighbor_size = max(3, int(SearchAgents_no*(0.3 - rg*0.15)))
    fitness = np.array([np.linalg.norm(p - Best_pos) for p in Positions])
    norm_fitness = 1 - (fitness - fitness.min())/(fitness.max()-fitness.min()+1e-12)
    
    for i in range(SearchAgents_no):
        # Roulette-wheel selection
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=norm_fitness/norm_fitness.sum())
        local_best = Positions[neighbors[np.argmin(fitness[neighbors])]]
        
        # Hybrid update with memory
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (Best_pos - Positions[i])
        social = c2 * r2 * (local_best - Positions[i])
        Positions[i] = w*Positions[i] + cognitive + social
    #EVOLVE-END       

    return Positions