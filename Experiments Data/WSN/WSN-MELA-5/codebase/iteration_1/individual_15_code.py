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
    w = 0.9 - (0.9-0.4)*rg
    
    # Neighborhood search
    neighbor_size = max(3, int(SearchAgents_no*0.2))
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, replace=False)
        # Get fitness values for neighbors
        neighbor_fitness = np.array([np.linalg.norm(Positions[n] - Best_pos) for n in neighbors])
        local_best_idx = neighbors[np.argmin(neighbor_fitness)]
        local_best = Positions[local_best_idx]
        
        # Hybrid update
        r1, r2 = np.random.rand(2)
        cognitive = 1.5 * r1 * (Best_pos - Positions[i])
        social = 1.5 * r2 * (local_best - Positions[i])
        Positions[i] = w*Positions[i] + cognitive + social
    #EVOLVE-END       

    return Positions