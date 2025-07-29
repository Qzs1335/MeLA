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
    # Adaptive non-linear inertia weight
    w = 0.9 - (0.9-0.4)*(rg**0.5)
    
    # Fitness-proportional neighborhood search
    neighbor_size = max(3, int(SearchAgents_no*(0.2 + 0.3*rg)))
    distances = np.linalg.norm(Positions - Best_pos, axis=1)
    fitness = 1/(1+distances)
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=fitness/fitness.sum())
        local_best = Positions[neighbors[np.argmin(distances[neighbors])]]
        
        # Adaptive hybrid update
        r1, r2 = np.random.rand(2)
        cognitive = (1.5 - rg) * r1 * (Best_pos - Positions[i])
        social = (1.0 + rg) * r2 * (local_best - Positions[i])
        Positions[i] = np.clip(w*Positions[i] + cognitive + social, lb_array[i], ub_array[i])
    #EVOLVE-END       

    return Positions