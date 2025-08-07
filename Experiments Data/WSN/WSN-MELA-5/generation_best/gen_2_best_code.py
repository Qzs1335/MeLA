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
    w = 0.9 * (1 - rg**2)
    c1 = 1.7 - rg
    c2 = 0.5 + rg
    
    # Personal best memory
    if not hasattr(heuristics_v2, 'Pbest'):
        heuristics_v2.Pbest = Positions.copy()
    
    # Fitness-weighted neighborhood
    neighbor_size = max(3, int(SearchAgents_no*(0.1 + 0.1*rg)))
    fitness = np.array([np.linalg.norm(p - Best_pos) for p in Positions])
    weights = 1/(1 + fitness)
    
    for i in range(SearchAgents_no):
        # Update personal best
        if np.linalg.norm(Positions[i] - Best_pos) < np.linalg.norm(heuristics_v2.Pbest[i] - Best_pos):
            heuristics_v2.Pbest[i] = Positions[i].copy()
            
        # Select neighbors
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=weights/weights.sum(), replace=False)
        local_best = Positions[neighbors[np.argmin(fitness[neighbors])]]
        
        # Hybrid update with cosine perturbation
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (heuristics_v2.Pbest[i] - Positions[i])
        social = c2 * r2 * (local_best - Positions[i])
        perturbation = 0.5*rg * np.cos(np.pi*np.random.rand())*(Best_pos - Positions[i])
        Positions[i] = w*Positions[i] + cognitive + social + perturbation
    #EVOLVE-END       

    return Positions