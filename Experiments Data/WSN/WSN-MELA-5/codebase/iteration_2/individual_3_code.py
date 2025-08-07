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
    # Enhanced adaptive components
    w = 0.9 * (1 - rg**2)  # Non-linear decay
    neighbor_size = max(3, int(SearchAgents_no*(0.1 + 0.1*rg)))  # Adaptive neighborhood
    
    for i in range(SearchAgents_no):
        # Fitness-proportional neighbor selection
        dists = np.linalg.norm(Positions - Best_pos, axis=1)
        probs = np.exp(-dists/dists.mean())
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=probs/probs.sum(), replace=False)
        
        # Hybrid update with velocity clamping
        local_best = Positions[neighbors[np.argmin(dists[neighbors])]]
        r1, r2 = np.random.rand(2)
        velocity = w*Positions[i] + 1.5*r1*(Best_pos-Positions[i]) + 1.5*r2*(local_best-Positions[i])
        Positions[i] = np.clip(velocity, lb_array[i], ub_array[i])
        
        # Non-uniform mutation
        if np.random.rand() < 0.1*rg:
            mutation = 0.1*(ub_array[i]-lb_array[i])*(1-rg)*np.random.randn(dim)
            Positions[i] = np.clip(Positions[i] + mutation, lb_array[i], ub_array[i])
    #EVOLVE-END       

    return Positions