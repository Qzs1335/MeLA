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
    w = 0.9 - (0.9-0.4)*rg
    c1 = 2.5 - 2*rg
    c2 = 0.5 + 2*rg
    
    # Diversity-based neighborhood
    diversity = np.mean(np.std(Positions, axis=0))
    neighbor_size = max(3, int(SearchAgents_no*(0.1 + 0.1*(1-rg))))
    
    for i in range(SearchAgents_no):
        # Elite preservation for top 10%
        if i < SearchAgents_no//10:
            continue
            
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, replace=False)
        dists = np.linalg.norm(Positions[neighbors] - Best_pos, axis=1)
        local_best = Positions[neighbors[np.argmin(dists)]]
        
        # Clamped velocity update
        r1, r2 = np.random.rand(2)
        velocity = w*Positions[i] + c1*r1*(Best_pos-Positions[i]) + c2*r2*(local_best-Positions[i])
        Positions[i] = np.clip(velocity, lb_array[i], ub_array[i])
        
        # Non-uniform mutation
        if rg < 0.3 and np.random.rand() < 0.1:
            idx = np.random.randint(0, dim)
            Positions[i,idx] += np.random.normal(0, 0.1*(ub_array[i,idx]-lb_array[i,idx]))
    #EVOLVE-END       

    return Positions