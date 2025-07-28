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
    c1 = 1.5 * (1 - 0.5*rg) 
    c2 = 1.5 * (0.5 + 0.5*rg)
    
    # Fitness-proportional neighborhood
    neighbor_size = max(3, int(SearchAgents_no*0.2))
    distances = np.linalg.norm(Positions - Best_pos, axis=1)
    probs = 1/(1+distances)
    probs /= probs.sum()
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=probs, replace=False)
        local_best = Positions[neighbors[np.argmin(distances[neighbors])]]
        
        # Vectorized update
        r1, r2 = np.random.rand(2)
        Positions[i] = w*Positions[i] + c1*r1*(Best_pos-Positions[i]) + c2*r2*(local_best-Positions[i])
    
    # Boundary check
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), 
                        lb_array + (ub_array-lb_array)*np.random.rand(*Positions.shape), 
                        Positions)
    #EVOLVE-END       

    return Positions