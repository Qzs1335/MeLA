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
    c1 = 2.5 - 2*rg  # Cognitive decay
    c2 = 0.5 + 2*rg  # Social growth
    
    # Fitness-proportional neighborhood
    neighbor_size = max(3, int(SearchAgents_no*0.2))
    fitness = np.array([np.linalg.norm(pos-Best_pos) for pos in Positions])
    prob = 1/(fitness+1e-8)
    prob /= prob.sum()
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=prob, replace=False)
        local_best = Positions[neighbors[np.argmin(fitness[neighbors])]]
        
        # Clamped hybrid update
        r1, r2 = np.random.rand(2)
        velocity = w*Positions[i] + c1*r1*(Best_pos-Positions[i]) + c2*r2*(local_best-Positions[i])
        Positions[i] = np.clip(velocity, lb_array[i], ub_array[i])
        
        # Mutation
        if np.random.rand() < 0.1*rg:
            Positions[i] += 0.1*(ub_array[i]-lb_array[i])*np.random.randn(dim)
    #EVOLVE-END       

    return Positions