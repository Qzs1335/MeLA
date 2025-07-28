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
    # Non-linear inertia weight
    w = 0.9*(1 - rg**2)
    
    # Adaptive neighborhood search
    neighbor_size = max(3, int(SearchAgents_no*(0.3 - 0.1*rg)))
    for i in range(SearchAgents_no):
        # Fitness-proportional selection
        distances = np.linalg.norm(Positions - Best_pos, axis=1)
        probs = 1/(1+distances)
        probs /= probs.sum()
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=probs, replace=False)
        
        # Hybrid update with adaptive coefficients
        r1, r2 = np.random.rand(2)
        c1 = 2.5 - 2*rg  # Cognitive coefficient
        c2 = 0.5 + 2*rg  # Social coefficient
        cognitive = c1 * r1 * (Best_pos - Positions[i])
        social = c2 * r2 * (Positions[neighbors].mean(0) - Positions[i])
        Positions[i] = w*Positions[i] + cognitive + social + 0.1*rg*np.random.randn(dim)
    #EVOLVE-END       

    return Positions