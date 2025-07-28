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
    # Adaptive neighborhood search
    neighborhood_size = max(3, int(SearchAgents_no * (1 - rg)))
    indices = np.random.choice(SearchAgents_no, neighborhood_size, replace=False)
    local_best = Positions[indices][np.argmin([Best_score]*neighborhood_size)]
    
    # Non-linear scaling factor
    alpha = 2 * np.exp(-2 * (Best_score / 500)) 
    beta = 0.5 + np.random.rand() * (1 - rg)
    
    # Hybrid update rule
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = alpha * r1 * (Best_pos - Positions)
    social = beta * r2 * (local_best - Positions)
    Positions += cognitive + social
    #EVOLVE-END       

    return Positions