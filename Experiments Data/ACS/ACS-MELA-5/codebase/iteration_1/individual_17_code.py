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
    temp = 1.0 - rg  # Simulated annealing temperature
    rand_mask = np.random.rand(SearchAgents_no, dim) < temp
    
    # Exploration component (random walk)
    rand_walk = 0.1 * np.random.randn(SearchAgents_no, dim) * rand_mask
    
    # Exploitation component (directed move)
    directed_move = 0.2 * (Best_pos - Positions) * (1 - rand_mask)
    
    # Adaptive step size based on current best score
    step_size = 0.5 / (1 + np.exp(-Best_score/1000))
    Positions += step_size * (rand_walk + directed_move)
    #EVOLVE-END
    
    return Positions