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
    alpha = 0.5 * (1 + np.sin(2*np.pi*rg/100))  # Adaptive scaling
    beta = 1 - Best_score/(Best_score + rg)     # Fitness-based balance
    
    # Hybrid update with memory and perturbation
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    idx = int(rg) % SearchAgents_no             # Convert rg to integer for indexing
    memory = Positions[idx]                     # Cyclic memory with proper indexing
    gradient = Best_pos - Positions
    
    Positions = beta * (alpha * Positions + (1-alpha)*memory) + (1-beta) * (r1*gradient - r2*(Positions - Best_pos))
    #EVOLVE-END       

    return Positions