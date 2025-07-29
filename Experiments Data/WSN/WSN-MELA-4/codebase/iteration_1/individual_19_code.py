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
    # Opposition-based learning
    opposition_pos = 1 - Positions
    combined_pos = np.vstack((Positions, opposition_pos))
    fitness = np.sum(combined_pos**2, axis=1)
    top_indices = np.argpartition(fitness, SearchAgents_no)[:SearchAgents_no]
    Positions = combined_pos[top_indices]
    
    # Adaptive parameter control
    w = 0.9 * np.exp(-rg)  # Exponential decay
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Memory-enhanced update
    memory = 0.5 * (Positions + Best_pos)
    Positions = w * Positions + r1 * (Best_pos - Positions) + r2 * (memory - Positions)
    #EVOLVE-END       
    
    return Positions