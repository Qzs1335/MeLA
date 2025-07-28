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
    # Adaptive mutation based on rg
    mutation_prob = 0.3 * (1 - rg)  # Decreases as rg decreases
    
    # Opposition-based learning for exploration
    if np.random.rand() < 0.2:
        Positions = 0.5 * (Positions + (ub_array - Positions - lb_array))
    
    # Crossover with best positions
    crossover_mask = np.random.rand(*Positions.shape) < 0.7
    Positions = np.where(crossover_mask, 
                        Positions + 0.5*(Best_pos - Positions), 
                        Positions)
    
    # Local search around best positions
    local_search_mask = np.random.rand(*Positions.shape) < mutation_prob
    Positions = np.where(local_search_mask,
                        Best_pos + rg * np.random.randn(*Positions.shape),
                        Positions)
    #EVOLVE-END       
    
    return Positions