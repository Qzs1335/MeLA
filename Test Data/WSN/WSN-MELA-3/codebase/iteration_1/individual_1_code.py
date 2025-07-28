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
    # Adaptive exploration-exploitation balance
    adaptive_rg = rg * (0.5 + 0.5 * np.random.rand())
    weights = 0.1 + 0.9 * (1 - np.exp(-Best_score/1000)) 
    
    # Differential evolution component
    a, b, c = np.random.choice(SearchAgents_no, 3, replace=False)
    mutant = Positions[a] + weights * (Positions[b] - Positions[c])
    
    # Local search around best solution
    local_search = Best_pos + adaptive_rg * np.random.normal(0, 0.1, dim)
    
    # Combine strategies
    mask = np.random.rand(SearchAgents_no, dim) < 0.7
    Positions = np.where(mask, 
                        weights * Positions + (1-weights) * mutant,
                        local_search + 0.1*(Positions - local_search))
    #EVOLVE-END       
    return Positions