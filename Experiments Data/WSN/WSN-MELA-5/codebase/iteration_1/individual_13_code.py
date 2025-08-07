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
    # Adaptive exploration with memory
    memory = 0.2 * np.random.randn(SearchAgents_no, dim)
    exploit_prob = 1 - (rg / 2)  # Adaptive based on rg
    
    # Local refinement around best solution
    local_search = 0.1 * (1 - exploit_prob) * np.random.randn(*Positions.shape)
    
    # Position update
    mask = np.random.rand(SearchAgents_no, 1) < exploit_prob
    Positions = np.where(mask,
                        Best_pos + local_search + memory,
                        Positions * (1 + 0.5*np.random.randn(*Positions.shape)))
    #EVOLVE-END       
    
    return Positions