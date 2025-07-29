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
    memory = 0.9 * Positions + 0.1 * Best_pos
    adaptive_radius = rg * (1 + np.sin(np.pi * np.random.rand(SearchAgents_no, 1)))
    
    local_search = np.random.normal(0, 0.1*rg, Positions.shape)
    elite_mask = np.random.rand(SearchAgents_no, 1) < 0.2
    Positions = np.where(elite_mask, 
                        Best_pos + local_search,
                        memory * (1 + adaptive_radius * np.random.randn(*Positions.shape)))
    #EVOLVE-END
    
    return Positions