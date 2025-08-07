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
    adaptive_factor = 0.5 + 0.4 * np.sin(np.pi * rg)
    elite_idx = np.random.randint(0, SearchAgents_no)
    local_guides = Positions[np.random.permutation(SearchAgents_no)[:3]]
    
    exploitation = Best_pos + adaptive_factor * (local_guides.mean(0) - Positions)
    exploration = Positions[elite_idx] - Positions * adaptive_factor * np.random.rand()
    
    r = np.random.rand(SearchAgents_no, 1)
    Positions = np.where(r < 0.7, exploitation, exploration)
    #EVOLVE-END       

    return Positions