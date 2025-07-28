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
    # Adaptive neighborhood search with guaranteed positive scale
    r_scale = np.clip(0.1 + 0.9 * (1 - rg), 0.01, 1.0)  # Bounded between 0.01 and 1.0
    neighborhood = np.random.normal(0, r_scale, Positions.shape)
    
    # Elite-guided perturbation
    elite_mask = np.random.rand(SearchAgents_no, dim) < 0.3
    elite_perturb = 0.5 * (Best_pos - Positions) * np.random.rand(*Positions.shape)
    
    # Memory-based update
    memory = 0.7 * Positions + 0.3 * Best_pos
    Positions = np.where(elite_mask, 
                        memory + neighborhood + elite_perturb,
                        memory + neighborhood)
    #EVOLVE-END       

    return Positions