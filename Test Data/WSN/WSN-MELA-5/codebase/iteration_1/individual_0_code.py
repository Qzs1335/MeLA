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
    opposite_pos = lb_array + ub_array - Positions
    eval_pos = np.vstack([Positions, opposite_pos])
    
    # Fitness-proportional mutation
    mutation_rate = 0.1 + (0.4 * (1 - rg))
    mutate_mask = np.random.rand(*Positions.shape) < mutation_rate
    mutation = rg * np.random.normal(0, 0.5, Positions.shape)
    Positions = np.where(mutate_mask, Positions + mutation, Positions)
    
    # Elite-guided search
    elite_guide = Best_pos.reshape(1, -1).repeat(SearchAgents_no, axis=0)  # Ensure same dimensions
    Positions = Positions + rg * (elite_guide - Positions) * np.random.rand(*Positions.shape)
    #EVOLVE-END
    
    return Positions