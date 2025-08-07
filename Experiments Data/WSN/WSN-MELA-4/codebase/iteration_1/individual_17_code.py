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
    # Opposition-based learning (modified to avoid calling Best_score)
    opposite_pos = lb_array + ub_array - Positions
    combined_pop = np.vstack((Positions, opposite_pos))
    
    # Randomly select half of the combined population (since we can't evaluate fitness)
    selected_idx = np.random.choice(combined_pop.shape[0], SearchAgents_no, replace=False)
    Positions = combined_pop[selected_idx]

    # Adaptive step control
    w = 0.9 * (1 - np.exp(-5 * (1 - rg)))
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = 1.5 * r1 * (Best_pos - Positions)
    social = 1.5 * r2 * (Best_pos[np.random.randint(0, SearchAgents_no)] - Positions)
    Positions = w * Positions + cognitive + social
    #EVOLVE-END

    return Positions