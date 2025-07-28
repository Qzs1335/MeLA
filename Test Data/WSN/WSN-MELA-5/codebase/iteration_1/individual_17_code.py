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
    progress = 1 - rg  # Normalized [0,1] where 1=start, 0=end
    mutation_rate = 0.1 + 0.4 * progress  # More mutation early
    crossover_prob = 0.7 - 0.4 * progress  # More crossover late
    
    # Adaptive step sizes
    step = 0.2 * rg * (1 + np.random.randn(*Positions.shape))
    
    # Mutation
    mask = np.random.rand(*Positions.shape) < mutation_rate
    Positions[mask] += step[mask]
    
    # Crossover with broadcasting
    cross_mask = np.random.rand(*Positions.shape) < crossover_prob
    Best_pos_expanded = np.tile(Best_pos, (SearchAgents_no, 1))
    Positions[cross_mask] = 0.7*Positions[cross_mask] + 0.3*Best_pos_expanded[cross_mask]
    #EVOLVE-END
    
    return Positions