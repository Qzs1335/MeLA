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
    # Dynamic mutation based on rg (decreasing over iterations)
    mutation_prob = 0.3 * rg
    mutation_mask = np.random.rand(*Positions.shape) < mutation_prob
    
    # Fitness-proportional neighborhood search
    fitness_weights = 1 - np.exp(-np.abs(Positions - Best_pos))
    neighborhood = Best_pos + rg * (np.random.rand(*Positions.shape) - 0.5) * fitness_weights
    
    # Combine strategies
    Positions = np.where(mutation_mask, 
                        neighborhood,
                        Positions + rg * (Best_pos - Positions) * np.random.rand(*Positions.shape))
    #EVOLVE-END       
    return Positions