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
    # Adaptive mutation
    mutation_rate = 0.1 * (1 - np.exp(-0.1 * rg))
    mutation_mask = np.random.rand(*Positions.shape) < mutation_rate
    mutation_values = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    
    # Crossover with best solutions
    crossover_mask = np.random.rand(*Positions.shape) < 0.7
    elite_count = min(5, SearchAgents_no)
    elite_indices = np.random.choice(SearchAgents_no, elite_count, replace=False)
    elite_pool = Positions[elite_indices]
    
    # Generate random selections from elite pool for each position
    elite_selections = elite_pool[np.random.randint(0, elite_count, size=SearchAgents_no)]
    
    # Random restart for 10% of worst solutions
    if rg % 10 == 0:
        worst_idx = np.argsort(Best_score)[-int(0.1*SearchAgents_no):]
        Positions[worst_idx] = lb_array[worst_idx] + (ub_array[worst_idx] - lb_array[worst_idx]) * np.random.rand(len(worst_idx), dim)
    
    # Apply updates
    Positions = np.where(mutation_mask, mutation_values, Positions)
    Positions = np.where(crossover_mask, elite_selections, Positions)
    #EVOLVE-END
    
    return Positions