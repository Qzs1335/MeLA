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
    fitness_weights = np.exp(-np.linspace(0, 1, SearchAgents_no)).reshape(-1, 1)
    elite_mask = (np.random.rand(SearchAgents_no, 1) < 0.3)
    
    # Neighborhood search with adaptive radius
    neighbor_dist = rg * (0.5 + 0.5*np.random.rand(SearchAgents_no, dim))
    neighbor_pos = Positions + neighbor_dist * np.random.randn(*Positions.shape)
    
    # Fitness-guided update
    Positions = np.where(elite_mask,
                        Best_pos + 0.1*rg*np.random.randn(*Positions.shape),
                        fitness_weights*neighbor_pos + (1-fitness_weights)*Positions)
    #EVOLVE-END

    return Positions