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
    fitness_weights = 0.5 + np.random.rand(SearchAgents_no, 1) * (1 - Best_score/1000)
    turbulence = 0.1 * np.random.randn(*Positions.shape) * (1 - rg)
    
    # Adaptive search directions
    leader_attraction = 0.7 * (Best_pos - Positions)
    neighbor_attraction = 0.3 * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    Positions = Positions + rg * (fitness_weights * leader_attraction + 
                                (1-fitness_weights) * neighbor_attraction + 
                                turbulence)
    #EVOLVE-END       
    return Positions