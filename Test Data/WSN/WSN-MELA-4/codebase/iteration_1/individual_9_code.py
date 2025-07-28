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
    # Adaptive inertia based on iteration progress
    w = 0.9 * (1 - np.exp(-5*(100-rg)/100))
    
    # Fitness-based neighborhood attraction
    fitness_weights = np.exp(-np.abs(Best_score - Positions)/Best_score)
    neighbor_attraction = np.random.rand() * fitness_weights * (Best_pos - Positions)
    
    # Hybrid update rule
    r1, r2 = np.random.rand(2)
    Positions = w*Positions + r1*neighbor_attraction + r2*(Best_pos - Positions)
    #EVOLVE-END       
    
    return Positions