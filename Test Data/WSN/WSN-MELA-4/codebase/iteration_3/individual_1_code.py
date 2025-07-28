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
    progress = 1 - (Best_score / 1000)
    cos_factor = np.cos(np.pi * progress)  # Dynamic cosine factor
    
    # Hybrid PSO with elite guidance
    velocity = (0.4 + 0.3*cos_factor) * np.random.randn(*Positions.shape) + \
               (1.5*progress) * np.random.rand() * (Best_pos - Positions) + \
               (2.0 - 1.5*progress) * np.random.rand() * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Smooth boundary handling
    Positions = Positions + velocity * rg * (0.5 + 0.5*np.tanh(progress))
    Positions = np.where(Positions > ub_array, ub_array - 0.1*np.abs(np.random.randn(*Positions.shape)), Positions)
    Positions = np.where(Positions < lb_array, lb_array + 0.1*np.abs(np.random.randn(*Positions.shape)), Positions)
    #EVOLVE-END       
    
    return Positions