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
    F = 0.5 + 0.3 * np.sin(rg * np.pi/50)  # Adaptive scaling
    rand_agents = Positions[np.random.permutation(SearchAgents_no)]  # Random permutation
    rand_agents2 = Positions[np.random.permutation(SearchAgents_no)]
    mutation = Best_pos + F * (rand_agents - rand_agents2)  # Simplified differential mutation
    cos_wave = np.cos(2*np.pi*rg/100)  # Remove reshape as scalar is fine
    Positions = cos_wave * mutation + (1-cos_wave) * (Best_pos - 0.5*Positions)
    #EVOLVE-END       
    return Positions