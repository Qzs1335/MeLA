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
    # Fitness-proportional mutation
    mutation_rate = 0.1 + (0.4 * (1 - np.exp(-Best_score/1000)))
    mutate_mask = np.random.rand(*Positions.shape) < mutation_rate
    
    # Non-linear adaptive search
    adaptive_rg = rg * (1 + np.sin(np.pi * np.random.rand()))
    weights = np.exp(-np.linspace(0, 1, dim)).reshape(1, -1)
    
    # Enhanced position update
    cognitive = 1.5 * np.random.rand() * (Best_pos - Positions)
    social = 1.5 * np.random.rand() * (Positions.mean(axis=0) - Positions)
    velocity = (cognitive + social) * weights * adaptive_rg
    
    Positions = np.where(mutate_mask, 
                        Positions + velocity,
                        Positions * (0.9 + 0.2*np.random.rand(*Positions.shape)))
    #EVOLVE-END       
    return Positions