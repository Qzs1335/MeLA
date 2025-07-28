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
    # Adaptive exploration-exploitation balance
    progress = 1 - np.log(1 + rg)/np.log(100)
    w = 0.9 * progress + 0.1  # Inertia weight
    
    # Neighborhood search
    neighborhood = np.random.randint(0, SearchAgents_no, (SearchAgents_no, 3))
    local_best = np.mean(Positions[neighborhood], axis=1)
    
    # Dynamic mutation
    mutation_prob = 0.1 * (1 - progress)
    mutation_mask = np.random.rand(*Positions.shape) < mutation_prob
    mutation = (ub_array - lb_array) * np.random.rand(*Positions.shape) * 0.1
    
    # Position update
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    cognitive = 1.5 * r1 * (Best_pos - Positions)
    social = 1.5 * r2 * (local_best - Positions)
    Positions = w * Positions + cognitive + social
    Positions = np.where(mutation_mask, Positions + mutation, Positions)
    #EVOLVE-END       
    
    return Positions