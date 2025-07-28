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
    # Opposition-based learning for diversity
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.5, Positions, opposite_pos)
    
    # Adaptive weights based on iteration progress
    w = 0.9 - (0.5 * (rg / 2.28))  # 2.28 is initial rg from history
    
    # Elite guidance with dimensional learning
    elite_mask = np.random.rand(SearchAgents_no, dim) < w
    dim_learn = np.random.rand(SearchAgents_no, dim) * (Best_pos - Positions)
    Positions = np.where(elite_mask, Positions + dim_learn, Positions)
    
    # Dynamic mutation
    mutation_rate = 0.1 * (1 - rg/2.28)
    mut_mask = np.random.rand(SearchAgents_no, dim) < mutation_rate
    Positions = np.where(mut_mask, Positions * (0.5 + np.random.rand(SearchAgents_no, dim)), Positions)
    #EVOLVE-END       
    
    return Positions