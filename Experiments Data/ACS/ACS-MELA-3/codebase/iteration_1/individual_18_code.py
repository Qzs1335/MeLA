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
    F = 0.6 + np.random.rand()*0.4  # Adaptive scaling factor
    CR = 0.9 - 0.5*(Best_score/rg)  # Dynamic crossover rate
    
    # Mutation and crossover
    idxs = np.random.permutation(SearchAgents_no)
    a,b,c = idxs[:3]
    mutant = Positions[a] + F*(Positions[b] - Positions[c])
    cross_points = np.random.rand(*Positions.shape) < CR
    Position_updates = np.where(cross_points, mutant, Positions)
    
    # Momentum-based blending
    alpha = 0.9 - 0.4*(Positions - Best_pos).mean()
    Positions = alpha*Position_updates + (1-alpha)*Best_pos
    #EVOLVE-END
    
    return Positions