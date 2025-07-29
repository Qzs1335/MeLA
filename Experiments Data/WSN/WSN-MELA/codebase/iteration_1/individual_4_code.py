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
    # Adaptive mutation based on Best_score history
    F = 0.5 + 0.3 * np.sin(np.pi * rg / 2)  # Dynamic scaling factor
    
    # Differential evolution mutation
    idxs = np.random.choice(SearchAgents_no, (SearchAgents_no, 3), replace=True)
    a, b, c = Positions[idxs[:,0]], Positions[idxs[:,1]], Positions[idxs[:,2]]
    mutant = a + F * (b - c)
    
    # Crossover with adaptive probability
    cross_prob = 0.9 * (1 - rg)
    cross_mask = np.random.rand(SearchAgents_no, dim) < cross_prob
    Positions = np.where(cross_mask, mutant, Positions)
    
    # Elite preservation
    elite_mask = (np.random.rand(SearchAgents_no) < 0.1).reshape(-1,1)
    Positions = np.where(elite_mask, Best_pos, Positions)
    #EVOLVE-END
    
    return Positions