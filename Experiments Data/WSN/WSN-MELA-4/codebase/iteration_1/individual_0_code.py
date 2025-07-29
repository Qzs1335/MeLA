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
    F = 0.5 * (1 + rg)  # Adaptive scaling factor
    CR = 0.9 - 0.5*rg   # Adaptive crossover rate
    
    # Differential evolution mutation with proper dimension handling
    idxs = np.random.permutation(SearchAgents_no)
    n = SearchAgents_no // 3
    a = Positions[idxs[:n]]
    b = Positions[idxs[n:2*n]]
    c = Positions[idxs[2*n:3*n]]
    
    # Ensure we have enough elements by repeating if necessary
    if len(a) < SearchAgents_no:
        a = np.vstack([a] * (SearchAgents_no//len(a) + 1))[:SearchAgents_no]
    if len(b) < SearchAgents_no:
        b = np.vstack([b] * (SearchAgents_no//len(b) + 1))[:SearchAgents_no]
    if len(c) < SearchAgents_no:
        c = np.vstack([c] * (SearchAgents_no//len(c) + 1))[:SearchAgents_no]
    
    mutant = a + F*(b - c)
    
    # Binomial crossover
    cross_points = np.random.rand(*Positions.shape) < CR
    Positions = np.where(cross_points, mutant, Positions)
    #EVOLVE-END
    
    return Positions