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
    F = 0.5 * (1 + np.sin(2 * np.pi * rg / 100))  # Adaptive scaling factor
    CR = 0.9 * (1 - np.exp(-rg/50))  # Adaptive crossover rate
    
    # Differential evolution mutation with wrapped indexing
    idxs = np.random.permutation(SearchAgents_no)
    a = Positions[idxs]
    b = Positions[np.roll(idxs, SearchAgents_no//3)]
    c = Positions[np.roll(idxs, 2*SearchAgents_no//3)]
    mutant = Best_pos + F * (a - b) + F * (Best_pos - c)
    
    # Crossover with adaptive CR
    cross_points = np.random.rand(*Positions.shape) < CR
    Positions = np.where(cross_points, mutant, Positions)
    #EVOLVE-END       

    return Positions