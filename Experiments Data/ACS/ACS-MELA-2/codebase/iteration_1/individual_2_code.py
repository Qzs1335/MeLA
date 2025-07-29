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
    temp = 1 - (Best_score/13672.1875)  # Normalized simulated annealing temperature
    
    rand_matrix = np.random.rand(*Positions.shape)
    mutation = (ub_array - lb_array) * (0.1 + 0.9*temp) * np.tanh(rand_matrix)
    
    exploit_mask = (np.random.rand(SearchAgents_no, 1) > temp)
    exploration = Best_pos + rg * (2*np.random.rand(*Positions.shape)-1) * mutation
    
    Positions = np.where(exploit_mask, 
                        Best_pos + temp * (Positions - Best_pos),
                        Positions + exploration*(1-temp))
    #EVOLVE-END

    return Positions