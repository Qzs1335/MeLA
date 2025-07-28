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
    # Levy flight component
    levy = np.random.standard_cauchy(SearchAgents_no).reshape(-1,1) * 0.1 / (1+progress)
    
    w = 0.4 + 0.4 * progress
    c1 = 1.5 * progress 
    c2 = 2.5 - c1
    
    velocity = w * levy + \
               c1 * np.random.rand() * (Best_pos - Positions) + \
               c2 * np.random.rand() * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    Positions = Positions + velocity * rg
    reflect_prob = np.random.rand(*Positions.shape)
    Positions = np.where((Positions > ub_array) & (reflect_prob<0.7), 
                         ub_array - (Positions-ub_array), Positions)
    Positions = np.where((Positions < lb_array) & (reflect_prob<0.7), 
                         lb_array + (lb_array-Positions), Positions)
    #EVOLVE-END       
    
    return Positions