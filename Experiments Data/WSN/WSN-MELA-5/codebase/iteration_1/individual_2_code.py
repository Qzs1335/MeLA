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
    fitness_weight = 1.0 - np.exp(-Best_score/1000)
    neighborhood = 0.1 * rg * (np.random.rand(SearchAgents_no, dim) - 0.5)
    random_restart = (np.random.rand(SearchAgents_no) < 0.05).reshape(-1,1)
    
    exploit = Best_pos + fitness_weight * neighborhood
    explore = np.random.rand(*Positions.shape)
    Positions = np.where(random_restart, explore, exploit)
    #EVOLVE-END       
    return Positions