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
    cognitive = 1.5 + np.random.rand()*0.5
    social = 1.5 + np.random.rand()*0.5
    w = 0.7 + 0.2*np.random.rand()
    
    r1, r2 = np.random.rand(2)
    cognitive_term = cognitive * r1 * (Best_pos - Positions)
    social_term = social * r2 * (Best_pos[np.random.randint(SearchAgents_no)] - Positions)
    
    velocity = w * Positions + cognitive_term + social_term
    Positions = Positions + velocity
    
    decay_factor = 0.95
    exploit_mask = np.random.rand(*Positions.shape) < decay_factor
    Positions = np.where(exploit_mask, velocity, Positions + 0.1*(Best_pos - Positions))
    #EVOLVE-END
    
    return Positions