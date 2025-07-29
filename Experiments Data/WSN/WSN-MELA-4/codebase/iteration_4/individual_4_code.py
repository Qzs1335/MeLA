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
    # Enhanced adaptive parameters
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    w = 0.9 - 0.5*progress
    c1 = 1.5*(1-progress)
    c2 = 1.5 + progress
    
    # Hybrid velocity update
    velocity = w*np.random.randn(*Positions.shape) + \
              c1*np.random.rand(*Positions.shape)*(Best_pos-Positions) + \
              c2*np.random.rand(*Positions.shape)*(Positions[np.random.permutation(SearchAgents_no)]-Positions)
    
    # Clamped position update with probabilistic reflection
    velocity = np.clip(velocity, -0.2*rg, 0.2*rg)
    Positions += velocity
    reflect_prob = np.random.rand(*Positions.shape)
    Positions = np.where((Positions > ub_array) & (reflect_prob < 0.7), 
                        ub_array - (Positions-ub_array), 
                        np.where((Positions < lb_array) & (reflect_prob < 0.7), 
                        lb_array + (lb_array-Positions), 
                        Positions))
    #EVOLVE-END       
    
    return Positions