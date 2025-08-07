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
    # Hybrid PSO-cosine strategy
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    w = 0.9 - 0.5*progress
    c1 = 1.5*(1-progress)
    c2 = 1.5*progress
    
    # Elite-guided cosine perturbation
    theta = 360 * np.random.rand(SearchAgents_no)
    cos_factor = np.cos(np.deg2rad(theta)).reshape(-1,1)
    elite_mask = (np.random.rand(SearchAgents_no,1) < progress)
    
    # Adaptive position update
    cognitive = c1 * np.random.rand() * (Best_pos - Positions)
    social = c2 * np.random.rand() * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    Positions = np.where(elite_mask,
                        Positions + w*(cognitive + social)*cos_factor,
                        Positions + w*np.random.randn(*Positions.shape)*rg)
    
    # Robust boundary handling
    Positions = np.where(Positions > ub_array, 
                        ub_array - np.random.rand()*(Positions-ub_array), 
                        np.where(Positions < lb_array, 
                                lb_array + np.random.rand()*(lb_array-Positions), 
                                Positions))
    #EVOLVE-END       
    
    return Positions