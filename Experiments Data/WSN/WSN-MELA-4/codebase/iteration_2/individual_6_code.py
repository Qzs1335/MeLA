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
    progress = np.clip(1 - (Best_score - 700)/(1000-700), 0.1, 0.9)  # Scaled progress
    w = 0.9 - 0.5*progress  # Dynamic inertia
    c1 = 1.5*(1-progress)    # Cognitive decay
    c2 = 1.5*progress        # Social growth
    
    # Hybrid velocity update
    cosine = np.cos(np.pi*(np.random.rand(SearchAgents_no,1)@np.random.rand(1,dim)))
    velocity = w*(Positions[np.random.permutation(SearchAgents_no)] - Positions)*cosine + \
               c1*np.random.rand()*(Best_pos - Positions) + \
               c2*np.random.rand()*(Positions.mean(0) - Positions)
    
    # Elite-guided position update
    elite_mask = np.random.rand(*Positions.shape) < 0.1
    Positions = np.where(elite_mask, Best_pos + 0.1*rg*np.random.randn(*Positions.shape),
                        np.clip(Positions + velocity*rg, lb_array, ub_array))
    #EVOLVE-END       
    
    return Positions