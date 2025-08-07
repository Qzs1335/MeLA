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
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)  # Safer scaling
    w = 0.5 * (1 + np.cos(np.pi*progress))  # Smoother inertia
    c1 = 1.5 * (1 - progress**0.5)  # Non-linear cognitive
    c2 = 1.5 * (1 + progress**0.5)  # Non-linear social
    
    # Hybrid PSO-cosine velocity
    elite_mask = (np.random.rand(SearchAgents_no,1) < progress)
    cos_term = np.cos(np.pi*np.random.rand(*Positions.shape))
    velocity = w*(elite_mask*cos_term + (1-elite_mask)*np.random.randn(*Positions.shape)) \
               + c1*np.random.rand()*(Best_pos - Positions) \
               + c2*np.random.rand()*(Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Dynamic boundary reflection
    Positions = np.clip(Positions + velocity*rg*(0.5+progress/2), -0.5, 1.5)
    Positions = np.where(Positions>ub_array, 2*ub_array-Positions, Positions)
    Positions = np.where(Positions<lb_array, 2*lb_array-Positions, Positions)
    #EVOLVE-END       
    
    return Positions