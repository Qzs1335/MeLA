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
    # Adaptive parameters with sigmoid scaling
    progress = 1 / (1 + np.exp(0.01*(Best_score-800)))
    w = 0.9 - 0.5*progress
    c1 = 2.5*(1-progress)
    c2 = 1.5 + progress
    
    # Elite-guided velocity with proper broadcasting
    elite_count = max(1, SearchAgents_no//4)
    elite_idx = np.random.choice(SearchAgents_no, size=elite_count, replace=False)
    elite_guide = np.mean(Positions[elite_idx], axis=0)  # Take mean of elite positions
    
    velocity = w*np.random.randn(*Positions.shape) + \
              c1*np.random.rand()*(Best_pos - Positions) + \
              c2*np.random.rand()*(elite_guide - Positions)
    velocity = np.clip(velocity, -0.2*rg, 0.2*rg)
    
    # Periodic boundary handling
    Positions = Positions + velocity
    Positions = np.where(Positions > ub_array, Positions%ub_array, Positions)
    Positions = np.where(Positions < lb_array, ub_array-(lb_array-Positions)%(ub_array-lb_array), Positions)
    #EVOLVE-END       
    
    return Positions