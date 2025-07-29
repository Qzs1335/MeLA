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
    # Sigmoid progress scaling
    progress = 1/(1+np.exp(0.01*(Best_score-700)))  
    w = 0.9 - 0.5*progress
    c1 = 2.5*progress
    c2 = 2.5 - c1
    
    # Elite-guided velocity with clamping
    elite_idx = np.random.choice(SearchAgents_no, size=5, replace=False)
    elite_mean = np.mean(Positions[elite_idx], axis=0)
    velocity = np.clip(w*np.random.randn(*Positions.shape) + \
              c1*np.random.rand()*(Best_pos-Positions) + \
              c2*np.random.rand()*(elite_mean-Positions), -0.2, 0.2)
    
    # Adaptive boundary reflection
    Positions += velocity*rg
    reflect = np.where(Positions>ub_array, 2*ub_array-Positions, 
                      np.where(Positions<lb_array, 2*lb_array-Positions, Positions))
    Positions = np.where(np.abs(Positions-reflect)>0.1*rg, reflect, Positions)
    #EVOLVE-END       
    
    return Positions