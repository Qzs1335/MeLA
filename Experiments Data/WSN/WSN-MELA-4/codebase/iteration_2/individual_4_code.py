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
    # Enhanced progress calculation
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    w = 0.4 * (1 + np.cos(np.pi*progress))
    c1 = 1.5 * (1 - progress**0.5)
    c2 = 1.5 + 0.5*np.sin(np.pi*progress/2)
    
    # Elite-guided velocity
    elite_size = max(1, int(SearchAgents_no*0.2))
    elite_indices = np.random.choice(SearchAgents_no, elite_size, replace=False)
    elite_guide = np.mean(Positions[elite_indices], axis=0)
    
    velocity = w*np.random.randn(*Positions.shape) + \
              c1*np.random.rand()*(Best_pos - Positions) + \
              c2*np.random.rand()*(elite_guide - Positions) * rg
    
    # Adaptive mutation boundary handling
    mutation_prob = 0.1*(1-progress)
    mutate_mask = np.random.rand(*Positions.shape) < mutation_prob
    Positions = np.where(mutate_mask, 
                        Positions + rg*(ub_array-lb_array)*np.random.randn(*Positions.shape),
                        Positions + velocity)
    
    # Opposition-based diversity
    if np.random.rand() < 0.3:
        opposite_pos = lb_array + ub_array - Positions
        Positions = np.where(np.random.rand(*Positions.shape) < 0.5, opposite_pos, Positions)
    #EVOLVE-END       
    
    return Positions