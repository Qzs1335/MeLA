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
    # Levy-flight enhanced exploration
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    levy = 0.01*rg*np.random.randn(*Positions.shape)*sigma/np.abs(np.random.randn(*Positions.shape))**(1/beta)
    
    # Cosine-adaptive scaling
    theta = np.random.rand(SearchAgents_no,1)*2*np.pi
    cos_scale = (1 - np.cos(theta)) * (1 - rg)
    
    # Elite guidance with dynamic threshold
    elite_mask = np.random.rand(*Positions.shape) < 0.1*(1+cos_scale)
    elite_perturb = Best_pos + cos_scale*levy
    
    # Reflective boundary handling
    out_of_bounds = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(out_of_bounds, 
                        np.clip(2*lb_array - Positions, lb_array, ub_array) if np.random.rand()<0.5 
                        else np.clip(2*ub_array - Positions, lb_array, ub_array),
                        np.where(elite_mask, elite_perturb, Positions + cos_scale*levy))
    #EVOLVE-END       
    
    return Positions