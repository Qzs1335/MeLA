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
    # Enhanced cosine-based scaling
    theta = 2*np.pi*np.random.rand(SearchAgents_no,1)
    cos_scale = np.cos(theta)*(1-rg) + rg
    
    # Levy flight perturbation
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    levy = 0.01*rg*(np.random.randn(SearchAgents_no,dim)*sigma/np.abs(np.random.randn(SearchAgents_no,dim))**(1/beta))
    
    # Elite-guided search with adaptive threshold
    elite_threshold = 0.1 + 0.3*rg
    elite_mask = np.random.rand(SearchAgents_no,dim) < elite_threshold
    Positions = np.where(elite_mask, 
                        Best_pos*(1+0.5*cos_scale*np.random.randn(*Positions.shape)) + levy,
                        Positions*(1+cos_scale) + levy)
    
    # Reflective boundary handling
    boundary_mask = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(boundary_mask, 
                        np.clip(2*lb_array - Positions, lb_array, ub_array) if np.random.rand()<0.5 
                        else np.clip(2*ub_array - Positions, lb_array, ub_array),
                        Positions)
    #EVOLVE-END       
    
    return Positions