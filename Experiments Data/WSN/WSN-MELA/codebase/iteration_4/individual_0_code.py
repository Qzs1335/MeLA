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
    levy = 0.01*rg*sigma*np.random.randn(*Positions.shape)/np.abs(np.random.randn(*Positions.shape))**(1/beta)
    
    # Non-linear elite guidance
    elite_scale = 0.2*(1+np.sin(rg*np.pi/2))
    elite_mask = np.random.rand(SearchAgents_no, dim) < elite_scale
    Positions = np.where(elite_mask, 
                        Best_pos + levy, 
                        Positions*(1-0.5*rg) + 0.5*rg*Best_pos)
    
    # Reflective boundary handling
    over = Positions > ub_array
    under = Positions < lb_array
    Positions = np.where(over, 2*ub_array-Positions, Positions)
    Positions = np.where(under, 2*lb_array-Positions, Positions)
    Positions = np.clip(Positions, lb_array, ub_array)
    #EVOLVE-END       
    
    return Positions