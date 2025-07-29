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
    # Levy flight enhanced exploration
    levy = np.random.randn(SearchAgents_no, dim) * np.sqrt(rg/np.abs(np.random.randn(SearchAgents_no, dim)))
    Positions += 0.5*(1-rg)*levy
    
    # Cosine-based local search
    theta = 2*np.pi*np.random.rand(SearchAgents_no, 1)
    cos_search = np.cos(theta) * (Best_pos - Positions) * rg
    Positions += 0.5*rg*cos_search
    
    # Reflective boundary handling
    over = Positions > ub_array
    under = Positions < lb_array
    Positions = np.where(over, 2*ub_array-Positions, Positions)
    Positions = np.where(under, 2*lb_array-Positions, Positions)
    #EVOLVE-END       
    
    return Positions