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
    levy_step = np.random.randn(*Positions.shape) * np.power(rg, 1.5)
    levy_mask = np.random.rand(SearchAgents_no, 1) < 0.3*rg
    Positions = np.where(levy_mask, Positions + levy_step, Positions)
    
    # Adaptive cosine exploitation
    cos_factor = np.cos(np.pi*rg/2)
    delta = np.abs(Best_pos - Positions)
    exploit_mask = np.random.rand(SearchAgents_no, 1) < (1-rg)
    Positions = np.where(exploit_mask, 
                        Best_pos - rg*cos_factor*delta*np.random.rand(*Positions.shape),
                        Positions)
    
    # Dynamic boundary reflection
    boundary_mask = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(boundary_mask, 
                        Best_pos + np.abs(ub_array-lb_array)*np.random.rand(*Positions.shape)*rg,
                        Positions)
    #EVOLVE-END       
    
    return Positions