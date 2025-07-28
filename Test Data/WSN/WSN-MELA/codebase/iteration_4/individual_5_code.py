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
    
    # Adaptive cosine-based exploitation
    cos_scale = np.cos(np.pi/2*(1-rg))*np.random.rand(*Positions.shape)
    delta = np.abs(Best_pos - Positions)
    
    # Memory-based update
    memory = 0.5*(Positions + Best_pos) if rg < 0.5 else Positions
    
    # Combined update
    update_rule = np.random.rand(SearchAgents_no,1) < 0.5*rg
    Positions = np.where(update_rule,
                        memory + cos_scale*delta + levy,
                        Positions + 0.5*rg*(Best_pos - Positions))
    
    # Reflective boundary handling
    over = Positions > ub_array
    under = Positions < lb_array
    Positions = np.where(over, ub_array - 0.5*(Positions - ub_array), Positions)
    Positions = np.where(under, lb_array + 0.5*(lb_array - Positions), Positions)
    #EVOLVE-END       
    
    return Positions