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
    # Adaptive opposition with dynamic decay
    opposition = Best_pos + (Best_pos - Positions)  
    w = 0.7*(1 - np.exp(-5*np.linspace(0,1,SearchAgents_no))) + 0.3  
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*np.clip(opposition,0,1)
    
    # Elite hybrid using cos-adjusted weights
    phi = np.pi*(1 - np.exp(-np.random.rand(SearchAgents_no)/3))
    cw = (0.5*(1 + np.cos(phi))).reshape(-1,1)  
    elite = Best_pos + 0.05*np.random.randn(*Positions.shape)
    Positions = cw*Positions + (1-cw)*elite
    
    # Contextual neighborhood perturbation  
    direction = rg*np.random.choice([-1,1], Positions.shape)
    mask = np.random.rand(*Positions.shape) < 0.2*(1 + np.cos(Best_score))
    Positions = np.where(mask, np.clip(Positions + direction*elite, 0,1), Positions)
    #EVOLVE-END
    return Positions