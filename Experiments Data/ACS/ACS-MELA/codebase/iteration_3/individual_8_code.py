import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 1/(1+np.exp(np.linspace(0, 10, SearchAgents_no)))  # Sigmoid cooling
    w = np.clip(np.sqrt(Best_score)*rg/np.ptp(Positions),0,1)  # Normalized adaptive weight
    
    R1 = np.random.normal(0,0.5,(SearchAgents_no,dim))
    R2 = np.random.rand(*Positions.shape)
    
    rand_dir = 2*(np.random.rand(SearchAgents_no)<0.5)-1  # Directional diversity 
    explore_mask = np.random.rand(*Positions.shape) < 0.4*T.reshape(-1,1)
    
    exploit_term = Positions + w*(Best_pos - Positions) + rand_dir.reshape(-1,1)*R1/(rg+1e-7)
    cp_idx = np.random.choice(SearchAgents_no,size=SearchAgents_no,replace=True)
    explore_term = Positions[cp_idx] + T.reshape(-1,1)*(R1-R2-R2.mean())
    
    Positions = np.where(explore_mask, explore_term, exploit_term).clip(0,1)
    #EVOLVE-END
    
    return Positions