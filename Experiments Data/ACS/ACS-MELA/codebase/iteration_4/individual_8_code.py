import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 0.5*np.cos(np.linspace(0,np.pi,SearchAgents_no)) + 1.5  # Smoother temperature wave
    w = (1.5 - np.exp(-Best_score*rg))  # Non-linear adaptive weight
    
    beta = 1 + np.random.rand() * (Best_score/(rg+1e-6))  # Fitness-sensitive scaling
    R1 = beta * np.random.randn(SearchAgents_no, dim)
    R2 = np.random.standard_cauchy((SearchAgents_no,dim))  # Heavy-tailed exploration
    
    explore_mask = np.random.rand(SearchAgents_no,dim) < (0.3 + 0.4*T.reshape(-1,1))  # Dynamic threshold
    exploit_term = Best_pos + w*(R1*Best_pos - (T.reshape(-1,1)*R2*Positions))
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = beta*Positions[cp_idx] * (1 + T.reshape(-1,1)*(R1 - np.tan(R2)))
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions