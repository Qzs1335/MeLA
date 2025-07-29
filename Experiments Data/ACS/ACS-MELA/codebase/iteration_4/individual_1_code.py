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
    T = 1/(1+np.linspace(0, 10, SearchAgents_no))  # Enhanced cooling
    w  = np.tanh(Best_score/rg)  # Normalized weight
    
    R1 = np.random.randn(SearchAgents_no, dim)
    R2 = np.random.randn(SearchAgents_no, dim)
    
    explore_mask = (np.random.rand(SearchAgents_no, dim) < 0.3 + 0.5*T.reshape(-1,1))
    exploit_term = Best_pos*(1-w) + w*(R1*Best_pos - 0.5*R2*Positions)
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] + np.abs(T.reshape(-1,1)*(R1-R2))
    
    Positions = np.where(explore_mask, explore_term*(1+w), exploit_term)
    #EVOLVE-END
    
    return Positions