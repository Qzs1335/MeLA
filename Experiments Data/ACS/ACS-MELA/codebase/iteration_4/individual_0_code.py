import numpy as np
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START  
    T = 1/(1 + np.linspace(0, 10, SearchAgents_no))  # Reciprocal cooling 
    w = 1/(1 + np.exp(-0.01*Best_score))  # Sigmoid weight
    
    chaos = 0.7*np.ones((SearchAgents_no,dim))
    for _ in range(3): chaos = 3.67*chaos*(1-chaos)  # Chaotic perturbation
    R1 = np.random.rand(SearchAgents_no, dim)*chaos
    R2 = chaos*np.random.rand(SearchAgents_no, dim)
    
    explore_prob = 0.6*T.reshape(-1,1)  # Cooling-dependent exploration
    explore_mask = (np.random.rand(SearchAgents_no, dim) < explore_prob)
    
    pbest = Positions + np.random.rand(SearchAgents_no,dim)*(Best_pos-Positions)
    exploit_term = w*pbest + (1-w)*(Best_pos + R1*(pbest - R2*Positions))
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] * (1 + 0.5*(R1-0.5)*T.reshape(-1,1))  

    Positions = np.where(explore_mask, explore_term, exploit_term) 
    #EVOLVE-END
    
    return Positions