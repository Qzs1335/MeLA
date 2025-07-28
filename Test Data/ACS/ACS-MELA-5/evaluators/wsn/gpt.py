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
    # Enhanced Levy flight with epsilon guard
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, Positions.shape)
    v = np.random.normal(0, 1, Positions.shape)
    epsilon = 1e-10
    levy = 0.01 * u / (np.abs(v)**(1/beta) + epsilon)
    
    # Dynamic adaptive weights
    w = (0.9 - 0.4*rg) * (1 + 0.1*np.sin(rg*np.pi/2))  # Oscillating decay
    
    # Neighborhood-based social learning
    Best_pos_reshaped = np.tile(Best_pos.reshape(1, -1), (SearchAgents_no, 1))
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = r1 * (Best_pos_reshaped - Positions)
    
    # Safe neighborhood selection
    neighborhood_size = min(max(2, int(SearchAgents_no*(0.1 + 0.4*rg))), SearchAgents_no - 1)
    if SearchAgents_no > 1:
        all_indices = np.tile(np.arange(SearchAgents_no), (SearchAgents_no, 1))
        for i in range(SearchAgents_no):
            mask = all_indices[i] != i
            available = all_indices[i, mask]
            permuted = np.random.permutation(available)[:neighborhood_size]
            all_indices[i, :neighborhood_size] = permuted
        permuted_indices = all_indices[:, :neighborhood_size]
        social = r2 * (np.mean(Best_pos_reshaped[permuted_indices], axis=1) - Positions)
    else:
        social = np.zeros_like(cognitive)
    
    # Bounded levy terms
    levy = np.where((levy < -1) | (levy > 1), 0.5*np.random.randn(*levy.shape), levy)
    Positions = w*Positions + cognitive + social + levy
    #EVOLVE-END       

    return Positions
