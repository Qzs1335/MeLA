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
    # Enhanced fitness-based adaptive parameters
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)  # Controlled progress
    w = 0.9 * progress  # Non-linear inertia decay
    c1 = 1.5 * (1-progress)  # Inverse cognitive adaption
    c2 = 2.5 - c1  # Dynamic social component
    
    # Implement Levy flight using Mantegna algorithm
    beta = 1.5  # Levy exponent
    sigma = (np.math.gamma(1+beta) * np.sin(np.pi*beta/2) / 
            (np.math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, (SearchAgents_no, dim))
    v = np.random.normal(0, 1, (SearchAgents_no, dim))
    levy = u / (np.abs(v)**(1/beta))
    
    # Hybrid velocity update with Levy flight
    velocity = w * levy + \
              c1 * (Best_pos - Positions) * np.random.rand() + \
              c2 * (Positions[np.random.permutation(SearchAgents_no)] - Positions) * (0.5 + np.random.rand())
    
    # Smart boundary handling
    Positions += velocity * (rg * (0.5 + progress/2))
    Positions = np.where(Positions > ub_array, Best_pos + (ub_array - Best_pos)*0.9, Positions)
    Positions = np.where(Positions < lb_array, Best_pos - (Best_pos - lb_array)*0.9, Positions)
    #EVOLVE-END       
    
    return Positions