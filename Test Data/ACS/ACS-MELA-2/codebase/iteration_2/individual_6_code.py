import numpy as np
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    #EVOLVE-START
    # Levy flight parameters
    beta = 1.5 + 0.5*np.sin(rg*np.pi/2)  # Dynamic beta
    sigma_value = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)
                  /(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    
    # Progressive search strategy
    r1 = np.random.rand(SearchAgents_no,1)
    exploitation = (Best_pos - Positions) * r1 * 0.8 * (1 - rg/2)
    exploration = np.random.standard_cauchy((SearchAgents_no,dim)) * 0.1 * sigma_value * (1 + rg)
    
    # Adaptive switching
    adaptive_factor = 0.5 + 0.4*(1 - np.exp(-np.linalg.norm(Positions-Best_pos,axis=1)/Best_score))
    Pos_new = np.where(np.random.rand(SearchAgents_no,1) < adaptive_factor.reshape(-1,1),
                      Best_pos + exploration,
                      Positions + exploitation)
    
    # Boundary handling with reflection
    out_of_bounds = (Pos_new < 0) | (Pos_new > 1)
    Pos_new = np.where(out_of_bounds, np.abs(Pos_new) % 1, Pos_new)
    
    # Final selection
    improved_mask = (np.linalg.norm(Pos_new,axis=1) < np.linalg.norm(Positions,axis=1))
    Positions = np.where(improved_mask.reshape(-1,1), Pos_new, Positions)
    #EVOLVE-END
    
    return Positions