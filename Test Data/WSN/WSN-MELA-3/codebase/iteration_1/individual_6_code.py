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
    # Levy flight parameters
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape)*sigma
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    
    # Adaptive mutation
    mutation_prob = 0.1 + (0.4 * (1 - rg))
    mutate_mask = np.random.rand(*Positions.shape) < mutation_prob
    
    # Opposition-based learning
    opp_positions = lb_array + ub_array - Positions
    select_opp = np.random.rand(*Positions.shape) < 0.5*rg
    
    # Combined update
    Positions = np.where(select_opp, opp_positions,
                np.where(mutate_mask, 
                         Best_pos + step * (Best_pos - Positions),
                         Positions))
    #EVOLVE-END
    
    return Positions