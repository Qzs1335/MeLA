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
    
    # Adaptive exploration-exploitation
    scale = 0.1 * rg * (Positions - Best_pos)
    levy = np.random.randn(SearchAgents_no, dim) * sigma / (np.abs(np.random.randn(SearchAgents_no, dim))**(1/beta))
    
    # Position update
    exploration_mask = np.random.rand(SearchAgents_no, dim) > 0.5
    Positions = np.where(exploration_mask,
                        Positions + scale * levy,
                        Best_pos + 0.1*(np.random.rand(SearchAgents_no, dim)-0.5))
    #EVOLVE-END

    return Positions