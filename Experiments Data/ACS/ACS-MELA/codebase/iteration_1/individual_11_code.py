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
    # Levy flight component
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = u/np.abs(v)**(1/beta)
    levy_step = 0.01 * step * (Positions - Best_pos)
    
    # Adaptive mutation
    mutation_prob = np.maximum(0.1, 1 - np.exp(-Best_score/rg))
    mutation_mask = np.random.rand(*Positions.shape) < mutation_prob
    gaussian_mutation = np.random.normal(0, 0.1, Positions.shape)
    
    # Combined update
    Positions = Positions + levy_step
    Positions[mutation_mask] += gaussian_mutation[mutation_mask]
    
    # Dimension-aware perturbation
    active_dims = np.random.rand(dim) > 0.7
    Positions[:, active_dims] *= 1 + 0.1*np.random.randn(np.sum(active_dims))
    #EVOLVE-END

    return Positions