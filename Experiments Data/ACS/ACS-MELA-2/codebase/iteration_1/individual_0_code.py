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
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    
    neighbor_idx = np.random.randint(0, SearchAgents_no, (SearchAgents_no, dim))
    neighbor_weights = np.random.rand(SearchAgents_no, dim)
    crossover = np.random.rand(SearchAgents_no, dim) < 0.7
    
    temperature = max(0.01, 1 - rg/50)
    anneal_prob = np.exp(-(Best_score - 666)/temperature) 
    
    mask = np.random.rand(*Positions.shape) < 0.5
    neighbors = Positions[neighbor_idx, np.arange(dim)[None,:]].reshape(SearchAgents_no, dim)
    Positions = np.where(mask,
        Best_pos + step * 0.01 * Positions * (1 if np.random.rand() < anneal_prob else -1),
        neighbor_weights*Positions + (1-neighbor_weights)*neighbors
    )
    #EVOLVE-END       
    return Positions