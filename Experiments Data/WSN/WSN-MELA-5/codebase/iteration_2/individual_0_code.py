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
    # Adaptive parameters
    w = 0.9 * (1 - rg**2)  # Non-linear decay
    neighbor_size = max(2, int(SearchAgents_no*(0.1 + 0.1*rg)))  # Adaptive neighborhood
    
    # Lévy flights
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    levy = 0.01*step*(Positions - Best_pos)
    
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no, neighbor_size, replace=False)
        distances = np.linalg.norm(Positions[neighbors] - Best_pos, axis=1)
        local_best = Positions[neighbors[np.argmin(distances)]]
        
        # Hybrid update with memory
        r1, r2 = np.random.rand(2)
        cognitive = (1.5-0.5*rg) * r1 * (Best_pos - Positions[i])
        social = (1.5-0.5*rg) * r2 * (local_best - Positions[i])
        Positions[i] = w*Positions[i] + cognitive + social + levy[i]
    #EVOLVE-END       
    return Positions