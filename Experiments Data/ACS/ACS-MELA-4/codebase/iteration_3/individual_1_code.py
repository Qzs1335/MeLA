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
    # Enhanced adaptive chaotic search
    iter_factor = np.clip(Best_score/10000, 0.1, 0.9)
    chaos = 4 * rg * (1 - rg) * (1 - iter_factor*np.random.rand())
    
    # Stochastic opposition learning
    opp_prob = 0.5 + 0.4*np.sin(iter_factor*np.pi/2)
    opposite_pos = lb_array + ub_array - Positions
    mask = (np.random.rand(SearchAgents_no, dim) < opp_prob*(1-chaos))
    Positions = np.where(mask, opposite_pos, Positions)
    
    # Differential evolution crossover
    a,b = np.random.choice(SearchAgents_no, 2, replace=False)
    F = 0.5*(1 + chaos)
    mutant = Positions + F*(Positions[a] - Positions[b])
    cross_points = np.random.rand(SearchAgents_no, dim) < 0.5
    Positions = np.where(cross_points, mutant, Positions)
    
    # Dimension-aware elite guidance
    elite_scale = 0.1 + 0.9*np.exp(-5*iter_factor)
    w = elite_scale * np.exp(-np.arange(1,dim+1)/dim)
    Positions = Best_pos*w + Positions*(1-w) + chaos*np.random.randn(SearchAgents_no, dim)*0.1
    #EVOLVE-END
    
    return Positions