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
    # Adaptive exploration-exploitation balance
    alpha = 0.5 * (1 + np.cos(np.pi * rg))
    beta = 1 - alpha
    
    # Fitness-guided directional search
    directional_step = alpha * (Best_pos - Positions) * np.random.rand(SearchAgents_no, dim)
    
    # Crossover between agents
    mask = np.random.rand(SearchAgents_no, dim) < 0.7
    crossover_partners = Positions[np.random.permutation(SearchAgents_no)]
    crossover_step = beta * (crossover_partners - Positions) * np.random.rand(SearchAgents_no, dim)
    
    Positions += directional_step + crossover_step
    #EVOLVE-END       
    return Positions