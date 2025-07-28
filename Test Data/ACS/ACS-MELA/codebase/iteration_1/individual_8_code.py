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
    F = 0.8 * (1 - rg) + 0.2  # Adaptive scaling factor
    CR = 0.9 * rg + 0.1        # Adaptive crossover rate
    
    # Differential mutation
    a = np.random.randint(0, SearchAgents_no, SearchAgents_no)
    b = np.random.randint(0, SearchAgents_no, SearchAgents_no)
    c = np.random.randint(0, SearchAgents_no, SearchAgents_no)
    mutant = Positions[a] + F * (Positions[b] - Positions[c])
    
    # Binomial crossover
    crossover_mask = np.random.rand(SearchAgents_no, dim) < CR
    greedy_mask = np.tile((np.random.rand(SearchAgents_no) < rg).reshape(-1,1), (1, dim))
    Positions = np.where(greedy_mask, np.where(crossover_mask, mutant, Positions),
                        Best_pos - (Best_pos - Positions) * np.random.rand(SearchAgents_no, dim))
    #EVOLVE-END       

    return Positions