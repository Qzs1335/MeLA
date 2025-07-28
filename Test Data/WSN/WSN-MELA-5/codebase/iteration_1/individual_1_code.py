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
    F = 0.5 * (1 + np.random.rand())
    CR = 0.9 * (1 - rg)
    
    for i in range(SearchAgents_no):
        if np.random.rand() < CR:
            idxs = [j for j in range(SearchAgents_no) if j != i]
            a, b, c = Positions[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)
            crossover_mask = np.random.rand(dim) < CR
            Positions[i] = np.where(crossover_mask, mutant, Positions[i])
    
    best_neighborhood = Best_pos + rg * (np.random.rand(dim) - 0.5)
    replace_idx = np.random.randint(0, SearchAgents_no)
    Positions[replace_idx] = best_neighborhood
    #EVOLVE-END       
    return Positions