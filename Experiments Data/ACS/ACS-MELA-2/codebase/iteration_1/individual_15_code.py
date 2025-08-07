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
    # Fitness-based exploitation
    probs = np.exp(-np.linalg.norm(Positions - Best_pos, axis=1))
    probs /= probs.sum()
    exploit_mask = np.random.rand(SearchAgents_no) < probs
    
    # Crossover with best solution
    cross_mask = np.random.rand(SearchAgents_no, dim) < 0.7
    Positions[exploit_mask] = np.where(cross_mask[exploit_mask], 
                                     (Positions[exploit_mask] + Best_pos)/2,
                                     Positions[exploit_mask])

    # Gaussian mutation for exploration
    mutation_mask = np.random.rand(SearchAgents_no, dim) < 0.1*rg
    Positions = np.where(mutation_mask,
                        Positions + np.random.normal(0, 0.1, (SearchAgents_no, dim)),
                        Positions)
    #EVOLVE-END       
    return Positions