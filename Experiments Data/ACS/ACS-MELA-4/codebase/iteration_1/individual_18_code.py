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
    # Opposition-based learning
    opposition_pos = 1 - Positions
    combined_pop = np.vstack((Positions, opposition_pos))
    fitness = np.array([np.linalg.norm(pos - Best_pos) for pos in combined_pop])
    top_indices = np.argpartition(fitness, SearchAgents_no)[:SearchAgents_no]
    Positions = combined_pop[top_indices]

    # Adaptive mutation
    mutation_rate = 0.5 * (1 - np.exp(-Best_score/10000))
    mutation_mask = np.random.rand(*Positions.shape) < mutation_rate
    Positions = np.where(mutation_mask, 
                       Positions + rg * np.random.normal(0, 0.1, Positions.shape),
                       Positions)
    #EVOLVE-END       

    return Positions