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
    # Opposition-based learning for diversity
    opposite_pos = lb_array + ub_array - Positions
    combined_pop = np.vstack((Positions, opposite_pos))
    fitness = np.array([np.linalg.norm(pos - Best_pos) for pos in combined_pop])
    top_indices = np.argpartition(fitness, SearchAgents_no)[:SearchAgents_no]
    Positions = combined_pop[top_indices]
    
    # Adaptive nonlinear parameter control
    t = rg / 100  # Normalized iteration counter
    a = 2 * (1 - t**2)
    r1 = a * (2 * np.random.rand() - 1)
    r2 = 2 * a * np.random.rand()
    
    # Memory-guided search
    memory = 0.1 * np.random.randn(SearchAgents_no, dim)
    Positions = Positions + r1 * (Best_pos - Positions) + r2 * memory
    #EVOLVE-END       

    return Positions