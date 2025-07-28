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
    # Dynamic neighborhood based on rg
    neighborhood = 0.1 + 0.9 * (1 - rg)  # Increases as rg decreases
    
    # Hybrid exploration-exploitation
    mask = np.random.rand(SearchAgents_no, dim) < neighborhood
    r1 = np.random.randn(SearchAgents_no, dim) * (1 - rg)
    r2 = np.random.randn(SearchAgents_no, dim) * rg
    
    # Memory-guided perturbation
    memory = 0.5 * (Best_pos + Positions.mean(axis=0))
    Positions = np.where(mask,
                        Positions + r1 * (Best_pos - Positions) + r2 * (memory - Positions),
                        Positions * (1 + 0.1 * np.random.randn(SearchAgents_no, dim)))
    #EVOLVE-END
    
    return Positions