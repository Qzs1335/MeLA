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
    # Differential mutation
    F = 0.5 + 0.3 * np.sin(rg * np.pi/50)  # Dynamic scaling
    a, b, c = np.random.choice(SearchAgents_no, 3, replace=False)
    mutant = Positions[a] + F * (Positions[b] - Positions[c])
    
    # Opposition-based learning
    opp_pos = 1 - Positions  # Simple opposition
    combined = np.vstack((Positions, opp_pos, mutant))
    fitness = np.array([np.linalg.norm(pos - Best_pos) for pos in combined])
    top_idx = np.argpartition(fitness, SearchAgents_no)[:SearchAgents_no]
    Positions = combined[top_idx]
    #EVOLVE-END

    return Positions