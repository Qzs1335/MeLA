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
    # Adaptive exploration rate
    adaptive_rg = rg * (1 + np.sin(np.pi * Best_score/1000))
    
    # Opposition-based learning
    opp_positions = 1 - Positions
    combined = np.vstack((Positions, opp_positions, Best_pos))
    fitness = np.array([np.linalg.norm(pos - Best_pos) for pos in combined])
    top_idx = np.argpartition(fitness, SearchAgents_no)[:SearchAgents_no]
    Positions = combined[top_idx]
    
    # Local refinement
    noise = adaptive_rg * np.random.randn(*Positions.shape)
    Positions = np.where(np.random.rand(*Positions.shape) < 0.3, 
                        Positions + noise, 
                        Positions)
    #EVOLVE-END

    return Positions