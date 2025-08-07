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
    # Improved chaotic map (logistic) - vectorized
    chaos = 4.0 * rg * (1.0 - rg) * (1.0 - 0.5*rg)
    
    # Dynamic opposition probability
    opp_prob = np.clip(0.7 - 0.5*(Best_score/10000), 0.1, 0.9)
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, opposite_pos, Positions)
    
    # Adaptive learning with elite guidance
    t = np.pi * np.random.rand(SearchAgents_no, 1)
    w = (0.9 - 0.4*t/np.pi) * chaos
    
    # Safe elite selection
    if np.random.rand() > 0.3:
        elite = Best_pos
    else:
        min_idx = np.argmin(np.sum(Positions, axis=1))  # Get agent index with minimum sum
        elite = Positions[min_idx].reshape(1, -1)  # Ensure proper shape
    
    new_pos = elite * (1 - w) + Positions * w
    
    # Dynamic scaling with chaotic factor
    scale = 0.8 + 0.2*chaos*np.random.rand(SearchAgents_no, dim)
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.7, new_pos * scale, new_pos)
    #EVOLVE-END
    
    return Positions