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
    F = 0.3 + 0.4 * np.cos(rg * np.pi/4)  # Cosine-modulated scaling
    CR = 0.95 / (1 + np.exp(0.1*rg))      # Nonlinear adaptive CR
    
    # Enhanced donor with elite guidance (fixed broadcasting)
    elite_idx = np.random.choice(SearchAgents_no, size=SearchAgents_no, replace=True)
    rand_idx = np.random.permutation(SearchAgents_no)
    donor = Positions + F*(Best_pos - Positions + Positions[elite_idx] - Positions[rand_idx])
    
    # Dynamic crossover with boundary awareness
    mask = (np.random.rand(*Positions.shape) < CR) & (np.random.rand(*Positions.shape) > 0.1)
    Positions = np.where(mask, donor, Positions)
    #EVOLVE-END       

    return Positions