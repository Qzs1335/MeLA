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
    momentum = 0.15*Positions + 0.85*Best_pos
    diffs = Positions - (momentum + rg*np.random.randn(*Positions.shape)*np.clip(Best_score,0.001,1))
    opp_positions = lb_array + ub_array - Positions
    if np.random.rand() < 0.2:
        best_mask = np.random.rand(*Positions.shape) < 0.3/Positions.shape[0]
        Positions = np.where(best_mask, Best_pos, (opp_positions+opp_positions)/3 + diffs/3 + momentum/3)
    else:
        mutation = 0.5*(Best_pos + Positions[np.random.randint(0,SearchAgents_no)]) 
        crossover_mask = np.random.rand(*Positions.shape) < 0.7 
        Positions = np.where(crossover_mask, mutation, Positions)
    #EVOLVE-END

    return Positions