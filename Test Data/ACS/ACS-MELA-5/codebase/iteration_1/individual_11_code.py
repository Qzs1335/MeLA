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
    o_Positions = 1 - Positions
    fit_original = np.apply_along_axis(fitness_func, 1, Positions)
    fit_opposite = np.apply_along_axis(fitness_func, 1, o_Positions)
    Positions = np.where((fit_opposite < fit_original).reshape(-1,1), o_Positions, Positions)
    
    # Adaptive perturbation
    alpha = 0.5 * (1 + np.cos(np.pi * (Best_score/initial_score)))
    perturbation = alpha * rg * (Best_pos - Positions) * np.random.normal(0, 1, Positions.shape)
    Positions = Positions + perturbation
    #EVOLVE-END       

    return Positions