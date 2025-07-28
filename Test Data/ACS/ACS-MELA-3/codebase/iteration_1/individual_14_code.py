import numpy as np
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)

    #EVOLVE-START
    temp = 1 - Best_score/10000  # Normalized temperature
    mutation_rate = temp * 0.3
    direction = np.random.choice([-1,1], Positions.shape)
    mutation_strength = np.random.uniform(0.1, 0.5) * temp
    
    explore_mask = np.random.rand(*Positions.shape) < mutation_rate
    Positions = np.where(explore_mask,
                        Positions + mutation_strength * direction,
                        (Positions + Best_pos) / 2)  # Exploit toward best
    #EVOLVE-END       

    return Positions