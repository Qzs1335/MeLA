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
    # Adaptive RG scaling
    rg = rg * (1 + np.tanh(np.random.randn(SearchAgents_no,1)))/2
    
    # Fitness-weighted OBL with cosine annealing
    w = 0.9 * np.cos(np.pi*(np.arange(SearchAgents_no)/SearchAgents_no)/2)**2
    op_pos = 1 - w.reshape(-1,1)*Positions
    Positions = 0.5*(Positions + op_pos)
    
    # Differential elite guidance
    elite = Best_pos + rg*(Best_pos - Positions[np.random.permutation(SearchAgents_no)])
    fitness_weights = (Positions.max(1).reshape(-1,1) - Positions.max(1).min())/(Positions.max(1).max()-Positions.max(1).min()+1e-8)
    Positions = fitness_weights*Positions + (1-fitness_weights)*elite
    
    # Adaptive probabilistic mutation
    mut_prob = np.clip(0.4*(1 - ((Positions-Best_pos)**2).mean(1)).reshape(-1,1),0.1,0.4)
    mask = np.random.rand(*Positions.shape) < mut_prob
    neighbor_indices = np.random.choice(SearchAgents_no, size=(SearchAgents_no,))
    perturbation = rg * np.random.randn(*Positions.shape) * (Positions[neighbor_indices] - Positions)
    Positions = np.where(mask, np.clip(Positions + perturbation, 0, 1), Positions)
    #EVOLVE-END       
    return Positions