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
    # Adaptive sigmoid-weighted OBL hybridization
    t = np.linspace(-3, 3, SearchAgents_no)
    w = 1 / (1 + np.exp(-t)) 
    opp_pos = (Positions + np.random.uniform(-0.2,0.2,Positions.shape)) % 1  # Jittered opposition 
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*opp_pos

    # Fitness-regulated elite guidance
    fitness_ratio = np.random.rand(SearchAgents_no,1) * (Best_score/(Best_score+1))
    elite_vec = Best_pos + 0.15*np.random.randn(*Positions.shape)
    Positions = (1-fitness_ratio)*Positions + fitness_ratio*elite_vec

    # Progressive-diminishing neighborhood perturbation
    decay = (1 - np.minimum(rg, 0.5))
    idx = (np.arange(SearchAgents_no) + np.random.randint(1,5)) % SearchAgents_no
    mask = np.random.rand(*Positions.shape) < 0.25*decay
    gradient = 0.5*(Positions[idx] - Positions) * np.random.randn(*Positions.shape)
    Positions = np.clip(np.where(mask, Positions+gradient, Positions), 0, 1)
    #EVOLVE-END       
    return Positions