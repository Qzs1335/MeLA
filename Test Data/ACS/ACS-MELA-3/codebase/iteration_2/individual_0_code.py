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
    # Opposition-based learning with adaptive exponential decay
    op_pos = 1 - Positions
    decay = np.exp(-2*np.arange(SearchAgents_no)/SearchAgents_no)
    w = 0.5 + 0.4*decay 
    mask_dim = np.random.choice([True, False], size=Positions.shape, p=[0.5, 0.5])
    Positions = np.where(mask_dim, w.reshape(-1,1)*Positions+(1-w.reshape(-1,1))*op_pos, Positions)
    
    # Hybrid elite guidance with Levy & Gaussian components
    alpha = np.random.rand(SearchAgents_no,1)*0.1 + 0.6
    levy = np.random.normal(0, alpha, Positions.shape)*(Best_pos-Positions)
    gauss = np.random.normal(0,1-rep.exp(Best_score/rg),Positions.shape)*(Best_pos-Positions)
    elite_learn = Positions + alpha*levy + (1-alpha)*gauss
    
    # Probability-based application
    mask_elite = np.random.rand(*Positions.shape) < 0.6 - decay.reshape(-1,1)*0.2
    Positions = np.where(mask_elite, elite_learn, Positions)
    #EVOLVE-END       
    return Positions