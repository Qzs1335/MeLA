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
    beta = 1.5 + np.random.rand()
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    step = 0.01 * (np.random.rand(*Positions.shape) * sigma / (np.abs(np.random.normal(0,1,size=Positions.shape)))**(1/beta))
    
    quantum_phase = np.random.rand(*Positions.shape)*2*np.pi
    r = abs(np.random.normal(0,1,(SearchAgents_no,1)))
    damping = 0.5 + 0.5*Best_score/(Best_score+1e-15)
    
    Positions = (damping*Positions + (r + 0.1*Best_pos)*(np.sin(quantum_phase)+1) + np.where(step >= 0.5, step*Best_pos, step))/2/abs(damping)
    #EVOLVE-END
    
    return Positions