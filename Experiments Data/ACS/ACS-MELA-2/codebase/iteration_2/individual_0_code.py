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
    beta = 1.5 
    u = np.random.normal(0, (np.pi/2)**(1/beta), (SearchAgents_no,1))
    v = np.random.normal(0, 1, (SearchAgents_no,1))
    levy_step = u/(np.abs(v)**(1/beta)) * 0.05
    
    prob = 1/(1+np.exp(-np.linalg.norm(Positions-Best_pos,axis=1)/Best_score))
    mask = np.random.rand(SearchAgents_no,dim) < prob.reshape(-1,1)
    
    center = np.mean(Best_pos, axis=0)
    radius = np.linalg.norm(Positions - center, axis=1)[:,None]
    
    Positions = np.where(mask,
                       center + levy_step * (Positions - center) + 0.1 * (Best_pos - center) * np.random.rand(*Positions.shape), 
                       Best_pos + 0.01 * radius * (2 * np.random.rand(*Positions.shape) - 1))
    #EVOLVE-END       
    return Positions