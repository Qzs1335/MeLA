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
    # Enhanced OBL with quadratic decay
    op_pos = 1 - Positions
    w = 0.9*(1-(np.arange(SearchAgents_no)/SearchAgents_no)**2)
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos

    # Elite-guided hybrid oscillation
    elite = Best_pos + 0.1*(Best_pos-Positions[np.random.permutation(SearchAgents_no)])
    rw = 0.5*(1+np.sin(np.linspace(0,np.pi,SearchAgents_no)*np.random.rand()))
    Positions = rw.reshape(-1,1)*Positions + (1-rw.reshape(-1,1))*elite

    # Probabilistic scaled mutation 
    neighbor_indices = np.random.choice(SearchAgents_no, size=SearchAgents_no)
    mask = np.random.rand(*Positions.shape) < 0.3
    scale = 0.1 + 0.9*np.random.rand(SearchAgents_no,1)
    perturbation = rg*scale*(Positions[neighbor_indices]-Positions)*np.random.rand(*Positions.shape)
    Positions = np.where(mask,np.clip(Positions+perturbation,0,1),Positions)
    #EVOLVE-END       
    return Positions