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
    if np.random.rand() < 0.2:  # Chaotic phase
        ch = 4*np.random.rand(SearchAgents_no,dim)*(1-np.random.rand(SearchAgents_no,dim))
        Positions = Positions * (1 + 0.5*(ch-0.5))
    
    # Opposition-based restart
    mask = (np.random.rand(SearchAgents_no,dim) < 0.1/np.sqrt(np.arange(1,dim+1)))
    Positions[mask] = ub_array[mask] - Positions[mask]
    
    # Adaptive exploitation
    exploitation = 0.9*(1 - np.min((1, 0.1*np.log(1+Best_score))))
    if np.random.rand() < exploitation:
        levy = np.random.randn(SearchAgents_no, dim)*np.abs(np.random.randn(SearchAgents_no, dim))**-0.5
        elite_mask = (np.random.rand(SearchAgents_no,1) < 0.3)
        Positions = np.where(elite_mask, Best_pos*(1+0.1*levy), Positions*0.95)
    else:
        orthogonal_dir = np.random.randn(SearchAgents_no, dim)-np.random.randn(SearchAgents_no,dim)
        Positions += 0.5*rg*np.linalg.norm(orthogonal_dir,axis=1,keepdims=True)*orthogonal_dir
    #EVOLVE-END       

    return Positions