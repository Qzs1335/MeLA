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
    # Adaptive weights
    w = 1/(1+np.exp(-10*(rg-0.5))) 
    
    # Elite-guided cosine search
    theta = np.random.rand(SearchAgents_no,dim)*2*np.pi
    elite_mask = np.random.rand(SearchAgents_no,1)<0.3*rg
    cosine_perturb = np.cos(theta)*rg*(Best_pos-Positions)
    
    # Fitness-proportional neighborhood
    dists = np.linalg.norm(Positions-Best_pos,axis=1)
    probs = 1/(1+dists)
    for i in range(SearchAgents_no):
        neighbors = np.random.choice(SearchAgents_no,3,p=probs/probs.sum())
        local_best = Positions[neighbors[np.argmin(dists[neighbors])]]
        
        # Hybrid update
        r1,r2 = np.random.rand(2)
        Positions[i] = w*Positions[i] + (1-w)*(r1*(Best_pos-Positions[i]) + r2*(local_best-Positions[i]))
    
    Positions = np.where(elite_mask, Positions+cosine_perturb, Positions)
    #EVOLVE-END       

    return Positions