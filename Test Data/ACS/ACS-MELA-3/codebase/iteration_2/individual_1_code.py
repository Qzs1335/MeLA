import numpy as np
def heuristics_v2(data_al, data_pb, Positions, Best_pos, Best_score, rg):
    #EVOLVE-START
    # Enhanced opposition+neighborhood search
    w1 = 0.5*(1 + np.cos(np.pi*(data_al)/data_pb))
    op_pos = (Positions.mean(0)*(1-w1) + w1*(1-Positions))
    Positions = 0.8*Positions + 0.2*op_pos
    
    # Smart dimensional learning
    D_leader = (np.repeat(Best_pos, Positions.shape[0], 0) - Positions)
    D_factor = 0.8*np.exp(-2*(data_al/data_pb)) 
    dim_learn = Positions + D_factor*D_leader

    # Dual mutation with probability
    mask1 = np.random.rand(*Positions.shape) < 0.4
    mask2 = np.random.rand(*Positions.shape) < 0.3 
    Positions = np.where(mask1, dim_learn, Positions)
    Positions = np.where(mask2, Positions + np.random.normal(0,0.1,Positions.shape), Positions)
    #EVOLVE-END      
    return Positions