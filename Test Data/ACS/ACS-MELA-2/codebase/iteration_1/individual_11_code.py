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
    t = np.linspace(0, np.pi/2, SearchAgents_no)
    rotation_angles = np.linspace(0, np.pi/2, SearchAgents_no)
    
    # Dynamically create rotation matrices for the given dimension
    rotation_matrices = []
    for angle in rotation_angles:
        rot = np.eye(dim)
        if dim >= 2:
            rot_2d = np.array([[np.cos(angle), -np.sin(angle)], 
                              [np.sin(angle), np.cos(angle)]])
            rot[:2,:2] = rot_2d
        rotation_matrices.append(rot)
    rotation_matrices = np.array(rotation_matrices)
    
    step_sizes = 0.5 * (1 + np.cos(rotation_angles)).reshape(-1,1) + 0.1
    random_partners = Positions[np.random.permutation(SearchAgents_no)]
    
    # Perform the rotation and combination in a vectorized way
    Positions = step_sizes * np.einsum('nij,nj->ni', rotation_matrices, Positions) + \
               (1 - step_sizes) * random_partners
    #EVOLVE-END       
    return Positions