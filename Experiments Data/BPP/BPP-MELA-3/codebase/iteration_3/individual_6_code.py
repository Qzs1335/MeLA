import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    src_attr = node_attr[:, None]
    tgt_attr = node_attr[None, :]
    constraint_diff = np.maximum(1e-6, abs(node_constraint - (src_attr + tgt_attr)))
    weights = 1 / (1 + np.log(1 + constraint_diff * (src_attr + tgt_attr)/2))
    return weights
    #EVOLVE-END