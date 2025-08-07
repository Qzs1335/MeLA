import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Convert inputs to numpy arrays if they aren't already
    node_attr = np.asarray(node_attr)
    node_constraint = np.asarray(node_constraint)
    
    n = node_attr.shape[0]
    size_ratio = node_attr[:, None] / (node_attr[None, :] + 1e-6)
    
    # Handle scalar constraint case
    if node_constraint.ndim == 0:
        constraint_diff = np.abs(node_constraint - (node_attr[:, None] + node_attr[None, :]))
        max_constraint = node_constraint
    else:
        constraint_diff = np.abs(node_constraint - (node_attr[:, None] + node_attr[None, :]))
        max_constraint = np.max(node_constraint) if node_constraint.size > 0 else 1.0
    
    normalized_diff = constraint_diff / (max_constraint + 1e-6)
    weights = np.exp(-normalized_diff) * (0.5 + 0.5 * np.minimum(size_ratio, 1/size_ratio))
    return weights
    #EVOLVE-END