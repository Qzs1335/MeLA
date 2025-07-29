import numpy as np
def heuristics_v1(node_attr, node_constraint):
    n = node_attr.shape[0]
    return np.ones((n, n))