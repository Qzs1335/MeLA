from wsn import NetworkEvaluation
import numpy as np
import sys
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


def solve():
    wsn = NetworkEvaluation()
    best_score = wsn.evaluate(priority=heuristics)

    return best_score


if __name__ == "__main__":
    print("[*] Running WSN optimization...")


    root_dir = sys.argv[1] if len(sys.argv) > 2 else "."
    mood = sys.argv[2] if len(sys.argv) > 3 else 'train'
    if mood == 'train':
        objs = []
        obj = solve()
        num_trials = 100
        for i in range(num_trials):
            print(f"[*] Trial {i}: {obj[i]}")
            objs.append(obj[i])

        print("[*] min objective value:")
        print(np.min(objs))