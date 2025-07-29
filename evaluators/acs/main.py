from acs import AcsEvaluation
import numpy as np
import sys
from data.data import Data_ACS, Data_Algorithm
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


def solve():


    # Initialize ACS evaluation
    acs = AcsEvaluation()

    # Run the optimization process using the heuristic function
    best_score = acs.evaluate(priority=heuristics)

    return best_score


if __name__ == "__main__":
    print("[*] Running ACS optimization...")


    root_dir = sys.argv[1] if len(sys.argv) > 2 else "."
    mood = sys.argv[2] if len(sys.argv) > 3 else 'train'

    if mood == 'train':
        print(f"[*] Training with 30 students")
        objs = []
        obj = solve()
        num_trials = 50
        for i in range(num_trials):
            print(f"[*] Trial {i}: {obj[i]}")
            objs.append(obj[i])

        print("[*] min objective value:")
        print(np.min(objs))