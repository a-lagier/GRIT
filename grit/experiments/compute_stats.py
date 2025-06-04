import json
import numpy as np
from sys import argv


def print_final_result(name='molbace', n_exp=4):
    prec = 4

    scores = []
    for i in range(1, n_exp + 1):
        with open(f"results/{name}-GRIT-RRWP/{i}/logging.log") as f:
            lines = f.readlines()
            last_epoch_line = [l for l in lines if l.startswith('> ')][-1]
            scores.append(float(last_epoch_line[-7:-1]))

    print("The result is", round(np.mean(scores), prec), "±", round(np.std(scores), prec))

print_final_result(argv[1])