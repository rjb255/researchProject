# %% Entire
# region Libraries
import os
import sys
from pprint import pprint
from functools import partial
import matplotlib.pyplot as plt

from pathos.multiprocessing import ProcessPool as Pool
import numpy as np
import random
import pandas as pd
import scipy.optimize as opt

import v4main as algs

proj_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)

# sys.path.insert(1, proj_path)
# from purePython.modules.shared.custom import split, getPI, Models

# endregion
# *
# ?
# !
# //
# todo

# region Subtle functions - functions similar to those in Python but improved in some way
def custom_print(output: int, *args, **kwargs):
    if output == 0:
        pass
    elif output == 1:
        pprint(*args, **kwargs)
    elif output == 2:
        with open("logs/log.txt", "a+") as f:
            f.write(*args, **kwargs)


# endregion

def to_minimise(data_set, alpha):
    exit
    temp = partial(algs.post_main, alpha=alpha)
    with Pool() as p:
        scores = p.map(temp, [data for data in data_set])
    print(f"alpha score: {np.mean(scores)}")
    return np.mean(scores)

def main(*, output=0, alpha=[]):
    ppprint = partial(custom_print, output)
    ppprint(output)
    data_location = os.path.join(proj_path, "data", "big", "qsar_data")

    data_names = os.listdir(data_location)
    random.shuffle(data_names)
    datasets = [os.path.join(data_location, data) for data in data_names]
    split = [int(len(datasets) * 0.8), int(len(datasets) * 0.8)]
    data_train = datasets[: split[0]]
    data_valid = datasets[split[0] : split[1]]
    data_test = datasets[split[1] :]

    ppprint(f"{len(data_train)}, {len(data_valid)}, {len(data_test)}")
    alpha = []
    # todo - Minimise alpha
    a0: list = [0.85, 50, 1, 1]
    a_boundary = [(0.5, 1), (10, 250), (0, 4), (0, 4)]

    alpha = opt.minimize(lambda a: to_minimise(data_train, a), a0, bounds=a_boundary)
    
    with_alpha = partial(algs.post_main, alpha=alpha)
    with Pool() as p:
        scores = p.map(with_alpha, [data for data in data_test])

    results = pd.DataFrame(data=scores, index=data_test)
    # ppprint(results)
    plt.ion()
    plt.plot([1, 101, 201, 301, 401], np.transpose(scores))
    plt.show(block=True)
    results.to_csv('output.csv')


if __name__ == "__main__":
    main(output=1)