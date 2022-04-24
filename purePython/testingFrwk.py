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

import v4main as algs

proj_path = os.path.join(
    "/", "home", "rjb255", "researchProject", "researchProject"
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


def main(*, output=0):
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
    alpha: list = [0]

    # todo - Minimise alpha

    with Pool() as p:
        scores = p.map(algs.post_main, [data for data in data_test])
    results = pd.DataFrame(data=scores, index=data_test)
    # ppprint(results)
    plt.ion()
    plt.plot([1, 101, 201, 301, 401], np.transpose(scores))
    plt.show(block=True)
    results.to_csv('output.csv')


if __name__ == "__main__":
    main(output=1)