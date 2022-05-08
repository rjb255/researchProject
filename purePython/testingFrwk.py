# %% Entire
# region Libraries
import os
import sys
from pprint import pprint
from functools import partial
import matplotlib.pyplot as plt
from itertools import product as itx

from pathos.multiprocessing import _ProcessPool as Pool
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


def to_minimise(data_set, alpha, alg, ongoing=[]):
    print(f"alpha: {alpha}")
    temp = partial(algs.post_main, alpha=alpha, alg=alg)
    with Pool() as p:
        scores = p.map(temp, [data for data in data_set], chunksize=1)
    scores = np.array(scores)
    ongoing.append(([alpha] + list(scores)))
    # print(len(scores[:, -1]))
    print(f"alpha score: {np.mean(scores[:,-1])}")
    return np.mean(scores[:, -1])


def callback_minimise(*args):
    print(f"CALLBACK: {args}")
    return False


def main(*, output=0, alpha=[]):
    alg = (
        # "dumb"  # ?: base,
        # "rod"  # ?: region_of_disagreement,
        # "broad" #?: broad_base,
        # "mine" #?: rod_hotspots,
        # "greedy"  # ?: greedy,
        # "rg"  # ?: rod_greed,
        "cluster"  # ?: clusterise,
    )
    minimise = 1
    ppprint = partial(custom_print, output)
    ppprint(output)
    data_location = os.path.join(proj_path, "data", "big", "qsar_data")

    data_names = os.listdir(data_location)
    random.seed(1)
    random.shuffle(data_names)
    datasets = np.array([os.path.join(data_location, data) for data in data_names])

    dataset_lens = np.zeros([len(datasets)])
    for i, data in enumerate(datasets):
        with open(data, "r") as f:
            dataset_lens[i] = len(f.readlines())
    print(len(datasets))
    datasets = datasets[dataset_lens > 1000]

    split = [int(len(datasets) * 0.8), int(len(datasets) * 0.8)]
    data_train = datasets[: split[0]]
    data_valid = datasets[split[0] : split[1]]
    data_test = datasets[split[1] :]

    ppprint(f"{len(data_train)}, {len(data_valid)}, {len(data_test)}")
    alpha = []
    # a0 = []
    # todo - Minimise alpha
    # a0: list = [0.85, 0, 0]
    # a_boundary = [(0.5, 1), (-4, 4), (-4, 4)]
    a0: list = [0]
    a_boundary = [(0, 150)]
    if a0:
        if minimise == 1:
            arrays = [np.linspace(_a[0], _a[1], 11) for _a in a_boundary]
            # arrays = [
            #     list(range(0, 115, 10))
            #     + list(range(115, 125))
            #     + list(range(130, 151, 10))
            # ]
            if len(arrays) > 1:
                grid = np.array(itx(arrays))
            else:
                grid = arrays[0]
            score = []
            keeping_track = []
            for g in grid:
                score.append(to_minimise(data_train, g, alg, keeping_track))

            alpha = grid[np.argsort(score)[0]]
            keeping_track_pd = pd.DataFrame(data=keeping_track)
            rosette = f"I_GIVE_UP_{alg}1.csv"
            ppprint(f"SCOREEEEEEEEEEEE: {score}")
            ppprint(f"ALPHAAAAAAAAAAAA: {alpha}")
            print(rosette)
            os.makedirs(rosette, exist_ok=True)
            keeping_track_pd.to_csv(rosette)

        elif minimise == 2:
            alef = opt.minimize(
                lambda a: to_minimise(data_train, a, alg),
                a0,
                bounds=a_boundary,
                # options={"maxiter": 8},
                method="Nelder-Mead",
                callback=callback_minimise,
            )
            alpha = alef["x"]
            print(alpha)

    with_alpha = partial(algs.post_main, alpha=alpha, alg=alg)
    with Pool() as p:
        scores = p.map(with_alpha, [data for data in data_test], chunksize=1)

    results = pd.DataFrame(data=scores, index=data_test)
    # ppprint(results)
    # plt.ion()
    # plt.plot([1, 101, 201, 301, 401], np.transpose(scores))
    # plt.show(block=True)
    rosette = f"data/4/{alg}1.csv"
    print(rosette)
    results.to_csv(rosette)


if __name__ == "__main__":
    main(output=1)
