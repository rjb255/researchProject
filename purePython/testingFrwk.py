# region Libraries
import logging
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

project_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)



# sys.path.insert(1, project_path)
# from purePython.modules.shared.custom import split, getPI, Models

# endregion
# *
# ?
# !
# //
# todo


def to_minimise(data_set, alpha, alg, ongoing=[]):
    print(f"alpha: {alpha}")
    temp = partial(algs.post_main, alpha=alpha, alg=alg)
    with Pool() as p:
        scores = p.map(temp, data_set, chunksize=1)
    scores = np.array(scores)
    ongoing.append(([alpha] + [score[-1] for score in scores]))
    print(f"alpha score: {np.mean(scores[:,-1])}")
    return np.mean(scores[:, -1])



def to_minimise2(data_set, alphas, alg, ongoing=[]):
    # print(f"alpha: {alphas}")
    temp = partial(algs.post_main, alg=alg)
    passthrough = [[(data, alpha) for data in data_set] for alpha in alphas]
    b = []
    for a in passthrough:
        b += a
    with Pool() as p:
        scores = p.starmap(temp, b)
    means = []
    for i in range(len(alphas)):
        score = np.array(scores[i * len(data_set) : (i + 1) * len(data_set)])
        ongoing.append([alphas[i]] + [s[-1] for s in score])
        means.append(np.mean(score[:, -1]))

    print(f"alpha score: {means}")
    return means


def callback_minimise(*args):
    print(f"CALLBACK: {args}")
    return False


def main(*, alpha=[]):
    
    algorithm_to_test = (
        # "dumb"  # ?: base,
        # "rod"  # ?: region_of_disagreement,
        # "broad" #?: broad_base,
        # "mine"  # ?: rod_hotspots,
        # "greedy"  # ?: greedy,
        # "rg"  # ?: rod_greed,
        # "clusterI"  # ?: clusteriseI,
        # "clusterII"  # ?: clusteriseII,
        # "clusterIII"  # ?: clusteriseIII,
        "holyGrail"  # ?: holy grail,
    )

    logging.basicConfig(filename='logging/testing_frwk.log', encoding='utf-8', level=logging.DEBUG)

    minimise = 0
    
    data_location = os.path.join(project_path, "data", "big", "qsar_with_lims_2")
    data_names = os.listdir(data_location)

    #* Uncomment for reproducibility
    # random.seed(1)
    
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

    logging.info(f"{len(data_train)}, {len(data_valid)}, {len(data_test)}")
    alpha = []
    # a0 = []
    # todo - Minimise alpha
    # a0: list = [0.85, 0, 0]
    # a_boundary = [(0.5, 1), (-4, 4), (-4, 4)]
    a0: list = [60, 0.47, 0.22]
    alpha = list(a0)
    a_boundary = [(0, 150)]
    if a0:
        if minimise == 1:
            # arrays = [[*np.arange(0, 1.1, 0.5)]]
            arrays = [
                np.arange(60, 61, 2),
                np.arange(0.47, 0.48, 0.02),
                np.arange(0.22, 0.23, 0.02),
            ]
            # arrays = [[*np.arange(0, 151, 15)]]
            # arrays = [[*np.arange(0, 0.5, 0.01), *np.arange(0.5, 1.01, 0.05)]]
            # arrays = [np.linspace(_a[0], _a[1], 11) for _a in a_boundary]
            # arrays = [
            #     list(range(0, 115, 10))
            #     + list(range(115, 125))
            #     + list(range(130, 151, 10))
            # ]
            if len(arrays) > 1:
                grid = np.array(list(itx(*arrays)))
            else:
                grid = arrays[0]

            keeping_track = []
            score = to_minimise2(data_train, grid, algorithm_to_test, keeping_track)

            alpha = grid[np.argsort(score)[0]]
            keeping_track_pd = pd.DataFrame(data=keeping_track)
            rosette = f"data/param/{algorithm_to_test}1.3.csv"
            logging.info(f"SCOREEEEEEEEEEEE: {score}")
            logging.info(f"ALPHAAAAAAAAAAAA: {alpha}")
            print(rosette)
            # os.makedirs(rosette, exist_ok=True)
            keeping_track_pd.to_csv(rosette, index=False)

        elif minimise == 2:
            alef = opt.minimize(
                lambda a: to_minimise(data_train, a, algorithm_to_test),
                a0,
                bounds=a_boundary,
                # options={"maxiter": 8},
                method="Nelder-Mead",
                callback=callback_minimise,
            )
            alpha = alef["x"]
            print(alpha)

    with_alpha = partial(algs.post_main, alpha=alpha, alg=algorithm_to_test)
    with Pool() as p:
        scores = p.map(with_alpha, data_test, chunksize=1)

    results = pd.DataFrame(data=scores, index=data_test)
    # logging.info(results)
    # plt.ion()
    # plt.plot([1, 101, 201, 301, 401], np.transpose(scores))
    # plt.show(block=True)
    rosette = f"data/5/{algorithm_to_test}M.csv"
    print(rosette)
    results.to_csv(rosette)


if __name__ == "__main__":
    main()
