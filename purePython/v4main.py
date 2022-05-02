# region libraries
import sys
import os
import math

from pprint import pprint
from typing import List

from functools import partial
import copy

from multiprocessing import Queue, Process

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as mse
from scipy.spatial import distance_matrix as dist_mat
from scipy.spatial import KDTree
from sklearn.linear_model import BayesianRidge as BR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.mixture import GaussianMixture as GMM

proj_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)
sys.path.insert(1, proj_path)

from purePython.modules.shared.custom import split, getPI, Models

# endregion


def score(Y_test, kwargs, q):
    y_predict = kwargs["model"].predict(kwargs["X_test"])
    q.put(mse(y_predict, Y_test, sample_weight=Y_test))


def spec_max(x):
    if len(x) > 0:
        return max(x)
    else:
        return x


def framework(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_unknown: pd.DataFrame,
    Y_unknown: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    model: object,
    algorithm: tuple,
    iterations: int,
    sample_size: int,
    score: object,
    alpha: list =[],
):
    inner_func = partial(
        first_split,
        X_train,
        Y_train,
        X_unknown,
        Y_unknown,
        model,
        iterations,
        sample_size,
        Y_test,
        X_test,
        alpha=alpha
    )
    results = [inner_func(alg) for alg in algorithm]

    pprint(f'{alpha}: {results}')
    return results


def first_split(
    X_train,
    Y_train,
    X_unknown,
    Y_unknown,
    model,
    iterations,
    sample_size,
    Y_test,
    X_test,
    f,
    alpha=[],
):
    # Deep copies
    X, Y = pd.DataFrame(X_train), pd.Series(Y_train)
    x, y = pd.DataFrame(X_unknown), pd.Series(Y_unknown)
    m = copy.deepcopy(model)
    mem = {}  # If sth is stored between executions
    score_record = []
    processes = []
    mem['alpha'] = alpha
    for i in range(iterations):
        if len(x) > 0:
            m.fit(X, Y)
            ranking = f(m, X, Y, x, mem)
            next_index = x.index[np.argsort(ranking)]
            X, Y, x, y = getPI((X, Y), (x, y), next_index[:sample_size])
        score_record.append(Queue())
        processes.append(
            Process(
                target=score,
                args=(
                    Y_test,
                    {
                        "model": copy.deepcopy(m),
                        "X_test": X_test,  # No need for deepcopy (no change to X_test)
                    },
                    score_record[-1],
                ),
            )
        )
        processes[-1].start()
        # print(f"{i} with {f.__name__}")
    return [score.get() for score in score_record]


def base(m, X, Y, x, *args, **kwargs):
    next_index = x.index
    return next_index


def uncertainty_sampling(m, X, Y, x, *args, **kwargs):
    _, Y_error = m.predict(x, return_std=True)
    return -Y_error


def region_of_disagreement(m, X, Y, x, *args, **kwargs):
    _, Y_error = m.predict_error(x)
    return -Y_error


def broad_base(m, X, Y, x, mem, *args, **kwargs):
    rho = density(X, x, mem)
    return rho


def rod_hotspots(m, X, Y, x, mem, *args, **kwargs):
    
    Y_predict, Y_error = m.predict_error(x)
    err = -Y_error

    # todo CLUSTER ALGORITHM FIRST
    # todo INCLUDE Y_predict and Y_std, consider a restriction on std error
    temp1 = pd.Series(Y_predict)
    temp2 = pd.Series(err)
    cluster_x = copy.deepcopy(pd.DataFrame(x))
    
    cluster_x['err'] = err
    cluster_x['y'] = Y_predict

    lower_lvl = np.quantile(cluster_x["err"], mem['alpha'][0])
    index = np.ones_like(err)
    index[cluster_x['err'] < lower_lvl] = 0

    cluster_x = cluster_x[index == 1]
    if "cluster" in mem:
        mem["cluster"].fit(cluster_x)
    else:
        mem["cluster"] = GMM(min(50, max(5, len(cluster_x)-10)), random_state=1, warm_start=True).fit(cluster_x)

    # if "tree" in mem:
    #     pass
    # else:
    #     mem["tree"] = KDTree(cluster_x)

    # # todo WORK WITH CLUSTERS

    # alias_points = mem["tree"].query(mem["cluster"])

    # todo return std^alpha*y_predict^beta for alias_points
    score = index
    try:
        err[err < 0] = 0
        Y_predict[Y_predict < 0] = 0
        score[index == 1] = (np.power(err[index == 1], mem['alpha'][1]) *
                             np.power(Y_predict[index == 1], mem['alpha'][2]) *
                             np.array(mem["cluster"].score_samples(cluster_x)))
    except e:
        print(err[index == 1])
        print(f"{mem['alpha'][1]}, {mem['alpha'][2]}")
        print(np.power(err[index == 1], mem['alpha'][1]))
        score[index == 1] = err[index == 1] * 10
        print(e)
    return score


def clusterise():
    pass


def density(x1, x2, mem):
    if not mem:
        pass  ## Use one tree in future
    tree = np.array(dist_mat(x1, x2))
    tree[tree == 0] = np.min(tree[np.nonzero(tree)])
    return np.sum(1 / tree, axis=0)


def start(
    data_path: str,
):
    pass


def main():
    paths = ["data_CHEMBL313.csv", "data_CHEMBL2637.csv", "data_CHEMBL4124.csv"]
    
    set_num = 0
    start(paths[set_num])
    #
    data_sets: List[str] = [
        os.path.join(proj_path, "data", "chembl", "Additional_datasets", path)
        for path in paths
    ]
    data = data_sets[set_num]
    post_main(data)


def post_main(dataset, alpha=[]):
    data = pd.read_csv(dataset)
    data: List[pd.DataFrame] = data.sample(frac=1, random_state=1)
    print(len(data))

    X_known, Y_known, X_unknown, Y_unknown, _, _ = split(data, 5, frac=1)
    X_test, Y_test = pd.concat([X_known, X_unknown]), pd.concat([Y_known, Y_unknown])
    models = {
        "BayesianRidge": BR(),
        "KNN": KNN(),
        "RandomForrest": RFR(random_state=1),
    }
    algorithms = {
        "dumb": base,
        "uncertainty_sampling": uncertainty_sampling,
        "rod": region_of_disagreement,
        "broad": broad_base,
        "mine": rod_hotspots,
    }

    # For when this isn't the only one: makes keeping track easier
    model = Models([models["BayesianRidge"], models["KNN"], models["RandomForrest"]])

    algorithm = (
        # algorithms["mine"],
        # algorithms["rod"],
        algorithms["dumb"],
    )

    return framework(
        X_known,
        Y_known,
        X_unknown,
        Y_unknown,
        X_test,
        Y_test,
        model,
        algorithm,
        iterations=4,
        sample_size=120,
        score=score,
        alpha=alpha
    )

if __name__ == "__main__":
    main()
