# region libraries
import sys
import os

from pprint import pprint
from typing import List

from functools import partial
import copy

from pathos.multiprocessing import ThreadPool as Pool
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
    "/", "home", "rjb255", "researchProject", "researchProject"
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
    )
    with Pool() as p:
        results = p.map(inner_func, algorithm)[0]

    pprint(results)
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
):
    # Deep copies
    X, Y = pd.DataFrame(X_train), pd.Series(Y_train)
    x, y = pd.DataFrame(X_unknown), pd.Series(Y_unknown)
    m = copy.deepcopy(model)
    mem = {}  # If sth is stored between executions
    score_record = []
    processes = []
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
        print(f"{i} with {f.__name__}")
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
    # todo REWRITE THIS PLS
    Y, Y_error = m.predict_error(x)
    err = -Y_error

    # todo CLUSTER ALGORITHM FIRST,
    # todo INCLUDE Y_predict and Y_std, consider a restriction on std error
    if "cluster" in mem:
        pass
    else:
        mem["cluster"] = GMM(n_components=200, random_state=1, warm_start=True).fit(X)

    if "tree" in mem:
        pass
    else:
        mem["tree"] = KDTree(x)

    # todo WORK WITH CLUSTERS
    alias_points = mem["tree"].query(mem["cluster"])

    # todo return std^alpha*y_predict^beta for alias_points

    return series * err * Y


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


def post_main(dataset):
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
        algorithms["mine"],
        # algorithms["rod"],
        # algorithms["dumb"],
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
        iterations=5,
        sample_size=20,
        score=score,
    )

if __name__ == "__main__":
    main()
