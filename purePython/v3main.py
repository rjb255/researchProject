# region
import sys
import os

from pprint import pprint
from typing import List

from functools import partial
import copy

from pathos.multiprocessing import ProcessPool as Pool
from multiprocessing import Queue, Process

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as mse
from scipy.spatial import distance_matrix as dist_mat
from scipy.spatial import KDTree
from sklearn.linear_model import BayesianRidge as BR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import RandomForestRegressor as RFR

proj_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)
sys.path.insert(1, proj_path)

from purePython.modules.shared.custom import split, getPI, Models

# endregion


def score(Y_test, kwargs, q):
    y_predict = kwargs["model"].predict(kwargs["X_test"])
    q.put(mse(y_predict, Y_test))


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
    def first_split(f):
        # Deep copies
        X, Y = pd.DataFrame(X_train), pd.Series(Y_train)
        x, y = pd.DataFrame(X_unknown), pd.Series(Y_unknown)
        m = copy.deepcopy(model)
        mem = {}  # If sth is stored between executions
        score_record = []
        processes = []
        for i in range(iterations):
            if len(x) > 0:
                next_index = f(m, X, Y, x, mem)
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

    with Pool() as p:
        results = p.map(first_split, algorithm)

    pprint(results)
    _file = os.path.join(proj_path, "purePython", "data", input("fileName ") + ".csv")

    os.makedirs(os.path.dirname(_file), exist_ok=True)

    with open(_file, "w") as f:
        print(f"\nwriting {_file}")
        f.write(str(results))


def base(m, X, Y, x, *args, **kwargs):
    m.fit(X, Y)
    next_index = x.index
    return next_index


def uncertainty_sampling(m, X, Y, x, *args, **kwargs):
    m.fit(X, Y)
    _, Y_error = m.predict(x, return_std=True)
    next_index = x.index[np.argsort(-Y_error)]
    return next_index


def broad_base(m, X, Y, x, mem, *args, **kwargs):
    rho = density(X, x, mem)
    m.fit(X, Y)
    next_index = x.index[np.argsort(rho)]
    return next_index


def rod_hotspots(m, X, Y, *args, **kwargs):
    r = 5
    m.fit(X, Y)
    pred, err = m.predict_error(x)
    tree = KDTree(X.values)
    indicies = tree.query_ball_point(np.array(x), r, workers=-1)
    sdev = [err[i] for i in indicies]


def hotspots():
    pass


def density(x1, x2, mem):
    if not mem:
        pass  ## Use one tree in future
    tree = np.array(dist_mat(x1, x2))
    tree[tree == 0] = np.min(tree[np.nonzero(tree)])
    return np.sum(1 / tree, axis=0)


def main():
    paths = ["data_CHEMBL313.csv", "data_CHEMBL2637.csv", "data_CHEMBL4124.csv"]
    data_sets: List[pd.DataFrame] = [
        pd.read_csv(
            os.path.join(proj_path, "data", "chembl", "Additional_datasets", path)
        )
        for path in paths
    ]

    set_num = 2
    data: List[pd.DataFrame] = data_sets[set_num].sample(frac=1, random_state=1)

    X_known, Y_known, X_unknown, Y_unknown, _, _ = split(data, 1, frac=1)
    X_test, Y_test = pd.concat([X_known, X_unknown]), pd.concat([Y_known, Y_unknown])
    models = {
        "BayesianRidge": BR(),
        "KKK": KNN(n_jobs=-1),
        "RandomForrest": RFR(random_state=1),
    }
    algorithms = {
        "dumb": base,
        "uncertainty_sampling": uncertainty_sampling,
        "broad": broad_base,
    }

    # For when this isn't the only one: makes keeping track easier
    model = Models([models["BayesianRidge"]])
    algorithm = (
        algorithms["uncertainty_sampling"],
        algorithms["dumb"],
    )
    framework(
        X_known,
        Y_known,
        X_unknown,
        Y_unknown,
        X_test,
        Y_test,
        model,
        algorithm,
        iterations=10,
        sample_size=160,
        score=score,
    )


if __name__ == "__main__":
    main()
