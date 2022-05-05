# region libraries
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
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR

proj_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)
sys.path.insert(1, proj_path)

from purePython.modules.shared.custom import split, getPI, Models

# endregion


def spec_max(x):
    if len(x) > 0:
        return max(x)
    else:
        return x


def score(Y_test, kwargs, q):
    y_predict = kwargs["model"].predict(kwargs["X_test"])
    q.put(mse(y_predict, Y_test))  # , sample_weight=Y_test))


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
            print(len(X))
        return [[score.get() for score in score_record], X]

    with Pool() as p:
        results = p.map(first_split, algorithm)

    _file = os.path.join(proj_path, "purePython", "data", input("fileName ") + ".csv")

    os.makedirs(os.path.dirname(_file), exist_ok=True)

    print(f"\nwriting {_file}")
    pd_results = pd.DataFrame(
        [r[0] for r in results],
        columns=list(
            range(len(X_train), len(X_train) + iterations * sample_size, sample_size)
        ),
        index=[a.__name__ for a in algorithm],
    )

    pd_results.to_csv(_file)
    _file = os.path.join(proj_path, "purePython", "data", input("fileName ") + ".csv")

    os.makedirs(os.path.dirname(_file), exist_ok=True)

    print(f"\nwriting {_file}")

    results[0][1].to_csv(_file)


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


def rod_hotspots(m, X, Y, x, *args, **kwargs):
    Y, Y_error = m.predict_error(x)
    err = -Y_error
    series = np.ones_like(err)
    tree = KDTree(x.values)
    for r in range(2, 9):
        indicies = tree.query_ball_point(np.array(x), r, workers=-1)
        sdev = [err[i] for i in indicies]
        sdevMax = [spec_max(s) for s in sdev]
        series += sdevMax == err
        print(f"{r}", end="\r")
    return series * err * Y


def clusterise():
    pass


def density(x1, x2, mem):
    if not mem:
        pass  ## Use one tree in future
    tree = np.array(dist_mat(x1, x2))
    tree[tree == 0] = np.min(tree[np.nonzero(tree)])
    return np.sum(1 / tree, axis=0)


def main():
    data_sets: List[pd.DataFrame] = [
        pd.read_csv(os.path.join(proj_path, "purePython", "data.csv"))
    ]

    set_num = 0
    data: List[pd.DataFrame] = data_sets[set_num].sample(frac=1, random_state=1)

    X_known, Y_known, X_unknown, Y_unknown, _, _ = split(data, 5, frac=1)
    X_test, Y_test = pd.concat([X_known, X_unknown]), pd.concat([Y_known, Y_unknown])
    models = {
        "BayesianRidge": BR(),
        "KNN": KNN(n_jobs=-1),
        "RandomForrest": RFR(random_state=1),
        "SGD": SGD(loss="huber", random_state=0),
        "SVM": SVR(),
    }
    algorithms = {
        "dumb": base,
        "uncertainty_sampling": uncertainty_sampling,
        "rod": region_of_disagreement,
        "broad": broad_base,
        "mine": rod_hotspots,
    }

    # For when this isn't the only one: makes keeping track easier
    model = Models(list(models.values()))

    algorithm = (
        # algorithms["mine"],
        # algorithms["uncertainty_sampling"],
        algorithms["rod"],
        # algorithms["dumb"],
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
        iterations=50 + 1,
        sample_size=1,
        score=score,
    )


if __name__ == "__main__":
    main()
