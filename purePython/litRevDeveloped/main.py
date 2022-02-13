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
from sklearn.linear_model import BayesianRidge as BR

proj_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)
sys.path.insert(1, proj_path)

from purePython.modules.shared.custom import split, getPI, Models


def loopDecorator(iterations, size):
    def decorator(func):
        def inner(X_train, Y_train, X_unknown, Y_unknown, model, test, kwargs):
            def doubleInner(f):
                ## Deep copy to maintain thread safety
                X, Y = pd.DataFrame(X_train), pd.Series(Y_train)
                x, y = pd.DataFrame(X_unknown), pd.Series(Y_unknown)
                m = copy.deepcopy(model)

                score_record = []
                processes = []
                for i in range(iterations):
                    next_index = f(m, X, Y, x, y)
                    X, Y, x, y = getPI((X, Y), (x, y), next_index[:size])
                    score_record.append(Queue())
                    processes.append(
                        Process(
                            target=test,
                            args=(
                                {
                                    "model": copy.deepcopy(m),
                                    "X_test": (
                                        kwargs["X_test"]
                                    ),  ## No need for deepcopy (no change to X_test)
                                },
                                score_record[-1],
                            ),
                        )
                    )
                    processes[-1].start()
                    print(f"{i} with {f.__name__}")
                return [score.get() for score in score_record]

            with Pool() as p:
                results = list(p.map(doubleInner, (func, base)))
            pprint(results)
            _file = os.path.join(
                proj_path,
                "purePython",
                "litRevDeveloped",
                str(type(model).__name__),
                str(func.__name__),
                str(iterations),
                str(size),
                str(len(X_train)),
                f"{kwargs['data_set']}.csv",
            )

            os.makedirs(os.path.dirname(_file), exist_ok=True)

            with open(_file, "w") as f:
                print(f"\nwriting {_file}")
                f.write(str(results))

        return inner

    return decorator


def base(model, X_train, Y_train, X_unknown, Y_unknown):
    model.fit(X_train, Y_train)
    next_index = X_unknown.index
    return next_index


def validate(Y_test, kwargs, q):
    y_predict = kwargs["model"].predict(kwargs["X_test"])
    q.put(mse(y_predict, Y_test))


@loopDecorator(300, 1)
def uncertainty_sampling(model, X_train, Y_train, X_unknown, Y_unknown):
    model.fit(X_train, Y_train)
    _, Y_error = model.predict(X_unknown, return_std=True)
    next_index = X_unknown.index[np.argsort(-Y_error)]
    return next_index


@loopDecorator(50, 1)
def broad_base(model, X_train, Y_train, X_unknown, Y_unknown):
    rho = density(X_train, X_unknown)
    model.fit(X_train, Y_train)
    next_index = X_unknown.index[np.argsort(rho)]
    return next_index


def density(x1, x2):
    tree = np.array(dist_mat(x1, x2))
    tree[tree == 0] = np.min(tree[np.nonzero(tree)])
    return np.sum(1 / tree, axis=0)


def main(*, set_num=0, model, sampling_method):
    paths = ["data_CHEMBL313.csv", "data_CHEMBL2637.csv", "data_CHEMBL4124.csv"]
    data_sets: List[pd.DataFrame] = [
        pd.read_csv(
            os.path.join(proj_path, "data", "chembl", "Additional_datasets", path)
        )
        for path in paths
    ]
    data: List[pd.DataFrame] = data_sets[set_num].sample(frac=1, random_state=1)
    X_known, Y_known, X_unknown, Y_unknown, X_test, Y_test = split(data, 1, frac=1)
    t = partial(validate, data.iloc[:, 1])
    models = {"BayesianRidge": BR()}

    defaults = (
        X_known,
        Y_known,
        X_unknown,
        Y_unknown,
        models[model],
        t,
        {
            "X_test": data.iloc[:, 2:],
            "Y_test": data.iloc[:, 1],
            "data_set": set_num,
        },
    )
    sampling_methods = {
        "uncertainty_sampling": lambda: uncertainty_sampling(*defaults),
        "broad_base": lambda: broad_base(*defaults),
    }
    sampling_methods[sampling_method]()


if __name__ == "__main__":
    data_set_num = 2

    main(
        set_num=data_set_num,
        model="BayesianRidge",
        sampling_method="broad_base",
    )
