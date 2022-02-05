import sys
import os
from pprint import pprint
from typing import List
from functools import partial
from multiprocessing import Process, Queue


import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix as dist_mat
from sklearn.linear_model import BayesianRidge as BR

proj_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)
sys.path.insert(1, proj_path)

from purePython.modules.shared.custom import split, getPI, validate, Models


def loopDecorator(iterations, size):
    def decorator(func):
        def inner(model, X_train, Y_train, X_unknown, Y_unknown, X_test, test, set_num):
            def doubleInner(f, q):
                X, Y = pd.DataFrame(X_train), pd.Series(Y_train)
                x, y = pd.DataFrame(X_unknown), pd.Series(Y_unknown)

                score_record = []
                for i in range(iterations):
                    next_index, score = f(model, X, Y, x, y, X_test, test)

                    X, Y, x, y = getPI((X, Y), (x, y), next_index[:size])
                    score_record.append(score)
                q.put(score_record)

            q = [Queue(), Queue()]
            active_learning = Process(target=doubleInner, args=(func, q[0]))
            base_run = Process(target=doubleInner, args=(base, q[1]))
            active_learning.start(), base_run.start()
            final_score_AL = q[0].get()
            final_score_base = q[1].get()
            active_learning.join(), base_run.join()

            _file = os.path.join(
                proj_path,
                "purePython",
                "litRevDeveloped",
                str(type(model).__name__),
                str(func.__name__),
                str(iterations),
                str(size),
                str(len(X_train)),
                f"{set_num}.csv",
            )

            os.makedirs(os.path.dirname(_file), exist_ok=True)

            with open(_file, "w") as f:
                print("\nwriting")
                f.write(f"{final_score_AL}\n{final_score_base}")

        return inner

    return decorator


def base(model, X_train, Y_train, X_unknown, Y_unknown, X_test, test):
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_unknown)
    next_index = X_unknown.index
    score = test(model.predict(X_test))
    return next_index, score


@loopDecorator(15, 160)
def uncertainty_sampling(model, X_train, Y_train, X_unknown, Y_unknown, X_test, test):
    model.fit(X_train, Y_train)
    Y_predict, Y_error = model.predict(X_unknown, return_std=True)
    next_index = X_unknown.index[np.argsort(-Y_error)]
    score = test(model.predict(X_test))
    return next_index, score


@loopDecorator(500, 1)
def broad_base(model, X_train, Y_train, X_unknown, Y_unknown, X_test, test):
    rho = density(X_train, X_unknown)
    model.fit(X_train, Y_train)
    next_index = X_unknown.index[np.argsort(rho)]
    score = test(model.predict(X_test))
    return next_index, score


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
    X_known, Y_known, X_unknown, Y_unknown, X_test, Y_test = split(data, 1)
    test = partial(validate, Y_test)
    models = {"BayesianRidge": BR()}

    defaults = (
        models[model],
        X_known,
        Y_known,
        X_unknown,
        Y_unknown,
        X_test,
        test,
        set_num,
    )
    sampling_methods = {
        "uncertainty_sampling": lambda: uncertainty_sampling(*defaults),
        "broad_base": lambda: broad_base(*defaults),
    }
    sampling_methods[sampling_method]()


if __name__ == "__main__":
    data_set_num = 0

    main(model="BayesianRidge", sampling_method="broad_base")
