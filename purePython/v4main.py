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
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import Birch as BIRCH
from sklearn.cluster import KMeans as KM
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.neural_network import MLPRegressor as NN

from modules.shared.custom import split, getPI, Models


proj_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)
sys.path.insert(1, proj_path)



# endregion


def score(Y_test, lims=[0, 1], **kwargs):
    y_predict = kwargs["model"].predict(kwargs["X_test"])
    weight = np.array(Y_test) - np.min(Y_test)
    if np.max(weight) == 0:
        print("Issue")
        weight = weight + 1
    else:
        weight /= np.max(weight)
    if (weight < 0).any():
        print(f"ERROR: {weight}")
    s = mse(y_predict, Y_test, sample_weight=weight)
    s = (s - lims[0]) / (lims[1] - lims[0])
    return s


def spec_max(x):
    if len(x) > 0:
        return max(x)
    else:
        return x


def riffle(a):
    lens = np.array([len(_a) for _a in a])
    eq1 = []
    eq2 = []
    order = np.argsort(-lens)
    for order_ in order:
        eq1 += list(a[order_])
        eq2 += list(np.arange(lens[order_]) / lens[order_])
    order = np.argsort(eq2, kind="mergesort")

    return np.array(eq1)[order]


def framework(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_unknown: pd.DataFrame,
    Y_unknown: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    model: object,
    algorithm,
    iterations: int,
    sample_size: int,
    score: object,
    alpha: list = [],
    lims=[0, 1],
):
    results = first_split(
        X_train,
        Y_train,
        X_unknown,
        Y_unknown,
        model,
        iterations,
        sample_size,
        Y_test,
        X_test,
        algorithm,
        alpha=alpha,
        lims=lims,
    )

    pprint(f"{alpha}: {results}")
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
    lims=[0, 1],
):
    # Deep copies
    X, Y = pd.DataFrame(X_train), pd.Series(Y_train)
    x, y = pd.DataFrame(X_unknown), pd.Series(Y_unknown)
    m = copy.deepcopy(model)
    mem = {}  # If sth is stored between executions
    score_record = []
    mem["alpha"] = alpha
    for i in range(iterations):
        if len(x) > 0:
            m.fit(X, Y)
            _m = copy.deepcopy(m)
            ranking = f(m, X, Y, x, mem)
            next_index = x.index[np.argsort(ranking)]
            _t = [len(X), len(x)]
            X, Y, x, y = getPI((X, Y), (x, y), next_index[:sample_size])
            _a = np.array([_t[0] - len(X) + sample_size, _t[1] - len(x) - sample_size])
            if np.count_nonzero(_a) > 0:
                print(f"ERROR: {_a}")
        score_record.append(score(Y_test, model=_m, X_test=X_test, lims=lims))

        print(f"{i} with {f.__name__}")
    return score_record


# TODO:


def rod_greed(m, X, Y, x, mem, *args, **kwargs):
    f = greedy(m, X, Y, x)
    g = region_of_disagreement(m, X, Y, x)
    return np.power(0.1 + f - np.min(f), mem["alpha"]) * np.power(
        0.1 + g - np.min(g), 1 - mem["alpha"]
    )


def base(m, X, Y, x, *args, **kwargs):
    return np.ones_like(x.index)


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

    cluster_x["err"] = err
    cluster_x["y"] = Y_predict

    lower_lvl = np.quantile(cluster_x["err"], mem["alpha"][0])
    index = np.ones_like(err)
    index[cluster_x["err"] < lower_lvl] = 0

    cluster_x = cluster_x[index == 1]
    if "cluster" in mem:
        mem["cluster"].fit(cluster_x)
    else:
        mem["cluster"] = GMM(
            min(50, max(5, len(cluster_x) - 10)), random_state=1, warm_start=True
        ).fit(cluster_x)

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
        score[index == 1] = (
            np.power(err[index == 1], mem["alpha"][1])
            * np.power(Y_predict[index == 1], mem["alpha"][2])
            * np.array(mem["cluster"].score_samples(cluster_x))
        )
    except e:
        print(err[index == 1])
        print(f"{mem['alpha'][1]}, {mem['alpha'][2]}")
        print(np.power(err[index == 1], mem["alpha"][1]))
        score[index == 1] = err[index == 1] * 10
        print(e)
    return score


def clusterI(m, X, Y, x, mem, *args, **kwargs):
    return clusterise(m, X, Y, x, mem, *args, **kwargs)


def clusterII(m, X, Y, x, mem, *args, **kwargs):
    _X = copy.deepcopy(X)
    _X["Y"] = Y
    y = m.predict(x)
    _x = copy.deepcopy(x)
    _x["Y"] = y
    return clusterise(m, _X, Y, _x, mem, *args, **kwargs)


def holyGrail(m, X, Y, x, mem, *args, **kwargs):
    alpha = mem["alpha"]
    p1 = clusterIII(m, X, Y, x, {"alpha": alpha[0]}, *args, **kwargs)
    p2 = rod_greed(m, X, Y, x, {"alpha": alpha[1]}, *args, **kwargs)
    p3 = np.power(p1, alpha[2])
    p4 = np.power(p2, 1 - alpha[2])
    return p3 * p4


def clusterIII(m, X, Y, x, mem, *args, **kwargs):
    _X = copy.deepcopy(X)
    _X["Y"] = Y
    _X["err"] = 0
    y, err = m.predict_error(x)
    _x = copy.deepcopy(x)
    _x["Y"] = y
    _x["err"] = err
    return clusterise(m, _X, Y, _x, mem, *args, **kwargs)


def clusterise(m, X, Y, x, mem, *args, **kwargs):
    Xx = pd.concat([X, x])
    if "cluster" in mem:
        mem["cluster"].set_params(n_clusters=10 + len(X))
        mem["cluster"].fit(Xx)
    else:
        mem["cluster"] = KM(n_clusters=int(mem["alpha"]) + len(X), random_state=1).fit(
            Xx
        )

    labels = mem["cluster"].predict(x)
    t1 = mem["cluster"].transform(x)
    t2 = np.min(t1, axis=1)
    minDif = np.max(t2) + 1
    lab2 = labels * minDif + t2

    s1 = np.argsort(lab2)
    t2 = t1[s1]

    bounds = np.unique(labels, return_counts=True)

    prev = 0
    b2 = []
    excess = []
    for index, i in zip(bounds[0], bounds[1]):
        if index in mem["cluster"].labels_[: len(X)]:
            excess.append(s1[prev : i + prev])
        else:
            b2.append(s1[prev : i + prev])
        prev += i

    r1 = list(riffle(b2))
    r2 = list(riffle(excess))
    temp = np.arange(len(r1 + r2))
    score = np.ones_like(temp)
    score[r1 + r2] = temp
    return score


def density(x1, x2, mem):
    if not mem:
        pass  ## Use one tree in future
    tree = np.array(dist_mat(x1, x2))
    tree[tree == 0] = np.min(tree[np.nonzero(tree)])
    return np.sum(1 / tree, axis=0)


def greedy(m, X, Y, x, *args, **kwargs):
    return -m.predict(x)


# def main():
#     paths = ["data_CHEMBL313.csv", "data_CHEMBL2637.csv", "data_CHEMBL4124.csv"]

#     set_num = 0
#     start(paths[set_num])
#     #
#     data_sets: List[str] = [
#         os.path.join(proj_path, "data", "chembl", "Additional_datasets", path)
#         for path in paths
#     ]
#     data = data_sets[set_num]
#     post_main(data)


def post_main(dataset, alpha=[], alg="dumb"):
    data = pd.read_csv(dataset)
    data: pd.DataFrame = data.sample(frac=1, random_state=1)
    print(f"{len(data)}, {alpha}")

    X_known, Y_known, X_unknown, Y_unknown, X_test, Y_test, llim, ulim = split(
        data,
        5,
        frac=1,
        lims=True,
    )
    X_test, Y_test = pd.concat([X_known, X_unknown]), pd.concat([Y_known, Y_unknown])
    models = {
        "BayesianRidge": BR(),
        "KNN": KNN(),
        # "RandomForrest": RFR(random_state=1),
        # "SGD": SGD(loss="huber", random_state=1),
        # "SVM": SVR(),
        # "ABR": ABR(random_state=1),
        "NN": NN(warm_start=True, random_state=1),
    }

    algorithms = {
        "dumb": base,
        "uncertainty_sampling": uncertainty_sampling,
        "rod": region_of_disagreement,
        "broad": broad_base,
        "mine": rod_hotspots,
        "greedy": greedy,
        "rg": rod_greed,
        "clusterI": clusterI,
        "clusterII": clusterII,
        "clusterIII": clusterIII,
        "holyGrail": holyGrail,
    }

    # For when this isn't the only one: makes keeping track easier
    model = Models(list(models.values()))

    algorithm = algorithms[alg]

    return framework(
        X_known,
        Y_known,
        X_unknown,
        Y_unknown,
        X_test,
        Y_test,
        model,
        algorithm,
        iterations=6,
        sample_size=100,
        score=score,
        alpha=alpha,
        lims=[ulim, llim],
    )


if __name__ == "__main__":
    main()
