import pandas as pd
import numpy as np
from pathos.multiprocessing import _ProcessPool as Pool
import os
import sys
import copy
from sklearn.linear_model import BayesianRidge as BR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import Birch as BIRCH
from sklearn.cluster import KMeans as KM
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor as SGD

proj_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)

sys.path.insert(1, proj_path)

from purePython.modules.shared.custom import split, Models
from purePython.v4main import score


def main(dataset):
    data = pd.read_csv(dataset[0])
    backup = copy.deepcopy(data)
    models = {
        "BayesianRidge": BR(),
        "KNN": KNN(),
        "RandomForrest": RFR(random_state=1),
        "SGD": SGD(loss="huber", random_state=1),
        "SVM": SVR(),
        "ABR": ABR(random_state=1),
    }
    m = Models(list(models.values()))
    data: pd.DataFrame = data.sample(frac=1, random_state=1)
    X_known, Y_known, X_unknown, Y_unknown, X_test, Y_test = split(data, 5, frac=1)
    X_test, Y_test = pd.concat([X_known, X_unknown]), pd.concat([Y_known, Y_unknown])

    m.fit(X_known, Y_known)
    _m = copy.deepcopy(m)
    s = [score(Y_test, model=_m, X_test=X_test)]

    m.fit(X_test, Y_test)
    _m = copy.deepcopy(m)
    s.append(score(Y_test, model=_m, X_test=X_test))

    backup["llim"] = s[0]
    backup["ulim"] = s[1]
    backup.to_csv(dataset[1])
    print("Survived")


data_location = os.path.join(proj_path, "data", "big", "qsar_data")
data_names = os.listdir(data_location)
datasets = np.array([os.path.join(data_location, data) for data in data_names])
data_location2 = os.path.join(proj_path, "data", "big", "qsar_with_lims")
data2 = np.array([os.path.join(data_location2, data) for data in data_names])


with Pool() as p:
    dataframes = p.map(main, [(a, b) for a, b in zip(datasets, data2)])
pass
