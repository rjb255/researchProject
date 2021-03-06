# Note, ##% refers to cell debugging in vscode
#%% Libraries
# # Standard Libraries
from os import path
from pathlib import Path
from pprint import pprint

# External Libraries
from IPython.display import display
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.linear_model import BayesianRidge as BR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import RandomForestRegressor as RFR
import statistics as stats
import matplotlib.pylab as mlt

# Custom Libraries
from modules.shared.custom import split, getPI
#%%

def main():
#%% Datasets
    # Just getting datasets
    _path = Path(__file__).parent.resolve()
    dataRoute = path.normpath(path.join(_path, '..', 'data', 'chembl', 'Additional_datasets'))
    dataSets = [pd.read_csv(path.join(dataRoute, 'data_CHEMBL313.csv')),
                pd.read_csv(path.join(dataRoute, 'data_CHEMBL2637.csv')),
                pd.read_csv(path.join(dataRoute, 'data_CHEMBL4124.csv'))]
    data = dataSets[0].sample(frac=1)

    # Splitting up the data
    X_known, Y_known, X_unknown, Y_unknown, X_test, Y_test = split(data, 160)

    # Each testing stage can test 160 cases and up to 2000 samples will be tested
    testSize = 160
    maxSamples = len(Y_unknown)
    display(data)
#%%
    # models to be used and pre-defining variables
    models = [BR(), KNN(), RFR()]
    predicitions: list
    scores: list
    smartScores = []
    tree = []

    # Each iteration represents another testing round
    for i in range(0, maxSamples, testSize):
        tree = KDTree(np.array(X_unknown))
        predictions = []
        scores = []
        for model in models:
            m = model.fit(X_known, Y_known)
            predictions.append(m.predict(X_unknown))
            scores.append(m.score(X_test, Y_test))

        np_predictions = np.array(predictions)
        stdd = np.std(np_predictions, axis=0, ddof=1)
        maxima = []


        # Optimise Here To Remove Searches. Allows to pin down faster
        # Look for points of maxima in an r=5
        for j, x, s in zip(X_unknown.index, np.array(X_unknown), stdd):
            neighbours = tree.query_ball_point(x, 5)
            if np.max(stdd[neighbours]) <= s:
                maxima.append(j)
            
        if len(maxima) <= testSize:
            semiMaxima = np.argpartition(stdd, -testSize)
            semiMaxima2 = X_unknown.index[semiMaxima[-testSize:]]
            semiMaxima3 = np.argsort(stdd[semiMaxima[-testSize:]])
            semiMaxima4 = semiMaxima2[semiMaxima3]
            semiMaxima5 = [e1 for e1 in semiMaxima4 if e1 not in maxima]
            
            index2 = maxima + semiMaxima5[len(maxima)-testSize:]
        else:
            index = np.argsort(-stdd[maxima])
            index2 = X_unknown.iloc[index].index
        X_known, Y_known, X_unknown, Y_unknown = getPI((X_known, Y_known), (X_unknown, Y_unknown), index2[:testSize])   
        print(f"\r{len(X_unknown)} left. {len(X_known)} known.", end="")

        
        # index = np.argsort(-1*stdd)
        # 
        # print(f"Samples: { len(Y_known) }, scores: {scores}")
        # smartScores.append(scores)


#%%
if __name__ == "__main__":
#%% Run everything
    main()
    
        
        


        

