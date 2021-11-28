import pandas as pd
from pandas.core.frame import DataFrame

def split(data: DataFrame, n: int=20):
    ## Splits data (dataframe) insto a known section, an unknown section, and a testing section
    
    splitBoundary = [n, int(0.9*len(data))]
    X_known = data.iloc[:splitBoundary[0], 2:]
    Y_known = data.iloc[:splitBoundary[0], 1]
    X_unknown = data.iloc[splitBoundary[0]:splitBoundary[1], 2:]
    Y_unknown = data.iloc[splitBoundary[0]:splitBoundary[1], 1]
    X_test = data.iloc[splitBoundary[1]:, 2:]
    Y_test = data.iloc[splitBoundary[1]:, 1]
    return X_known, Y_known, X_unknown, Y_unknown, X_test, Y_test

def getPI(known: tuple, unknown: tuple, index: int):
    X_known, Y_known = known
    X_unknown, Y_unknown = unknown
    X_known = X_known.append(X_unknown.loc[index])
    Y_known = Y_known.append(Y_unknown.loc[index])
    X_unknown = X_unknown.drop(index)
    Y_unknown = Y_unknown.drop(index)

    return X_known, Y_known, X_unknown, Y_unknown

def bubbles(index: tuple, indicies: list, data: DataFrame):
    index1, index2 = index
    
