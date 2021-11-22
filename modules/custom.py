import pandas as pd

def split(data: pd.core.frame.DataFrame, n: int=20):
    ## Splits data (dataframe) insto a known section, an unknown section, and a testing section
    splitBoundary = [n, int(0.9*len(data))]
    X_known = data.iloc[:splitBoundary[0], 2:]
    Y_known = data.iloc[:splitBoundary[0], 1]
    X_unknown = data.iloc[splitBoundary[0]:splitBoundary[1], 2:]
    Y_unknown = data.iloc[splitBoundary[0]:splitBoundary[1], 1]
    X_test = data.iloc[splitBoundary[1]:, 2:]
    Y_test = data.iloc[splitBoundary[1]:, 1]
    return X_known, Y_known, X_unknown, Y_unknown, X_test, Y_test

def getPI(known: tuple, unknown: tuple, index: int, data: pd.core.frame.DataFrame):
    X_known, Y_known = known
    X_unknown, Y_unknown = unknown
    X_unknown = X_unknown.drop(index)
    Y_unknown = Y_unknown.drop(index)
    X_known = X_known.append(data.loc[index][2:])
    Y_known[index] = data.loc[index][1]
    return X_known, Y_known, X_unknown, Y_unknown